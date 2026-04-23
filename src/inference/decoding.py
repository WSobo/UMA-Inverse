"""Sampling and autoregressive decoding primitives.

This module exposes three levels of abstraction:

* :func:`sample_next` — apply temperature/top-p/bias/forbidden-mask to a
  logit tensor and return a sampled token. Pure, no model calls.
* :func:`autoregressive_design` — the main design loop. Runs the decoder
  one step per position in decoding order, handles batched samples, ties,
  and fixed residues. Returns a list of :class:`DesignSample` records.
* :func:`score_sequence` — compute log-likelihoods for the native sequence
  (or a provided sequence). Supports LigandMPNN's two scoring modes
  (``"autoregressive"`` averages over random orders; ``"single-aa"``
  masks positions one at a time).

Throughout, the convention follows LigandMPNN's internal alphabet
(``ACDEFGHIKLMNPQRSTVWY`` + ``X`` at index 20). Token 20 is never sampled
(``omit_global[20]=True`` is set by every :class:`ResolvedConstraints`).
"""
from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import torch

from src.inference.constraints import ResolvedConstraints
from src.inference.session import InferenceSession, StructureContext

logger = logging.getLogger(__name__)

ScoringMode = Literal["autoregressive", "single-aa"]


# ─── Sampling primitive ───────────────────────────────────────────────────────


def sample_next(
    logits: torch.Tensor,
    *,
    temperature: float = 0.1,
    top_p: float | None = None,
    bias: torch.Tensor | None = None,
    forbidden_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one token per row from a ``[B, 21]`` logit tensor.

    Args:
        logits: Raw decoder logits of shape ``[B, 21]``.
        temperature: Sampling temperature. ``0.0`` selects the argmax
            token; positive values divide the logits before softmax. Very
            small values approximate argmax while still producing a valid
            probability distribution.
        top_p: Optional nucleus (Holtzman) threshold in ``(0, 1]``. Only
            tokens whose cumulative probability reaches ``top_p`` are
            retained; the rest are masked before sampling. ``None`` (the
            default) disables nucleus filtering.
        bias: Optional additive bias ``[B, 21]`` or ``[21]``. Applied to
            the logits *before* temperature scaling and masking.
        forbidden_mask: Optional bool tensor (``[B, 21]`` or ``[21]``)
            marking tokens that must never be sampled.
        generator: Optional ``torch.Generator`` for reproducibility.

    Returns:
        ``(tokens, probs)`` where ``tokens`` is an int64 ``[B]`` of sampled
        indices and ``probs`` is the ``[B, 21]`` distribution the tokens
        were drawn from (after all transformations). ``probs`` sums to 1
        row-wise (except when every token is forbidden, in which case the
        row is uniform — we fall back rather than raise, because tie
        handling occasionally produces fully-masked rows for non-primary
        members).
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D [B, K]; got {logits.shape}")
    B, K = logits.shape

    work = logits.clone()
    if bias is not None:
        work = work + bias

    if forbidden_mask is not None:
        work = work.masked_fill(forbidden_mask, float("-inf"))

    if temperature <= 0.0:
        # Pure argmax
        probs = torch.zeros_like(work)
        tokens = work.argmax(dim=-1)
        probs.scatter_(1, tokens.unsqueeze(-1), 1.0)
        return tokens, probs

    scaled = work / temperature

    if top_p is not None:
        scaled = _apply_top_p(scaled, top_p=float(top_p))

    # Detect rows where every entry is -inf (fully forbidden) before softmax,
    # since softmax on such rows produces NaN which then poisons multinomial.
    all_inf = torch.isinf(scaled).all(dim=-1) & (scaled < 0).any(dim=-1)
    if all_inf.any():
        fallback_logits = torch.zeros_like(scaled)
        scaled = torch.where(all_inf.unsqueeze(-1), fallback_logits, scaled)

    probs = torch.softmax(scaled, dim=-1)

    tokens = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    return tokens, probs


def _apply_top_p(logits: torch.Tensor, *, top_p: float) -> torch.Tensor:
    """Mask tokens outside the top-p nucleus with ``-inf``.

    Preserves the original logit magnitudes for kept tokens so the softmax
    distribution after this filter is renormalised correctly. Always keeps
    at least one token per row (the argmax) to guarantee a valid output.
    """
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1]; got {top_p}")

    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    keep = cum <= top_p
    # Always keep the argmax — the shift below ensures element 0 is kept
    keep[..., 0] = True

    discard_sorted = ~keep
    discard = torch.zeros_like(logits, dtype=torch.bool)
    discard.scatter_(-1, sorted_indices, discard_sorted)
    return logits.masked_fill(discard, float("-inf"))


# ─── Design sample record ─────────────────────────────────────────────────────


@dataclass
class DesignSample:
    """One produced sequence plus the per-position statistics behind it.

    Attributes:
        token_ids: ``[L]`` int64 — sampled AA token for each residue.
        log_probs: ``[L]`` float — log-probability of the sampled token at
            each position (0.0 at fixed positions).
        probs_full: ``[L, 21]`` float — full post-temperature/top-p/bias/omit
            distribution each sample was drawn from. Saved so downstream
            confidence, entropy, and top-k reporting don't need to re-run
            the decoder.
        decoding_order: ``[L]`` int64 — the order each position was visited
            (0 = decoded first). Useful for debugging and scoring modes.
        seed: The ``torch.Generator`` seed that produced this sample, for
            reproducibility when the caller wants to replay one sample.
        temperature, top_p: Echoed back for provenance.
    """

    token_ids: torch.Tensor
    log_probs: torch.Tensor
    probs_full: torch.Tensor
    decoding_order: torch.Tensor
    seed: int
    temperature: float
    top_p: float | None = None
    tied_groups: list[list[int]] = field(default_factory=list)

    def overall_confidence(self, designable_mask: torch.Tensor) -> float:
        """Mean probability of the sampled token across designable positions.

        Matches LigandMPNN's ``overall_confidence`` definition:
        ``exp(mean(log_probs over redesigned residues))``. Higher = more
        confident. Range ``(0, 1]``. Returns ``1.0`` when there are zero
        designable positions (nothing to be uncertain about).
        """
        mask = designable_mask.bool()
        if not mask.any():
            return 1.0
        mean_logp = self.log_probs[mask].mean().item()
        return math.exp(mean_logp)

    def ligand_confidence(
        self,
        designable_mask: torch.Tensor,
        ligand_neighbour_mask: torch.Tensor,
    ) -> float:
        """Same as :meth:`overall_confidence` but restricted to ligand-neighbour residues.

        Matches LigandMPNN's ``ligand_confidence`` — takes only positions
        that are both designable and flagged as proximal to the ligand.
        """
        mask = designable_mask.bool() & ligand_neighbour_mask.bool()
        if not mask.any():
            return 0.0
        mean_logp = self.log_probs[mask].mean().item()
        return math.exp(mean_logp)


# ─── Autoregressive design ────────────────────────────────────────────────────


@torch.no_grad()
def autoregressive_design(
    session: InferenceSession,
    ctx: StructureContext,
    constraints: ResolvedConstraints,
    *,
    num_samples: int = 1,
    batch_size: int = 1,
    temperature: float = 0.1,
    top_p: float | None = None,
    seed: int | None = None,
    decoding_order: Literal["random", "left-to-right"] = "random",
) -> list[DesignSample]:
    """Sample one or more sequences autoregressively from a loaded structure.

    The encoder output in ``ctx`` is re-used across every sample — only
    the decoder runs per position.

    Args:
        session: Session holding the loaded model.
        ctx: Structure context from :meth:`InferenceSession.load_structure`.
        constraints: Indexed constraints from :meth:`DesignConstraints.resolve`.
        num_samples: Total number of sequences to produce.
        batch_size: How many sequences to decode in parallel. Set smaller
            than ``num_samples`` to trade speed for GPU memory.
        temperature: Sampling temperature (see :func:`sample_next`).
        top_p: Optional nucleus threshold (see :func:`sample_next`).
        seed: Base seed. When provided, sample *i* uses seed ``seed + i``
            so each sequence is independently reproducible.
        decoding_order: ``"random"`` (recommended, matches LigandMPNN) or
            ``"left-to-right"`` (useful for debugging).

    Returns:
        A list of ``num_samples`` :class:`DesignSample` records, in the
        order they were produced.
    """
    device = ctx.device
    model = session.model
    L = ctx.residue_count

    # "Known" (fixed + non-designable) residues hold native values; designable
    # positions are sampled. The known/designable split is recomputed inside
    # _decode_batch from `constraints.designable_mask`.
    tie_assignments, tie_weights_per_pos = _build_tie_tables(constraints, L, device)

    base_seed = seed if seed is not None else torch.randint(0, 2**31 - 1, (1,)).item()

    samples: list[DesignSample] = []
    for start in range(0, num_samples, batch_size):
        B = min(batch_size, num_samples - start)
        batch_seeds = [int(base_seed + start + i) for i in range(B)]

        pred_tokens, log_probs, probs_full, decoding_order_tensor = _decode_batch(
            model=model,
            ctx=ctx,
            constraints=constraints,
            tie_assignments=tie_assignments,
            tie_weights_per_pos=tie_weights_per_pos,
            batch_seeds=batch_seeds,
            temperature=temperature,
            top_p=top_p,
            decoding_order=decoding_order,
        )

        for i in range(B):
            samples.append(
                DesignSample(
                    token_ids=pred_tokens[i].cpu(),
                    log_probs=log_probs[i].cpu(),
                    probs_full=probs_full[i].cpu(),
                    decoding_order=decoding_order_tensor[i].cpu(),
                    seed=batch_seeds[i],
                    temperature=temperature,
                    top_p=top_p,
                    tied_groups=[list(group) for group, _ in constraints.ties],
                )
            )

    return samples


def _build_tie_tables(
    constraints: ResolvedConstraints, L: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-position tie-group id (-1 = untied) and per-position weight.

    Used by the decoder to (1) detect when a position is a tie member and
    (2) weight that position's logits when aggregating the group's decision.
    """
    group_id = torch.full((L,), -1, dtype=torch.long, device=device)
    weights = torch.zeros(L, device=device)
    for gi, (indices, group_weights) in enumerate(constraints.ties):
        for idx, w in zip(indices, group_weights):
            if group_id[idx] != -1:
                raise ValueError(
                    f"residue index {idx} belongs to multiple tie groups — "
                    "each residue may appear in at most one --tie group"
                )
            group_id[idx] = gi
            weights[idx] = w
    return group_id, weights


def _decode_batch(
    *,
    model: torch.nn.Module,
    ctx: StructureContext,
    constraints: ResolvedConstraints,
    tie_assignments: torch.Tensor,
    tie_weights_per_pos: torch.Tensor,
    batch_seeds: Sequence[int],
    temperature: float,
    top_p: float | None,
    decoding_order: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one batched forward decode. Internal; see :func:`autoregressive_design`."""
    device = ctx.device
    L = ctx.residue_count
    B = len(batch_seeds)

    z = ctx.z.expand(B, -1, -1, -1).contiguous()
    node_repr_res = ctx.node_repr_res.expand(B, -1, -1).contiguous()
    lig_ctx = ctx.lig_ctx.expand(B, -1, -1).contiguous()
    residue_mask = ctx.residue_mask.expand(B, -1)
    native = ctx.native_sequence  # [L]

    fixed_mask = constraints.fixed_mask  # [L]
    designable_mask = constraints.designable_mask  # [L]
    bias_global = constraints.bias_global  # [21]
    bias_per_pos = constraints.bias_per_residue  # [L, 21]
    omit_global = constraints.omit_global  # [21]
    omit_per_pos = constraints.omit_per_residue  # [L, 21]

    # Build decoding order per sample: known positions first, then the
    # remaining (designable) positions in random (or left-to-right) order.
    known_mask = ~designable_mask
    known_indices = torch.nonzero(known_mask, as_tuple=False).squeeze(-1)
    designable_indices = torch.nonzero(designable_mask, as_tuple=False).squeeze(-1)
    num_designable = designable_indices.numel()

    decoding_order_tensor = torch.zeros(B, L, dtype=torch.long, device=device)
    sample_order_lists: list[torch.Tensor] = []
    for b, s in enumerate(batch_seeds):
        if decoding_order == "random" and num_designable > 0:
            g = torch.Generator(device="cpu").manual_seed(int(s))
            perm = torch.randperm(num_designable, generator=g).to(device)
            ordered_designable = designable_indices[perm]
        else:
            ordered_designable = designable_indices
        ordered_all = torch.cat([known_indices, ordered_designable])
        # decoding_order[b, position_index] = rank at which that position decodes
        ranks = torch.empty(L, dtype=torch.long, device=device)
        ranks[ordered_all] = torch.arange(L, device=device)
        decoding_order_tensor[b] = ranks
        sample_order_lists.append(ordered_designable)

    # Initialise predicted sequence: known positions filled with native,
    # designable positions held at X (token 20) until decoded.
    pred_tokens = torch.full((B, L), fill_value=20, dtype=torch.long, device=device)
    pred_tokens[:, known_mask] = native[known_mask].unsqueeze(0).expand(B, -1)

    log_probs = torch.zeros(B, L, device=device)
    probs_full = torch.zeros(B, L, 21, device=device)
    probs_full[:, :, 20] = 1.0  # placeholder; overwritten on decode

    # Track which tie groups have been sampled this batch (per sample)
    tie_done_by_sample = [set() for _ in range(B)]

    # Per-sample RNG so seeds drive reproducibility through sampling too
    # (CPU generator; multinomial uses device copy internally for GPU probs).
    generators = [torch.Generator(device="cpu").manual_seed(int(s) + 7919) for s in batch_seeds]

    # Iterate over positions in decoding-rank order. Since order can differ
    # across samples in the batch, we iterate by rank (0..L-1) and gather
    # each sample's "position at this rank" individually. This keeps a
    # shared forward pass per rank even though positions differ per sample.
    rank_to_position_per_sample = torch.stack([sample_order_lists[b] for b in range(B)])  # [B, num_designable]

    # Pre-inverse: at each rank (for designable phase), pick the position
    # each sample is decoding.  Known phase is a no-op — positions already
    # filled, nothing to sample.

    for step in range(num_designable):
        # Current position per sample
        current_positions = rank_to_position_per_sample[:, step]  # [B]

        # Run the model's autoregressive context + decoder forward once
        # with the *current* partial sequence. We leverage the decoding
        # order we already built so the causal mask reveals exactly the
        # positions already decoded to each residue under construction.
        ar_ctx = model._autoregressive_context(
            z=z,
            sequence=pred_tokens,
            residue_mask=residue_mask,
            decoding_order=decoding_order_tensor,
        )  # [B, L, node_dim]
        decoder_input = torch.cat([node_repr_res, ar_ctx, lig_ctx], dim=-1)
        logits = model.decoder(decoder_input)  # [B, L, 21]

        # For each sample, gather logits at the designable positions that
        # share its current tie group (or just the current position when
        # untied), compute the weighted logit, apply bias + omit, sample.
        step_tokens = torch.empty(B, dtype=torch.long, device=device)
        step_log_probs = torch.empty(B, device=device)
        step_probs = torch.empty(B, 21, device=device)

        for b in range(B):
            pos = int(current_positions[b].item())
            gid = int(tie_assignments[pos].item())

            if gid >= 0:
                if gid in tie_done_by_sample[b]:
                    # Already sampled earlier in this batch — fill from the
                    # existing token and zero this step's contribution to
                    # log-prob (no extra decision was made).
                    tok = pred_tokens[b, pos].item()
                    step_tokens[b] = int(tok)
                    step_log_probs[b] = 0.0
                    step_probs[b] = 0.0
                    step_probs[b, int(tok)] = 1.0
                    continue

                member_indices, _ = constraints.ties[gid]
                member_tensor = torch.tensor(member_indices, device=device)
                member_weights = tie_weights_per_pos[member_tensor]
                member_logits = logits[b, member_tensor, :]  # [M, 21]
                merged = (member_weights.unsqueeze(-1) * member_logits).sum(dim=0, keepdim=True)  # [1, 21]

                bias = bias_global + bias_per_pos[pos]  # [21]
                forbidden = omit_global | omit_per_pos[pos]

                tok, probs = sample_next(
                    merged,
                    temperature=temperature,
                    top_p=top_p,
                    bias=bias.unsqueeze(0),
                    forbidden_mask=forbidden.unsqueeze(0),
                    generator=generators[b],
                )
                tok_i = int(tok.item())
                step_tokens[b] = tok_i
                step_log_probs[b] = torch.log(probs[0, tok_i].clamp_min(1e-30))
                step_probs[b] = probs[0]

                # Apply the sampled token to every member of the tie group
                for m in member_indices:
                    pred_tokens[b, m] = tok_i
                    probs_full[b, m] = probs[0]
                    log_probs[b, m] = step_log_probs[b]
                tie_done_by_sample[b].add(gid)

            else:
                pos_logits = logits[b, pos : pos + 1, :]  # [1, 21]
                bias = bias_global + bias_per_pos[pos]
                forbidden = omit_global | omit_per_pos[pos]
                tok, probs = sample_next(
                    pos_logits,
                    temperature=temperature,
                    top_p=top_p,
                    bias=bias.unsqueeze(0),
                    forbidden_mask=forbidden.unsqueeze(0),
                    generator=generators[b],
                )
                tok_i = int(tok.item())
                step_tokens[b] = tok_i
                step_log_probs[b] = torch.log(probs[0, tok_i].clamp_min(1e-30))
                step_probs[b] = probs[0]
                pred_tokens[b, pos] = tok_i
                probs_full[b, pos] = probs[0]
                log_probs[b, pos] = step_log_probs[b]

    # Fill probs_full for fixed residues with a delta at the native token
    # so confidence calculations and per-residue JSON stay well-defined.
    fixed_indices = torch.nonzero(fixed_mask, as_tuple=False).squeeze(-1)
    for idx in fixed_indices.tolist():
        tok = int(native[idx].item())
        probs_full[:, idx, :] = 0.0
        probs_full[:, idx, tok] = 1.0
        log_probs[:, idx] = 0.0

    return pred_tokens, log_probs, probs_full, decoding_order_tensor


# ─── Scoring ──────────────────────────────────────────────────────────────────


@dataclass
class ScoreResult:
    """Per-position log-likelihoods for a given sequence.

    Attributes:
        sequence: ``[L]`` int64 — the scored token ids.
        log_probs: ``[L]`` float — log-probability of each token under the
            chosen scoring mode.
        mode: ``"autoregressive"`` or ``"single-aa"``.
        use_sequence: Whether the scored sequence was provided to the AR
            decoder (``True``) or masked out as all-X (``False``).
        num_batches: How many random decoding orders were averaged (AR
            mode only; always ``1`` for single-aa).
    """

    sequence: torch.Tensor
    log_probs: torch.Tensor
    mode: ScoringMode
    use_sequence: bool
    num_batches: int = 1

    def mean_log_prob(self, mask: torch.Tensor | None = None) -> float:
        lp = self.log_probs if mask is None else self.log_probs[mask.bool()]
        if lp.numel() == 0:
            return 0.0
        return lp.mean().item()


@torch.no_grad()
def score_sequence(
    session: InferenceSession,
    ctx: StructureContext,
    *,
    sequence: torch.Tensor | None = None,
    mode: ScoringMode = "autoregressive",
    use_sequence: bool = True,
    num_batches: int = 10,
    seed: int | None = None,
) -> ScoreResult:
    """Score a sequence under the trained model.

    Args:
        session: Session with a loaded checkpoint.
        ctx: Structure context for the PDB to score against.
        sequence: Optional token-id sequence ``[L]``. Defaults to
            ``ctx.native_sequence``.
        mode: ``"autoregressive"`` computes ``p(AA_t | structure, AA_{<t})``
            averaged over ``num_batches`` random decoding orders.
            ``"single-aa"`` computes ``p(AA_t | structure, AA_{all except t})``
            for each position (one forward pass per position, no averaging).
        use_sequence: When ``False``, the decoder only sees structural
            context (sequence slot is all-X). Useful for measuring how much
            signal the sequence context adds.
        num_batches: Number of random decoding orders to average (AR mode).
        seed: Optional base seed; each random order uses ``seed + i``.

    Returns:
        A :class:`ScoreResult` with per-position log-probabilities.
    """
    device = ctx.device
    model = session.model

    scored_sequence = sequence if sequence is not None else ctx.native_sequence
    scored_sequence = scored_sequence.to(device)

    if mode == "autoregressive":
        log_probs = _score_autoregressive(
            model=model,
            ctx=ctx,
            scored_sequence=scored_sequence,
            use_sequence=use_sequence,
            num_batches=num_batches,
            seed=seed,
        )
        return ScoreResult(
            sequence=scored_sequence.cpu(),
            log_probs=log_probs.cpu(),
            mode=mode,
            use_sequence=use_sequence,
            num_batches=num_batches,
        )

    if mode == "single-aa":
        log_probs = _score_single_aa(
            model=model,
            ctx=ctx,
            scored_sequence=scored_sequence,
            use_sequence=use_sequence,
        )
        return ScoreResult(
            sequence=scored_sequence.cpu(),
            log_probs=log_probs.cpu(),
            mode=mode,
            use_sequence=use_sequence,
            num_batches=1,
        )

    raise ValueError(f"unknown scoring mode: {mode!r}")


def _score_autoregressive(
    *,
    model: torch.nn.Module,
    ctx: StructureContext,
    scored_sequence: torch.Tensor,
    use_sequence: bool,
    num_batches: int,
    seed: int | None,
) -> torch.Tensor:
    """Average per-position log-probs across random decoding orders."""
    device = ctx.device
    L = ctx.residue_count
    base_seed = seed if seed is not None else torch.randint(0, 2**31 - 1, (1,)).item()

    acc_log_probs = torch.zeros(L, device=device)
    for b in range(num_batches):
        g = torch.Generator(device="cpu").manual_seed(int(base_seed + b))
        perm = torch.randperm(L, generator=g).to(device)
        # decoding_order[pos_index] = rank at which that pos is decoded
        ranks = torch.empty(L, dtype=torch.long, device=device)
        ranks[perm] = torch.arange(L, device=device)

        seq_input = scored_sequence if use_sequence else torch.full_like(scored_sequence, 20)
        seq_batched = seq_input.unsqueeze(0)

        ar_ctx = model._autoregressive_context(
            z=ctx.z,
            sequence=seq_batched,
            residue_mask=ctx.residue_mask,
            decoding_order=ranks.unsqueeze(0),
        )
        decoder_input = torch.cat([ctx.node_repr_res, ar_ctx, ctx.lig_ctx], dim=-1)
        logits = model.decoder(decoder_input)[0]  # [L, 21]
        # Token 20 (X) is not a valid output — mask it before computing log-probs
        logits = logits.clone()
        logits[:, 20] = float("-inf")
        log_probs_all = torch.log_softmax(logits, dim=-1)
        acc_log_probs = acc_log_probs + log_probs_all.gather(
            1, scored_sequence.unsqueeze(-1)
        ).squeeze(-1)

    return acc_log_probs / float(num_batches)


def _score_single_aa(
    *,
    model: torch.nn.Module,
    ctx: StructureContext,
    scored_sequence: torch.Tensor,
    use_sequence: bool,
) -> torch.Tensor:
    """Score each position as if it were the only unknown residue.

    Runs one forward pass per position with that position masked and all
    other positions fed either the native sequence (``use_sequence=True``)
    or the all-X baseline. Decoding order is rigged so the scored position
    is the last to decode — giving it context from all others.
    """
    device = ctx.device
    L = ctx.residue_count

    log_probs_out = torch.zeros(L, device=device)
    for target in range(L):
        seq_input = scored_sequence.clone() if use_sequence else torch.full_like(scored_sequence, 20)
        seq_input[target] = 20  # mask target regardless of use_sequence
        seq_batched = seq_input.unsqueeze(0)

        ranks = torch.empty(L, dtype=torch.long, device=device)
        # Put target last; other positions fill in the remaining ranks
        others = torch.arange(L, device=device)
        others = others[others != target]
        ranks[others] = torch.arange(L - 1, device=device)
        ranks[target] = L - 1

        ar_ctx = model._autoregressive_context(
            z=ctx.z,
            sequence=seq_batched,
            residue_mask=ctx.residue_mask,
            decoding_order=ranks.unsqueeze(0),
        )
        decoder_input = torch.cat([ctx.node_repr_res, ar_ctx, ctx.lig_ctx], dim=-1)
        logits = model.decoder(decoder_input)[0]
        logits = logits.clone()
        logits[:, 20] = float("-inf")
        log_probs_all = torch.log_softmax(logits, dim=-1)
        tok_id = int(scored_sequence[target].item())
        log_probs_out[target] = log_probs_all[target, tok_id]

    return log_probs_out
