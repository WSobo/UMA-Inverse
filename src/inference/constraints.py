"""Design constraints and residue-ID parsing.

LigandMPNN exposes constraint state through eight separate flags that use
two different syntactic conventions (whitespace-delimited selectors for
residues, comma-delimited colon-separated pairs for biases) with subtle
rules for insertion codes. This module consolidates all of that into one
:class:`DesignConstraints` dataclass with a single :meth:`resolve` step
against a :class:`~src.inference.session.StructureContext`.

The resolver converts user-facing residue identifier strings like ``"A23"``
and ``"B42C"`` into tensor indices that match the decoding path, raising
:class:`ConstraintError` with a pointed message if any token fails to
resolve. Catching misconfiguration at resolve time (before sampling) is
the main reason this module exists.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from src.inference.session import StructureContext
from src.utils.io import ID_TO_AA

PathLike = str | Path

# Reverse map for CLI-facing single-letter AA codes
AA_TO_ID: dict[str, int] = {aa: idx for idx, aa in ID_TO_AA.items()}
# Exclude X (idx 20) from the set of AAs the user may bias or omit — it has
# no biological meaning and letting the user emit it would produce garbage.
_DESIGNABLE_AA: set[str] = {aa for aa in AA_TO_ID if aa != "X"}

_RESIDUE_ID_RE = re.compile(r"^(?P<chain>[A-Za-z])(?P<num>-?\d+)(?P<ins>[A-Za-z]?)$")


class ConstraintError(ValueError):
    """Raised when user-provided constraints cannot be parsed or resolved."""


# ─── Parsing helpers ──────────────────────────────────────────────────────────


def parse_residue_selection(selection: str | None) -> list[str]:
    """Split a whitespace- or comma-separated residue selector into tokens.

    Accepts either of LigandMPNN's two conventions::

        "A1 A2 A3"        # space-separated
        "A1,A2,A3"        # comma-separated (more robust in shell)

    Duplicate tokens are silently deduplicated while preserving order.
    Returns an empty list for ``None`` or whitespace-only input.
    """
    if not selection:
        return []
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in re.split(r"[\s,]+", selection.strip()):
        if not raw:
            continue
        if raw in seen:
            continue
        _validate_residue_id(raw)
        tokens.append(raw)
        seen.add(raw)
    return tokens


def _validate_residue_id(token: str) -> None:
    """Confirm ``token`` matches ``<chain><resnum>[<insertion>]``.

    Raises :class:`ConstraintError` with a pointed message otherwise.
    """
    if not _RESIDUE_ID_RE.match(token):
        raise ConstraintError(
            f"invalid residue id: {token!r} — expected <chain-letter><resnum>[<insertion-code>] "
            "(examples: 'A23', 'B42C', 'C-5')"
        )


def parse_aa_bias(bias: str | None) -> dict[int, float]:
    """Parse a global AA-bias expression like ``"W:3.0,A:-1.0"``.

    Returns:
        Mapping of AA token index (0..19) to logit offset. The bias is
        *added* to logits before softmax, matching LigandMPNN semantics.
    """
    if not bias:
        return {}
    out: dict[int, float] = {}
    for pair in bias.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ConstraintError(
                f"invalid bias pair: {pair!r} — expected '<AA>:<float>' (example: 'W:3.0')"
            )
        aa, value = pair.split(":", 1)
        aa = aa.strip().upper()
        try:
            offset = float(value.strip())
        except ValueError as exc:
            raise ConstraintError(f"invalid bias value for {aa!r}: {value!r}") from exc
        if aa not in _DESIGNABLE_AA:
            raise ConstraintError(
                f"invalid AA in bias: {aa!r} — must be one of {sorted(_DESIGNABLE_AA)}"
            )
        out[AA_TO_ID[aa]] = offset
    return out


def parse_aa_omit(omit: str | None) -> set[int]:
    """Parse a global omit expression — a concatenation of AA letters.

    Accepts either ``"CDFG"`` or ``"C,D,F,G"``. Whitespace is ignored.
    """
    if not omit:
        return set()
    cleaned = re.sub(r"[\s,]", "", omit).upper()
    bad = [c for c in cleaned if c not in _DESIGNABLE_AA]
    if bad:
        raise ConstraintError(
            f"invalid AA letters in omit: {sorted(set(bad))!r} — allowed: {sorted(_DESIGNABLE_AA)}"
        )
    return {AA_TO_ID[c] for c in cleaned}


def parse_tie_groups(
    ties: str | None,
    weights: str | None = None,
) -> list[tuple[list[str], list[float]]]:
    """Parse LigandMPNN-style symmetry ties.

    Args:
        ties: String of tied residue groups separated by ``|``, with each
            group containing comma-separated residue IDs. Example::

                "A1,A10,A20|B5,B15"

        weights: Optional per-group weights in the same ``|``/``,`` layout.
            When omitted, equal weights (``1/N`` within each group) are used.

    Returns:
        List of ``(residue_ids, weights)`` tuples — one entry per tied group.
    """
    if not ties:
        return []

    group_strings = [g.strip() for g in ties.split("|") if g.strip()]
    parsed_groups = [parse_residue_selection(g) for g in group_strings]

    if weights is None:
        return [(g, [1.0 / len(g)] * len(g)) for g in parsed_groups]

    weight_groups_raw = [w.strip() for w in weights.split("|") if w.strip()]
    if len(weight_groups_raw) != len(parsed_groups):
        raise ConstraintError(
            f"tie-weights group count ({len(weight_groups_raw)}) does not match "
            f"tie group count ({len(parsed_groups)})"
        )

    result: list[tuple[list[str], list[float]]] = []
    for grp_idx, (group, weight_spec) in enumerate(zip(parsed_groups, weight_groups_raw)):
        try:
            weights_list = [float(x.strip()) for x in weight_spec.split(",") if x.strip()]
        except ValueError as exc:
            raise ConstraintError(
                f"invalid tie weight in group {grp_idx}: {weight_spec!r}"
            ) from exc
        if len(weights_list) != len(group):
            raise ConstraintError(
                f"tie group {grp_idx} has {len(group)} residues but "
                f"{len(weights_list)} weights"
            )
        result.append((group, weights_list))
    return result


def load_per_residue_bias(path: PathLike) -> dict[str, dict[int, float]]:
    """Load a JSON file of shape ``{"A23": {"W": 3.0, "A": -1.0}, ...}``.

    Each outer key is a residue ID; each inner key is a one-letter AA code.
    """
    raw = _read_json(path)
    out: dict[str, dict[int, float]] = {}
    for res_id, entries in raw.items():
        _validate_residue_id(res_id)
        if not isinstance(entries, Mapping):
            raise ConstraintError(
                f"bias file: entry for {res_id!r} must be a mapping of AA → bias"
            )
        per_pos: dict[int, float] = {}
        for aa, value in entries.items():
            aa_up = aa.strip().upper()
            if aa_up not in _DESIGNABLE_AA:
                raise ConstraintError(
                    f"bias file: invalid AA {aa!r} for residue {res_id!r}"
                )
            try:
                per_pos[AA_TO_ID[aa_up]] = float(value)
            except (TypeError, ValueError) as exc:
                raise ConstraintError(
                    f"bias file: invalid value for {res_id!r}[{aa!r}]: {value!r}"
                ) from exc
        out[res_id] = per_pos
    return out


def load_per_residue_omit(path: PathLike) -> dict[str, set[int]]:
    """Load ``{"A23": "CDFG", ...}`` — per-residue omit lists from JSON."""
    raw = _read_json(path)
    out: dict[str, set[int]] = {}
    for res_id, letters in raw.items():
        _validate_residue_id(res_id)
        if not isinstance(letters, str):
            raise ConstraintError(
                f"omit file: value for {res_id!r} must be a string of AA letters"
            )
        out[res_id] = parse_aa_omit(letters)
    return out


def _read_json(path: PathLike) -> Mapping[str, object]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"constraints file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, Mapping):
        raise ConstraintError(f"{path}: expected a JSON object at the top level")
    return data


# ─── Constraint container ─────────────────────────────────────────────────────


@dataclass
class DesignConstraints:
    """User-specified sequence design constraints, ready to be resolved.

    Fields use chain-letter residue IDs as their canonical key — resolution
    against a specific structure happens via :meth:`resolve`.

    Attributes:
        fix: Residue IDs whose native AA must be preserved.
        redesign: Residue IDs the model is *allowed* to change. When ``None``
            every non-fixed residue is designable (the common case).
        design_chains: Chain letters whose residues should be redesigned.
            Independent from ``redesign``; both narrow the designable set.
        parse_chains: Chain letters to feed to the parser, for the subset of
            inputs where irrelevant chains should be excluded entirely. Not
            a design-time constraint; just passed through to the loader.
        bias: Global AA bias (token index → logit offset).
        bias_per_residue: Per-residue AA bias.
        omit: Global set of AA token indices to forbid.
        omit_per_residue: Per-residue set of forbidden AA token indices.
        ties: List of tied residue groups with per-residue weights.
        mask_ligand: Ablation flag forwarded to ``load_structure``.
        include_zero_occupancy: Parser flag forwarded to ``load_structure``.
        ligand_cutoff: Parser flag forwarded to ``load_structure``.
        max_total_nodes: Parser flag forwarded to ``load_structure``.
    """

    fix: set[str] = field(default_factory=set)
    redesign: set[str] | None = None
    design_chains: set[str] | None = None
    parse_chains: list[str] | None = None
    bias: dict[int, float] = field(default_factory=dict)
    bias_per_residue: dict[str, dict[int, float]] = field(default_factory=dict)
    omit: set[int] = field(default_factory=set)
    omit_per_residue: dict[str, set[int]] = field(default_factory=dict)
    ties: list[tuple[list[str], list[float]]] = field(default_factory=list)
    mask_ligand: bool = False
    include_zero_occupancy: bool = False
    ligand_cutoff: float | None = None
    max_total_nodes: int | None = None

    # ── Factories ────────────────────────────────────────────────────────────

    @classmethod
    def from_cli(
        cls,
        *,
        fix: str | None = None,
        redesign: str | None = None,
        design_chains: str | None = None,
        parse_chains: str | None = None,
        bias: str | None = None,
        bias_file: PathLike | None = None,
        omit: str | None = None,
        omit_file: PathLike | None = None,
        tie: str | None = None,
        tie_weights: str | None = None,
        mask_ligand: bool = False,
        include_zero_occupancy: bool = False,
        ligand_cutoff: float | None = None,
        max_total_nodes: int | None = None,
    ) -> DesignConstraints:
        """Build from raw CLI strings (every field optional)."""
        fix_set = set(parse_residue_selection(fix))
        redesign_set = set(parse_residue_selection(redesign)) if redesign else None
        conflicts = fix_set & (redesign_set or set())
        if conflicts:
            raise ConstraintError(
                f"residues appear in both --fix and --redesign: {sorted(conflicts)}"
            )

        design_chain_set = _parse_chain_list(design_chains)
        parse_chain_list = sorted(_parse_chain_list(parse_chains) or set()) or None

        bias_per_res = load_per_residue_bias(bias_file) if bias_file else {}
        omit_per_res = load_per_residue_omit(omit_file) if omit_file else {}

        return cls(
            fix=fix_set,
            redesign=redesign_set,
            design_chains=design_chain_set,
            parse_chains=parse_chain_list,
            bias=parse_aa_bias(bias),
            bias_per_residue=bias_per_res,
            omit=parse_aa_omit(omit),
            omit_per_residue=omit_per_res,
            ties=parse_tie_groups(tie, tie_weights),
            mask_ligand=mask_ligand,
            include_zero_occupancy=include_zero_occupancy,
            ligand_cutoff=ligand_cutoff,
            max_total_nodes=max_total_nodes,
        )

    # ── Resolution ───────────────────────────────────────────────────────────

    def resolve(self, ctx: StructureContext) -> ResolvedConstraints:
        """Map residue IDs to tensor indices and validate.

        Args:
            ctx: A loaded structure context from :meth:`InferenceSession.load_structure`.

        Returns:
            A :class:`ResolvedConstraints` whose tensors live on the same
            device as the context.

        Raises:
            ConstraintError: If any referenced residue ID is missing from
                the context, chains in ``design_chains`` don't exist, or a
                tie group references a residue outside the structure.
        """
        L = ctx.residue_count
        device = ctx.device

        fixed_mask = torch.zeros(L, dtype=torch.bool, device=device)
        for rid in self.fix:
            idx = _lookup_residue(rid, ctx, context="--fix")
            fixed_mask[idx] = True

        designable_mask = ctx.design_mask.clone()
        if self.redesign is not None:
            redesign_mask = torch.zeros(L, dtype=torch.bool, device=device)
            for rid in self.redesign:
                idx = _lookup_residue(rid, ctx, context="--redesign")
                redesign_mask[idx] = True
            designable_mask = redesign_mask

        if self.design_chains is not None:
            unknown = self.design_chains - set(ctx.chain_ids)
            if unknown:
                raise ConstraintError(
                    f"--design-chains references chains not present in PDB: {sorted(unknown)} "
                    f"(available: {sorted(set(ctx.chain_ids))})"
                )
            chain_mask = torch.tensor(
                [cid in self.design_chains for cid in ctx.chain_ids],
                dtype=torch.bool,
                device=device,
            )
            designable_mask = designable_mask & chain_mask

        # Fixed residues are never in the designable set
        designable_mask = designable_mask & ~fixed_mask

        bias_global = torch.zeros(21, device=device)
        for aa_idx, offset in self.bias.items():
            bias_global[aa_idx] = offset

        bias_per_pos = torch.zeros((L, 21), device=device)
        for rid, aa_biases in self.bias_per_residue.items():
            idx = _lookup_residue(rid, ctx, context="bias file")
            for aa_idx, offset in aa_biases.items():
                bias_per_pos[idx, aa_idx] = offset

        omit_global = torch.zeros(21, dtype=torch.bool, device=device)
        for aa_idx in self.omit:
            omit_global[aa_idx] = True

        omit_per_pos = torch.zeros((L, 21), dtype=torch.bool, device=device)
        for rid, aa_set in self.omit_per_residue.items():
            idx = _lookup_residue(rid, ctx, context="omit file")
            for aa_idx in aa_set:
                omit_per_pos[idx, aa_idx] = True

        # Token index 20 = "X": always forbidden from the output vocabulary
        omit_global[20] = True

        ties_indexed: list[tuple[list[int], list[float]]] = []
        for group_idx, (members, weights) in enumerate(self.ties):
            indices = [_lookup_residue(rid, ctx, context=f"tie group {group_idx}") for rid in members]
            ties_indexed.append((indices, list(weights)))

        return ResolvedConstraints(
            fixed_mask=fixed_mask,
            designable_mask=designable_mask,
            bias_global=bias_global,
            bias_per_residue=bias_per_pos,
            omit_global=omit_global,
            omit_per_residue=omit_per_pos,
            ties=ties_indexed,
        )


# ─── Resolved (indexed) constraints ───────────────────────────────────────────


@dataclass
class ResolvedConstraints:
    """Tensor-valued constraints ready for the decoder.

    All tensors live on the same device as the originating StructureContext.
    """

    fixed_mask: torch.Tensor            # [L] bool — positions locked to native
    designable_mask: torch.Tensor       # [L] bool — positions to redesign
    bias_global: torch.Tensor           # [21] float — global AA bias
    bias_per_residue: torch.Tensor      # [L, 21] float — per-position bias
    omit_global: torch.Tensor           # [21] bool — globally forbidden AAs
    omit_per_residue: torch.Tensor      # [L, 21] bool — per-position forbidden
    ties: list[tuple[list[int], list[float]]]  # grouped index lists + weights

    def logit_bias(self, residue_index: int) -> torch.Tensor:
        """Return the additive logit bias (``[21]``) for one position."""
        return self.bias_global + self.bias_per_residue[residue_index]

    def forbidden_mask(self, residue_index: int) -> torch.Tensor:
        """Return a boolean ``[21]`` mask of tokens to exclude at one position."""
        return self.omit_global | self.omit_per_residue[residue_index]


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _parse_chain_list(raw: str | None) -> set[str] | None:
    """Turn ``"A,B,C"`` into ``{"A","B","C"}`` — or ``None`` when raw is empty."""
    if not raw:
        return None
    chains = {c.strip() for c in re.split(r"[\s,]+", raw) if c.strip()}
    bad = [c for c in chains if not re.fullmatch(r"[A-Za-z]", c)]
    if bad:
        raise ConstraintError(f"invalid chain identifier(s): {bad!r} — must be single letters")
    return chains


def _lookup_residue(rid: str, ctx: StructureContext, *, context: str) -> int:
    """Look up a residue index or raise a descriptive error."""
    idx = ctx.residue_id_to_index.get(rid)
    if idx is None:
        hint = ""
        if len(rid) > 1 and rid[0] not in set(ctx.chain_ids):
            hint = f" (chain {rid[0]!r} not in PDB; present chains: {sorted(set(ctx.chain_ids))})"
        raise ConstraintError(
            f"{context}: residue {rid!r} not found in {ctx.pdb_path}{hint}"
        )
    return idx


# ─── Convenience re-exports ───────────────────────────────────────────────────


def as_token_ids(aa_letters: Iterable[str]) -> list[int]:
    """Convert a sequence of single-letter AA codes to token indices.

    Mirrors LigandMPNN's internal alphabet (``ACDEFGHIKLMNPQRSTVWY``).
    """
    out: list[int] = []
    for letter in aa_letters:
        letter = letter.strip().upper()
        if letter not in AA_TO_ID:
            raise ConstraintError(f"unknown AA letter: {letter!r}")
        out.append(AA_TO_ID[letter])
    return out


def as_aa_letters(token_ids: Sequence[int]) -> str:
    """Inverse of :func:`as_token_ids` — token indices to a string of letters."""
    return "".join(ID_TO_AA.get(int(idx), "X") for idx in token_ids)
