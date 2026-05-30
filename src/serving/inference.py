"""CPU inference adapter: a request-friendly wrapper over :mod:`src.inference`.

The serving layer never reloads the checkpoint per request. A single
:class:`InferenceEngine` loads the model once (lazily, behind a lock) and
every request reuses it. ``run_inference`` is the one function the FastAPI
endpoint and the Gradio UI both call.

Design notes
------------
* **CPU-only.** The engine forces ``device="cpu"`` regardless of host.
* **Autoregressive decoding** (matching ``scripts/SLURM/04_inference.sh``):
  the decoder runs once per residue, so latency scales with structure size.
  The :data:`DEFAULT_MAX_RESIDUES` cap is the real protection against a slow
  request hogging a worker — oversized structures are rejected *before* any
  decode begins (see :class:`InputTooLargeError`).
* **Confidence is surfaced, not recalibrated.** Per-residue confidence is the
  softmax max-probability per position; the aggregate is the model's existing
  LigandMPNN-style ``overall_confidence`` averaged over samples.
"""
from __future__ import annotations

import logging
import math
import os
import tempfile
import time
from pathlib import Path
from threading import Lock

import torch
from omegaconf import OmegaConf

from src.benchmarks.metrics import recovery_rate
from src.inference.constraints import DesignConstraints, as_token_ids
from src.inference.decoding import autoregressive_design, score_sequence
from src.inference.session import InferenceSession
from src.inference.weights import resolve_checkpoint
from src.serving.schemas import InferenceResult, ScorePosition, ScoreResult
from src.utils.io import ID_TO_AA, ids_to_sequence

logger = logging.getLogger(__name__)

# Live-endpoint residue cap. Autoregressive decoding is O(L) decoder passes;
# on a 2-vCPU CPU box a few hundred residues is tens of seconds. Override via
# the UMA_MAX_RESIDUES env var. The bundled examples are served from precomputed
# results so the cap doesn't limit the demo experience.
DEFAULT_MAX_RESIDUES = int(os.environ.get("UMA_MAX_RESIDUES", "120"))


def _project_root() -> Path:
    """Repo root, derived from this file's location (``src/serving/inference.py``)."""
    return Path(__file__).resolve().parents[2]


def _default_config_path() -> Path:
    """Hydra config matching the checkpoint architecture (self-contained YAML)."""
    env = os.environ.get("UMA_CONFIG_PATH")
    return Path(env) if env else _project_root() / "configs" / "config.yaml"


# Featurization flags that must match the checkpoint's training config or the
# model receives wrong inputs (these live in cfg.data and drive load_structure).
_FEATURIZER_FLAGS = (
    "ligand_featurizer",
    "residue_anchor",
    "pair_distance_atoms",
    "pair_distance_atoms_ligand",
    "frame_relative_angles",
)


def _checkpoint_matched_config(cfg_path: Path, ckpt_path: Path):
    """Return a config whose model architecture matches the checkpoint.

    Lightning checkpoints embed the exact ``model_config`` they were trained
    with under ``hyper_parameters``. We take the base config from ``cfg_path``
    (for data params like cutoff/ligand_context_atoms) but overlay the model
    architecture and featurizer flags from the checkpoint — so a drifting
    ``configs/config.yaml`` can't desync the served model from its weights.
    """
    cfg = OmegaConf.load(cfg_path)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 — fall back to the YAML on any load error
        logger.warning("could not read checkpoint hyper_parameters (%s); using config.yaml", exc)
        return cfg

    model_config = None
    if isinstance(ckpt, dict):
        model_config = (ckpt.get("hyper_parameters") or {}).get("model_config")

    if not model_config:
        logger.warning("checkpoint has no embedded model_config; using config.yaml as-is")
        return cfg

    mc = OmegaConf.create(dict(model_config))
    cfg.model = mc  # exact training architecture
    # Mirror the featurizer flags into cfg.data so featurization matches too.
    if "data" not in cfg:
        cfg.data = {}
    for flag in _FEATURIZER_FLAGS:
        if flag in mc:
            cfg.data[flag] = mc[flag]
    logger.info(
        "using checkpoint-embedded architecture (ligand_featurizer=%s, "
        "pair_distance_atoms=%s, frame_relative_angles=%s)",
        mc.get("ligand_featurizer"),
        mc.get("pair_distance_atoms"),
        mc.get("frame_relative_angles"),
    )
    return cfg


class InputTooLargeError(ValueError):
    """Raised when a structure exceeds the serving residue cap (→ HTTP 413)."""

    def __init__(self, n_residues: int, max_residues: int) -> None:
        self.n_residues = n_residues
        self.max_residues = max_residues
        super().__init__(
            f"structure has {n_residues} residues, which exceeds the serving cap "
            f"of {max_residues}. This is a CPU demo — design smaller structures "
            f"or run the model on a GPU for larger ones."
        )


class InferenceEngine:
    """Holds the loaded model; encodes + decodes structures on CPU.

    Construct via :func:`get_engine` (process-wide singleton). The constructor
    loads the checkpoint immediately and records :attr:`model_load_seconds`.
    """

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        checkpoint: str | Path | None = None,
        max_residues: int = DEFAULT_MAX_RESIDUES,
    ) -> None:
        self.max_residues = max_residues
        cfg_path = Path(config_path) if config_path is not None else _default_config_path()

        load_start = time.perf_counter()
        # resolve_checkpoint(None) lazily fetches the canonical weights from
        # Hugging Face into the user cache; an explicit path is honoured as-is.
        ckpt_path = resolve_checkpoint(Path(checkpoint) if checkpoint is not None else None)
        logger.info("loading UMA-Inverse checkpoint on CPU: %s", ckpt_path)

        # Build the model from the architecture the checkpoint was TRAINED with
        # (embedded in its Lightning hyper_parameters) rather than trusting
        # configs/config.yaml, which drifts across model versions. Mismatched
        # architecture/featurizer flags load silently via strict=False and yield
        # a degenerate model — so we make the served model self-describing.
        matched_cfg = _checkpoint_matched_config(cfg_path, ckpt_path)
        cfg_file = tempfile.NamedTemporaryFile(  # noqa: SIM115 — managed in finally
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        )
        try:
            OmegaConf.save(matched_cfg, cfg_file.name)
            cfg_file.close()
            self.session = InferenceSession.from_checkpoint(
                config_path=cfg_file.name,
                checkpoint=ckpt_path,
                device="cpu",
            )
        finally:
            try:
                os.unlink(cfg_file.name)
            except OSError:
                pass

        self.model_load_seconds = time.perf_counter() - load_start
        self.checkpoint_path = str(ckpt_path)
        logger.info("model loaded in %.2fs", self.model_load_seconds)

    def run(
        self,
        pdb_str: str,
        *,
        ligand: str | None = None,
        temperature: float = 0.1,
        n_samples: int = 1,
        seed: int | None = None,
    ) -> InferenceResult:
        """Design ``n_samples`` sequences for one structure. See :func:`run_inference`."""
        # ``load_structure`` reads a file path, not a string, so persist the
        # posted PDB to a temp file for the duration of the call.
        tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 — managed in finally
            mode="w", suffix=".pdb", delete=False, encoding="utf-8"
        )
        try:
            tmp.write(pdb_str)
            tmp.close()

            # Use a generous node budget so the parser doesn't silently crop;
            # the residue cap below is the explicit, user-facing guard.
            ctx = self.session.load_structure(
                pdb_path=tmp.name,
                max_total_nodes=max(self.max_residues * 4, 1024),
            )
            if ctx.residue_count > self.max_residues:
                raise InputTooLargeError(ctx.residue_count, self.max_residues)

            resolved = DesignConstraints.from_cli().resolve(ctx)

            start = time.perf_counter()
            samples = autoregressive_design(
                session=self.session,
                ctx=ctx,
                constraints=resolved,
                num_samples=n_samples,
                temperature=temperature,
                seed=seed,
            )
            inference_ms = (time.perf_counter() - start) * 1000.0

            designable = resolved.designable_mask.bool().cpu()
            sequences: list[str] = []
            per_residue_confidence: list[list[float]] = []
            overall_confidences: list[float] = []
            for sample in samples:
                sequences.append(_ids_to_sequence(sample.token_ids.tolist()))
                # Softmax max-probability per position (post-temperature
                # distribution the sample was drawn from).
                max_probs = sample.probs_full.max(dim=-1).values
                per_residue_confidence.append([round(float(p), 4) for p in max_probs.tolist()])
                overall_confidences.append(sample.overall_confidence(designable))

            mean_confidence = (
                sum(overall_confidences) / len(overall_confidences) if overall_confidences else 1.0
            )

            return InferenceResult(
                sequences=sequences,
                per_residue_confidence=per_residue_confidence,
                mean_confidence=round(mean_confidence, 4),
                n_residues=ctx.residue_count,
                inference_ms=round(inference_ms, 2),
            )
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def score(
        self,
        pdb_str: str,
        *,
        sequence: str | None = None,
        mode: str = "autoregressive",
        use_sequence: bool = True,
        num_batches: int = 10,
        seed: int | None = None,
    ) -> ScoreResult:
        """Score a structure's sequence under the model. See :func:`score_inference`."""
        tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 — managed in finally
            mode="w", suffix=".pdb", delete=False, encoding="utf-8"
        )
        try:
            tmp.write(pdb_str)
            tmp.close()

            ctx = self.session.load_structure(
                pdb_path=tmp.name,
                max_total_nodes=max(self.max_residues * 4, 1024),
            )
            if ctx.residue_count > self.max_residues:
                raise InputTooLargeError(ctx.residue_count, self.max_residues)

            seq_tensor = None
            if sequence is not None:
                try:
                    ids = as_token_ids(sequence.strip())
                except Exception as exc:  # ConstraintError etc. → 400
                    raise ValueError(f"invalid sequence: {exc}") from exc
                if len(ids) != ctx.residue_count:
                    raise ValueError(
                        f"sequence length {len(ids)} does not match the parsed residue "
                        f"count {ctx.residue_count}"
                    )
                seq_tensor = torch.tensor(ids, dtype=torch.long)

            start = time.perf_counter()
            result = score_sequence(
                session=self.session,
                ctx=ctx,
                sequence=seq_tensor,
                mode=mode,  # type: ignore[arg-type]
                use_sequence=use_sequence,
                num_batches=num_batches,
                seed=seed,
                return_distribution=True,
            )
            inference_ms = (time.perf_counter() - start) * 1000.0

            scored = result.sequence  # [L] token ids
            log_probs = result.log_probs  # [L]
            full = result.full_log_probs  # [L, 21] log-probs (X col masked to -inf)
            top_tokens = full.argmax(dim=-1)  # [L] model's preferred residue

            # Aggregate over non-X positions (X = unknown/unscoreable).
            non_x = scored != 20
            mean_lp = result.mean_log_prob(non_x)
            perplexity = math.exp(-mean_lp) if non_x.any() else 1.0
            recovery = recovery_rate(top_tokens, scored)  # excludes X internally

            positions: list[ScorePosition] = []
            for i in range(ctx.residue_count):
                tok = int(scored[i].item())
                top_tok = int(top_tokens[i].item())
                lp = float(log_probs[i].item())
                # An 'X' (unknown) residue is masked to -inf; floor it so the
                # response stays valid JSON (no Infinity).
                if not math.isfinite(lp):
                    lp = -30.0
                positions.append(
                    ScorePosition(
                        position=i,
                        residue_id=ctx.residue_ids[i],
                        aa=ID_TO_AA.get(tok, "X"),
                        log_prob=round(lp, 4),
                        prob=round(math.exp(lp), 4),
                        top_aa=ID_TO_AA.get(top_tok, "X"),
                        top_prob=round(math.exp(float(full[i, top_tok].item())), 4),
                    )
                )

            return ScoreResult(
                positions=positions,
                mean_log_prob=round(mean_lp, 4),
                perplexity=round(perplexity, 4),
                recovery=round(recovery, 4),
                n_residues=ctx.residue_count,
                mode=mode,
                use_sequence=use_sequence,
                num_batches=num_batches if mode == "autoregressive" else 1,
                sequence_scored=_ids_to_sequence(scored.tolist()),
                inference_ms=round(inference_ms, 2),
            )
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def warm_up(self, pdb_path: str | Path) -> None:
        """Run one inference so the first real request isn't cold.

        Safe to call with any small structure; failures are logged, not raised,
        so a missing warm-up fixture never blocks startup.
        """
        try:
            pdb_text = Path(pdb_path).read_text(encoding="utf-8")
            self.run(pdb_text, n_samples=1, temperature=0.1, seed=0)
            logger.info("warm-up inference complete (%s)", pdb_path)
        except Exception as exc:  # noqa: BLE001 — warm-up is best-effort
            logger.warning("warm-up skipped: %s", exc)


# ── Process-wide singleton ─────────────────────────────────────────────────────

_engine: InferenceEngine | None = None
_engine_lock = Lock()


def get_engine() -> InferenceEngine:
    """Return the lazily-initialised, process-wide :class:`InferenceEngine`."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = InferenceEngine()
    return _engine


def run_inference(
    pdb_str: str,
    *,
    ligand: str | None = None,
    temperature: float = 0.1,
    n_samples: int = 1,
    seed: int | None = None,
) -> InferenceResult:
    """Design sequences for a PDB structure on CPU.

    Args:
        pdb_str: Full PDB file contents (ligand atoms read from HETATM records).
        ligand: Reserved for forward-compat; does not inject a separate ligand.
        temperature: Sampling temperature (0.0 = argmax).
        n_samples: Number of sequences to design.
        seed: Optional base RNG seed for reproducibility (sample ``i`` uses ``seed + i``).

    Returns:
        An :class:`~src.serving.schemas.InferenceResult`.

    Raises:
        InputTooLargeError: If the structure exceeds the residue cap.
        ValueError: If the PDB contains no parseable protein residues.
    """
    return get_engine().run(
        pdb_str, ligand=ligand, temperature=temperature, n_samples=n_samples, seed=seed
    )


def score_inference(
    pdb_str: str,
    *,
    sequence: str | None = None,
    mode: str = "autoregressive",
    use_sequence: bool = True,
    num_batches: int = 10,
    seed: int | None = None,
) -> ScoreResult:
    """Score a structure's sequence under the model on CPU.

    Returns per-residue log-probabilities, the model's preferred residue at each
    position (mutation candidates), and aggregate perplexity + recovery.

    Args:
        pdb_str: Full PDB file contents.
        sequence: Optional one-letter AA sequence to score (must match the parsed
            residue count). Defaults to the structure's native sequence.
        mode: "autoregressive" (fast; num_batches passes) or "single-aa" (slower).
        use_sequence: If False, the decoder sees structure only (sequence masked).
        num_batches: Random decoding orders to average (autoregressive mode).
        seed: Optional RNG seed for the decoding-order sampling.

    Raises:
        InputTooLargeError: If the structure exceeds the residue cap.
        ValueError: If a provided sequence is invalid or length-mismatched.
    """
    return get_engine().score(
        pdb_str,
        sequence=sequence,
        mode=mode,
        use_sequence=use_sequence,
        num_batches=num_batches,
        seed=seed,
    )


def _ids_to_sequence(token_ids: list[int]) -> str:
    """Map AA token indices to a one-letter string (reuses the project map)."""
    return ids_to_sequence(token_ids)
