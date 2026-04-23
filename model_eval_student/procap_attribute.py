"""
Wrapper script to calculate DeepLIFT/SHAP attributions for ProCapNet-style
PyTorch models.

This is intended to mirror ProCapNet's interpretation logic in
`src/4_interpret_models/deepshap_utils.py` as closely as possible while still
supporting evaluating multiple models in one run (e.g. student + teachers).

Key semantics:
- Profile attribution target matches ProCapNet's ProfileModelWrapper:
  flatten both strands, mean-normalize logits, apply a *detached* softmax over
  the flattened profile, and return sum(mean_norm_logits * softmax_probs).
- Reverse-complement averaging is supported and enabled by default.
- By default, `output_fname` is the **ensemble mean across all provided models**
  (this matches ProCapNet's "merge across folds" convention). Per-model files
  are always written alongside it.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.io import extract_loci

from mpra_utils import (
    ensure_writable_paths,
    load_mpra_one_hot,
    read_fasta_records,
    validate_paired_fasta_headers,
)

# Make sure the training-time modules are importable when loading saved models.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
for extra_path in [
    PROJECT_ROOT / "src" / "2_train_models",
    PROJECT_ROOT / "src" / "4_interpret_models",
    PROJECT_ROOT / "src" / "utils",
]:
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))
import BPNet_strand_merged_umap


def reverse_complement_one_hot(X):
    # X shape: (B, 4, L) with channels ordered A,C,G,T
    # RC = reverse length axis and swap channels A<->T, C<->G
    return torch.flip(X[:, [3, 2, 1, 0], :], dims=[2])


def reverse_complement_attr(attr):
    # hypothetical attributions have same shape as input: (B, 4, L)
    return torch.flip(attr[:, [3, 2, 1, 0], :], dims=[2])


class ProCapNetProfileWrapper(torch.nn.Module):
    """
    Match ProCapNet's profile attribution target:
    flatten both strands together, mean-normalize logits,
    compute softmax over the flattened profile, and return
    the scalar sum(mean_norm_logits * softmax_probs).
    """

    def __init__(self, model, detach_softmax_probs: bool = True):
        super().__init__()
        self.model = model
        self.detach_softmax_probs = bool(detach_softmax_probs)
        # Register as a submodule so tangermeme can hook it if needed.
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, X, X_ctl=None, **kwargs):
        out = self.model(X, **kwargs)

        if isinstance(out, (tuple, list)):
            profile_logits = out[0]
        else:
            profile_logits = out

        # expected shape: (B, 2, L)
        profile_logits = profile_logits.reshape(profile_logits.shape[0], -1)
        mean_norm_logits = profile_logits - torch.mean(
            profile_logits, dim=-1, keepdim=True
        )
        logits_for_probs = mean_norm_logits.detach() if self.detach_softmax_probs else mean_norm_logits
        softmax_probs = self.softmax(logits_for_probs)
        # tangermeme expects (B, n_targets); keep a trailing dim of size 1.
        return (mean_norm_logits * softmax_probs).sum(dim=-1, keepdim=True)


class ProCapNetCountsWrapper(torch.nn.Module):
    """Match ProCapNet's counts attribution target (the model's log-count head)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X, X_ctl=None, **kwargs):
        out = self.model(X, **kwargs)
        if not isinstance(out, (tuple, list)) or len(out) < 2:
            raise ValueError(
                "Counts attribution expects model(X) -> (profile_logits, log_counts)."
            )
        log_counts = out[1]
        if log_counts.ndim == 1:
            log_counts = log_counts[:, None]
        return log_counts

class SingleInputAdapter(torch.nn.Module):
    """Adapter so bpnetlite wrappers can handle models without control inputs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X, X_ctl=None, **kwargs):
        return self.model(X, **kwargs)


def _load_bed_inputs(params: dict, log) -> tuple[torch.Tensor, dict]:
    if "loci" not in params:
        raise ValueError("BED input mode requires params['loci']")

    log(f"Reading loci from {params['loci']}")
    loci = pd.read_csv(
        params["loci"],
        sep="\t",
        usecols=[0, 1, 2],
        header=None,
        index_col=False,
        names=["chrom", "start", "end"],
        dtype={"chrom": str},
    )
    log(f"Loaded {len(loci)} loci")

    log("Extracting loci sequences to one-hot tensors")
    X = extract_loci(
        loci=loci,
        sequences=params["sequences"],
        in_window=params["in_window"],
        out_window=params["out_window"],
        chroms=params.get("chroms"),
        max_jitter=int(params.get("max_jitter", 0) or 0),
        verbose=params["verbose"],
        min_counts=None,
        max_counts=None,
        n_loci=params.get("n_loci"),
        ignore=list("QWERYUIOPSDFHJKLZXVBNMnN"),
    ).to(torch.float32)
    log(f"Extraction complete. Tensor shape: {tuple(X.shape)}")

    return X, {
        "input_mode": "bed",
        "input_record_count": int(X.shape[0]),
        "loci": params["loci"],
    }


def _load_mpra_fasta_inputs(params: dict, log) -> tuple[torch.Tensor, dict]:
    fasta_path = params.get("mpra_fasta")
    if fasta_path is None:
        raise ValueError("MPRA FASTA input mode requires params['mpra_fasta']")

    expected_len = int(params["in_window"])
    log(f"Reading MPRA FASTA records from {fasta_path}")
    headers, X_np = load_mpra_one_hot(
        fasta_path,
        expected_length=expected_len,
        max_records=params.get("n_loci"),
    )

    paired_fasta = params.get("paired_mpra_fasta")
    if paired_fasta is not None:
        log(f"Validating paired MPRA FASTA order against {paired_fasta}")
        paired_headers, paired_seqs = read_fasta_records(paired_fasta)
        if params.get("n_loci") is not None:
            limit = int(params["n_loci"])
            paired_headers = paired_headers[:limit]
            paired_seqs = paired_seqs[:limit]
        validate_paired_fasta_headers(
            headers,
            paired_headers,
            label_a="mpra_fasta",
            label_b="paired_mpra_fasta",
        )
        if paired_seqs and len(paired_seqs[0]) != expected_len:
            raise ValueError(
                f"paired_mpra_fasta has sequence length {len(paired_seqs[0])}, "
                f"expected {expected_len}"
            )

    X = torch.from_numpy(X_np).to(torch.float32)
    log(
        "Loaded MPRA FASTA one-hot tensor in exact file order. "
        f"Tensor shape: {tuple(X.shape)}"
    )
    return X, {
        "input_mode": "mpra_fasta",
        "input_record_count": int(X.shape[0]),
        "mpra_fasta": fasta_path,
        "paired_mpra_fasta": paired_fasta,
    }


def _load_input_tensor(params: dict, log) -> tuple[torch.Tensor, dict]:
    input_mode = params.get("input_mode", "bed")
    if input_mode == "bed":
        return _load_bed_inputs(params, log)
    if input_mode == "mpra_fasta":
        return _load_mpra_fasta_inputs(params, log)
    raise ValueError(
        f"Unsupported input_mode={input_mode!r}; expected 'bed' or 'mpra_fasta'"
    )


def _planned_output_paths(params: dict) -> list[Path]:
    output_path = Path(params["output_fname"])
    planned = []
    model_stems = [Path(model_path).stem for model_path in params["model_fnames"]]

    planned.extend(
        output_path.parent / f"{output_path.stem}__{model_stem}.npz"
        for model_stem in model_stems
    )
    if params.get("save_ensemble_mean_to_output", True):
        planned.append(output_path)
    if len(model_stems) > 1:
        planned.append(output_path.parent / f"{output_path.stem}__ensemble_mean.npz")
    if params.get("save_ohe") is not None:
        planned.append(Path(params["save_ohe"]))

    deduped = []
    seen = set()
    for path in planned:
        key = str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped

def _deep_shap_tangermeme(
    model: torch.nn.Module,
    X: torch.Tensor,
    *,
    batch_size: int,
    n_shuffles: int,
    random_state,
    hypothetical: bool,
    rc_average: bool,
    device: str,
    verbose: bool,
) -> np.ndarray:
    attr_fwd = deep_lift_shap(
        model,
        X,
        hypothetical=hypothetical,
        batch_size=batch_size,
        n_shuffles=n_shuffles,
        random_state=random_state,
        verbose=verbose,
        device=device,
        warning_threshold=0.01,
    )

    if rc_average:
        X_rc = reverse_complement_one_hot(X)
        attr_rc = deep_lift_shap(
            model,
            X_rc,
            hypothetical=hypothetical,
            batch_size=batch_size,
            n_shuffles=n_shuffles,
            random_state=random_state,
            verbose=verbose,
            device=device,
            warning_threshold=0.01,
        )
        attr = 0.5 * (attr_fwd + reverse_complement_attr(attr_rc))
    else:
        attr = attr_fwd

    return attr.detach().cpu().numpy()


def _deep_shap_captum(
    model: torch.nn.Module,
    X: torch.Tensor,
    *,
    n_shuffles: int,
    rc_average: bool,
    device: torch.device,
    random_state,
    verbose: bool,
) -> np.ndarray:
    """Replicate ProCapNet's Captum DeepLiftShap procedure (slow, per-sequence)."""
    try:
        from captum.attr import DeepLiftShap  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "captum is not installed but params.method='captum' was requested. "
            "Install it (e.g. `pip install captum`) or set params.method='tangermeme'."
        ) from e

    # ProCapNet's dinucleotide shuffler used in interpretation scripts.
    try:
        from dinuc_shuffle import dinuc_shuffle  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Could not import ProCapNet dinuc_shuffle. Ensure "
            "`ProCapNet/src/4_interpret_models` is on PYTHONPATH."
        ) from e

    model = model.to(device).eval()
    explainer = DeepLiftShap(model)

    if random_state is not None:
        np.random.seed(int(random_state))

    X = X.to(device)
    attrs = []
    it = range(X.shape[0])
    if verbose:
        try:
            from tqdm import tqdm  # type: ignore

            it = tqdm(it, desc="captum DeepLiftShap", total=X.shape[0])
        except Exception:
            pass

    for i in it:
        seq = X[i : i + 1]
        # ProCapNet's dinuc_shuffle takes a (4, L) tensor.
        # Note: its random_state handling is imperfect upstream; we expose a best-effort
        # determinism by varying the global seed per sequence.
        if random_state is not None:
            np.random.seed(int(random_state) + int(i))
        refs = dinuc_shuffle(seq[0].detach().cpu(), n_shuffles=n_shuffles).float().to(device)

        attr_fwd = explainer.attribute(seq, refs)
        if rc_average:
            seq_rc = reverse_complement_one_hot(seq)
            refs_rc = reverse_complement_one_hot(refs)
            attr_rc = explainer.attribute(seq_rc, refs_rc)
            attr = 0.5 * (attr_fwd + reverse_complement_attr(attr_rc))
        else:
            attr = attr_fwd

        attrs.append(attr.detach().cpu())

    return torch.cat(attrs, dim=0).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--parameters", type=str, required=True)
    args = parser.parse_args()

    def log(msg):
        """Print progress with timestamp and flush so output appears promptly."""
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    log(f"Loading parameters from {args.parameters}")
    # Load parameters 
    with open(args.parameters) as f:
        params = json.load(f)

    # Set defaults if not specified:
    default_params = {
        "input_mode": "bed",
        "in_window": 2114,
        "out_window": 1000,
        "batch_size": 16,
        "n_shuffles": 20,
        "random_state": None,
        "verbose": True,
        "save_ohe": None,
        "chroms": None,
        # Optional cap for quick debugging.
        "n_loci": None,
        "max_jitter": 0,
        "n_cpus": 64,
        "device": "cuda",
        # "tangermeme" matches the current wrapper implementation; "captum" matches
        # ProCapNet's original interpretation scripts in `src/4_interpret_models/`.
        "method": "tangermeme",
        # Whether to average forward and reverse-complement attributions.
        "rc_average": True,
        # tangermeme option: return hypothetical attributions (4-channel per base)
        # rather than "observed" attributions. ProCapNet's original Captum pipeline
        # uses non-hypothetical attribution values and then multiplies by one-hot
        # to get observed-base contribution scores; set this to False to match
        # that intent more closely.
        "hypothetical": False,
        # ProCapNet's historical wrapper detaches softmax probabilities to avoid
        # backpropagating through them. With DeepLIFT/SHAP this can lead to large
        # convergence deltas (completeness violations) because the forward pass
        # still depends on the probabilities while the backward pass cannot.
        # Setting this to False often yields better DeepLIFT/SHAP convergence.
        "detach_softmax_probs": True,
        # Always write `output_fname` even when multiple models are provided.
        # If True, save an aggregated attribution to `output_fname`.
        "save_ensemble_mean_to_output": True,
        # Which subset to aggregate when multiple models are provided.
        # - "teachers_only": drop any model whose stem matches student_model_stem
        # - "all_models": include all successfully loaded models
        "aggregate_mode": "teachers_only",
        "student_model_stem": "student",
        "overwrite": False,
    }

    for k, v in default_params.items():
        if k not in params:
            params[k] = v

    if "output_fname" not in params:
        raise ValueError("output_fname must be specified")
    log(f"Parameters loaded. Output will be saved to {params['output_fname']}")
    planned_outputs = _planned_output_paths(params)
    ensure_writable_paths(
        planned_outputs, overwrite=bool(params.get("overwrite", False))
    )
    log(
        f"Preflighted {len(planned_outputs)} output path(s) with "
        f"overwrite={params.get('overwrite', False)}"
    )

    if torch.cuda.is_available():
        device_str = params.get("device", "cuda")
        torch_device = torch.device(device_str)
        torch.cuda.set_device(torch_device)
        log(f"CUDA available; using device {torch_device}")
    else:
        torch.set_num_threads(max(os.cpu_count(), params["n_cpus"]))
        torch.set_num_interop_threads(max(os.cpu_count(), params["n_cpus"]))
        device_str = "cpu"
        torch_device = torch.device("cpu")
        log("CUDA not available; using CPU")

    X, input_meta = _load_input_tensor(params, log)
    if X.ndim != 3 or X.shape[1] != 4:
        raise ValueError(f"Expected input tensor shape (B, 4, L), found {tuple(X.shape)}")
    if int(X.shape[2]) != int(params["in_window"]):
        raise ValueError(
            f"Input sequence length {X.shape[2]} does not match expected model input "
            f"length {params['in_window']}"
        )
    # tangermeme requires strict one-hot; replace any all-zero positions (unknowns)
    # with an 'A' to keep validation happy.
    unknown_mask = X.sum(dim=1) == 0
    if unknown_mask.any():
        X = X.clone()
        X[:, 0, :][unknown_mask] = 1.0
        log(f"Replaced {unknown_mask.sum().item()} unknown bases with 'A'")

    if params["save_ohe"] is not None:
        np.savez_compressed(params["save_ohe"], X.to(torch.uint8).numpy())
        log(f"Saved one-hot inputs to {params['save_ohe']}")
        written_save_ohe = Path(params["save_ohe"])
    else:
        written_save_ohe = None

    attributions = []
    model_stems = []
    output_path = Path(params["output_fname"])
    per_model_dir = output_path.parent
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameters_path": str(Path(args.parameters).resolve()),
        "loci": params.get("loci"),
        "sequences": params.get("sequences"),
        "in_window": params["in_window"],
        "out_window": params["out_window"],
        "max_jitter": int(params.get("max_jitter", 0) or 0),
        "n_loci": params.get("n_loci"),
        "attribute_type": params["attribute_type"],
        "n_shuffles": params["n_shuffles"],
        "random_state": params["random_state"],
        "rc_average": params["rc_average"],
        "hypothetical": params["hypothetical"],
        "detach_softmax_probs": params.get("detach_softmax_probs", True),
        "method": params.get("method", "tangermeme"),
        "input_mode": params.get("input_mode", "bed"),
        "mpra_fasta": params.get("mpra_fasta"),
        "paired_mpra_fasta": params.get("paired_mpra_fasta"),
        "input_record_count": int(input_meta["input_record_count"]),
        "models": [],
    }
    written_outputs = []
    if written_save_ohe is not None:
        written_outputs.append(written_save_ohe)
    for f in params["model_fnames"]:
        log(f"Starting attribution for model {f}")
        # Load model inside of for loop to prevent VRAM leak
        try:
            model = torch.load(f, weights_only=False, map_location=torch_device)
        except FileNotFoundError:
            log(f"Model file not found; skipping {f}")
            continue
        log("Model loaded into memory")
        model.eval()
        # Make wrappers compatible with bpnetlite's expected (X, X_ctl) signature
        model = SingleInputAdapter(model)

        # Wrap models depending on args.attribute_type
        if params["attribute_type"] == "counts":
            model = ProCapNetCountsWrapper(model)
            log("Wrapped model for counts attribution")
        elif params["attribute_type"] == "profile":
            model = ProCapNetProfileWrapper(
                model, detach_softmax_probs=params.get("detach_softmax_probs", True)
            )
            log("Wrapped model for ProCapNet profile attribution")
        else:
            raise ValueError(
                f"Unknown attribute_type: {params['attribute_type']}."
                "Must be one of ['counts', 'profile']"
            )

        # Calculate attributions
        log(
            f"Running deep_lift_shap with batch_size={params['batch_size']}, "
            f"n_shuffles={params['n_shuffles']}, random_state={params['random_state']}"
        )
        method = params.get("method", "tangermeme")
        if method == "tangermeme":
            attr = _deep_shap_tangermeme(
                model,
                X,
                batch_size=int(params["batch_size"]),
                n_shuffles=int(params["n_shuffles"]),
                random_state=params["random_state"],
                hypothetical=bool(params["hypothetical"]),
                rc_average=bool(params["rc_average"]),
                device=device_str,
                verbose=bool(params["verbose"]),
            )
        elif method == "captum":
            attr = _deep_shap_captum(
                model,
                X,
                n_shuffles=int(params["n_shuffles"]),
                rc_average=bool(params["rc_average"]),
                device=torch_device,
                random_state=params["random_state"],
                verbose=bool(params["verbose"]),
            )
        else:
            raise ValueError("params.method must be one of: 'tangermeme', 'captum'")

        log(f"Attribution complete for model {f}")
        if attr.shape != tuple(X.shape):
            raise ValueError(
                f"Attribution tensor for {f} has shape {attr.shape}, expected {tuple(X.shape)}"
            )
        if (
            params.get("input_mode", "bed") == "mpra_fasta"
            and int(attr.shape[0]) != int(input_meta["input_record_count"])
        ):
            raise ValueError(
                "MPRA FASTA record count does not match attribution entry count: "
                f"{input_meta['input_record_count']} vs {attr.shape[0]}"
            )
        model_stem = Path(f).stem
        per_model_fname = per_model_dir / f"{output_path.stem}__{model_stem}.npz"
        # Save the attribution as `arr_0` for backwards-compatibility with older
        # `np.savez_compressed(path, array)` readers, but include `meta` too.
        np.savez_compressed(
            per_model_fname,
            attr,
            meta=np.array(json.dumps({**meta, "models": [f]}), dtype=np.string_),
        )
        log(f"Saved per-model attributions to {per_model_fname}")
        written_outputs.append(per_model_fname)
        attributions.append(attr)
        meta["models"].append(f)
        model_stems.append(model_stem)

        # clear VRAM
        del model
        torch.cuda.empty_cache()
        log("Released model and cleared CUDA cache")

    # Save
    if len(attributions) == 0:
        raise RuntimeError("No attribution files were generated; all model loads failed.")
    log("Saving aggregated attributions")
    if len(attributions) == 1:
        agg = attributions[0]
        agg_meta = {**meta, "aggregate_mode": "single_model", "aggregate_models": meta["models"]}
    else:
        aggregate_mode = params.get("aggregate_mode", "teachers_only")
        student_stem = params.get("student_model_stem", "student")

        if aggregate_mode == "teachers_only":
            keep = [i for i, stem in enumerate(model_stems) if stem != student_stem]
            if not keep:
                log(
                    "aggregate_mode=teachers_only but no teachers detected; "
                    "falling back to all_models."
                )
                keep = list(range(len(attributions)))
        elif aggregate_mode == "all_models":
            keep = list(range(len(attributions)))
        else:
            raise ValueError(
                f"Unknown aggregate_mode={aggregate_mode!r}; expected "
                "'teachers_only' or 'all_models'."
            )

        agg = np.stack([attributions[i] for i in keep]).mean(axis=0)
        agg_meta = {
            **meta,
            "aggregate_mode": aggregate_mode,
            "aggregate_models": [meta["models"][i] for i in keep],
        }

    if params.get("save_ensemble_mean_to_output", True):
        np.savez_compressed(
            params["output_fname"],
            agg,
            meta=np.array(json.dumps(agg_meta), dtype=np.string_),
        )
        log(f"Saved aggregated attributions to {params['output_fname']}")
        written_outputs.append(Path(params["output_fname"]))
    else:
        log("Skipped writing output_fname (save_ensemble_mean_to_output=false)")

    if len(attributions) > 1:
        ensemble_fname = output_path.parent / f"{output_path.stem}__ensemble_mean.npz"
        np.savez_compressed(
            ensemble_fname,
            agg,
            meta=np.array(json.dumps(agg_meta), dtype=np.string_),
        )
        log(f"Saved ensemble-mean attributions to {ensemble_fname}")
        written_outputs.append(ensemble_fname)
    for written_path in written_outputs:
        if not written_path.exists():
            raise FileNotFoundError(f"Expected output file was not written: {written_path}")
    log(f"Validated {len(written_outputs)} written output file(s)")
    log("Done")

if __name__ == "__main__":
    main()
