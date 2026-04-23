from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
STUDENT_PATH = ROOT / "attributions__student.npz"
EXCLUDE = {
    "attributions__student.npz",
    "attributions__smoke_out.npz",
    "attributions__smoke_out__student.npz",
}


def load_attr(path: Path) -> np.ndarray:
    arr = np.load(path)["arr_0"]
    print(f"loaded {path.name} shape={arr.shape} dtype={arr.dtype}", flush=True)
    return arr


def per_sample_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    a2 = a.reshape(a.shape[0], -1).astype(np.float64, copy=False)
    b2 = b.reshape(b.shape[0], -1).astype(np.float64, copy=False)
    a2 = a2 - a2.mean(axis=1, keepdims=True)
    b2 = b2 - b2.mean(axis=1, keepdims=True)
    num = np.sum(a2 * b2, axis=1)
    denom = np.sqrt(np.sum(a2 * a2, axis=1) * np.sum(b2 * b2, axis=1))
    out = np.full(a2.shape[0], np.nan, dtype=np.float64)
    np.divide(num, denom, out=out, where=denom > 0)
    return out


def main() -> None:
    student = load_attr(STUDENT_PATH)
    other_paths = sorted(
        [path for path in ROOT.glob("attributions__*.npz") if path.name not in EXCLUDE],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    rows = []
    for path in other_paths:
        other = load_attr(path)
        pearson = per_sample_pearson(student, other)
        global_flat = np.corrcoef(
            student.reshape(-1).astype(np.float64),
            other.reshape(-1).astype(np.float64),
        )[0, 1]
        rows.append(
            {
                "student_file": STUDENT_PATH.name,
                "other_file": path.name,
                "n_samples": int(pearson.shape[0]),
                "pearson_mean": float(np.nanmean(pearson)),
                "pearson_median": float(np.nanmedian(pearson)),
                "pearson_std": float(np.nanstd(pearson)),
                "pearson_min": float(np.nanmin(pearson)),
                "pearson_max": float(np.nanmax(pearson)),
                "global_flat_pearson": float(global_flat),
            }
        )
        print(
            f"done {STUDENT_PATH.name} vs {path.name}: "
            f"mean={np.nanmean(pearson):.6f} "
            f"median={np.nanmedian(pearson):.6f} "
            f"global={global_flat:.6f}",
            flush=True,
        )

    out = pd.DataFrame(rows).sort_values("pearson_mean", ascending=False)
    out_path = ROOT / "student_vs_other_attribution_pearson.tsv"
    out.to_csv(out_path, sep="\t", index=False)
    print(out.to_string(index=False), flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
