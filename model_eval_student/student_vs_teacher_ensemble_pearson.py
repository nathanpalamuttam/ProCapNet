from __future__ import annotations

from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
STUDENT_FILE = ROOT / "attributions.npz"
TEACHER_ENSEMBLE_FILE = ROOT / "attributions__ensemble_mean.npz"
OUT_FILE = ROOT / "student_vs_teacher_ensemble_pearson.tsv"
CHUNK_ROWS = 128


def open_npy_from_npz(npz_path: Path):
    zf = zipfile.ZipFile(npz_path)
    names = zf.namelist()
    if names != ["arr_0.npy"]:
        zf.close()
        raise ValueError(f"{npz_path} expected only arr_0.npy, found {names}")
    fh = zf.open("arr_0.npy")
    version = np.lib.format.read_magic(fh)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(fh)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(fh)
    else:
        fh.close()
        zf.close()
        raise ValueError(f"Unsupported npy version {version} in {npz_path}")
    if fortran_order:
        fh.close()
        zf.close()
        raise ValueError(f"Fortran-ordered arrays are not supported: {npz_path}")
    return zf, fh, shape, np.dtype(dtype)


def read_chunk(fh, dtype: np.dtype, shape: tuple[int, ...], rows: int) -> np.ndarray:
    row_size = int(np.prod(shape[1:], dtype=np.int64))
    count = rows * row_size
    needed = count * dtype.itemsize
    raw = fh.read(needed)
    if len(raw) != needed:
        raise EOFError(f"Expected {needed} bytes, got {len(raw)}")
    return np.frombuffer(raw, dtype=dtype).reshape((rows, *shape[1:]))


def per_sample_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = a.reshape(a.shape[0], -1).astype(np.float64, copy=False)
    b2 = b.reshape(b.shape[0], -1).astype(np.float64, copy=False)
    a2 = a2 - a2.mean(axis=1, keepdims=True)
    b2 = b2 - b2.mean(axis=1, keepdims=True)
    num = np.sum(a2 * b2, axis=1)
    denom = np.sqrt(np.sum(a2 * a2, axis=1) * np.sum(b2 * b2, axis=1))
    out = np.full(a2.shape[0], np.nan, dtype=np.float64)
    np.divide(num, denom, out=out, where=denom > 0)
    return out


def global_flat_pearson_from_sums(n: int, sx: float, sy: float, sxx: float, syy: float, sxy: float) -> float:
    num = n * sxy - sx * sy
    denom = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
    if denom <= 0:
        return float("nan")
    return float(num / denom)


def main() -> None:
    start = time.time()
    student_zip, student_fh, student_shape, student_dtype = open_npy_from_npz(STUDENT_FILE)
    teacher_zip, teacher_fh, teacher_shape, teacher_dtype = open_npy_from_npz(TEACHER_ENSEMBLE_FILE)
    try:
        if student_shape != teacher_shape:
            raise ValueError(f"shape mismatch: {student_shape} vs {teacher_shape}")

        n_samples = student_shape[0]
        pearsons = []
        total_count = 0
        sum_x = 0.0
        sum_y = 0.0
        sum_xx = 0.0
        sum_yy = 0.0
        sum_xy = 0.0

        for start_idx in range(0, n_samples, CHUNK_ROWS):
            rows = min(CHUNK_ROWS, n_samples - start_idx)
            student_chunk = read_chunk(student_fh, student_dtype, student_shape, rows)
            teacher_chunk = read_chunk(teacher_fh, teacher_dtype, teacher_shape, rows)

            pearsons.append(per_sample_pearson(student_chunk, teacher_chunk))

            x = student_chunk.reshape(-1).astype(np.float64, copy=False)
            y = teacher_chunk.reshape(-1).astype(np.float64, copy=False)
            total_count += x.size
            sum_x += float(np.sum(x))
            sum_y += float(np.sum(y))
            sum_xx += float(np.sum(x * x))
            sum_yy += float(np.sum(y * y))
            sum_xy += float(np.sum(x * y))

            processed = start_idx + rows
            if processed % 1024 == 0 or processed == n_samples:
                print(f"processed {processed}/{n_samples} samples", flush=True)

        pearson = np.concatenate(pearsons)
        global_flat = global_flat_pearson_from_sums(
            total_count, sum_x, sum_y, sum_xx, sum_yy, sum_xy
        )

        out = pd.DataFrame(
            [
                {
                    "file_a": STUDENT_FILE.name,
                    "file_b": TEACHER_ENSEMBLE_FILE.name,
                    "n_samples": int(pearson.shape[0]),
                    "pearson_mean": float(np.nanmean(pearson)),
                    "pearson_median": float(np.nanmedian(pearson)),
                    "pearson_std": float(np.nanstd(pearson)),
                    "pearson_min": float(np.nanmin(pearson)),
                    "pearson_max": float(np.nanmax(pearson)),
                    "global_flat_pearson": float(global_flat),
                }
            ]
        )
        out.to_csv(OUT_FILE, sep="\t", index=False)

        elapsed = time.time() - start
        print(f"wrote {OUT_FILE} in {elapsed:.1f}s", flush=True)
        print(out.to_string(index=False), flush=True)
    finally:
        student_fh.close()
        student_zip.close()
        teacher_fh.close()
        teacher_zip.close()


if __name__ == "__main__":
    main()
