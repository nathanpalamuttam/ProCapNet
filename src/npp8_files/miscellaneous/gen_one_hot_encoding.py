import os
import sys
import numpy as np

proj_dir = os.path.abspath("ProCapNet")

# Prefer local npp8_files data_loader instead of 2_train_models/data_loading
_npp8_dir = os.path.join(proj_dir, "src", "npp8_files")
if _npp8_dir not in sys.path:
    sys.path.append(_npp8_dir)

import data_loader as _dl  # ProCapNet/src/npp8_files/data_loader.py



def extract_sequences(genome_path, peak_path, negatives_path):
    """Extract peak one-hot sequences using DistillerPeakGenerator.

    Uses the project’s DistillerPeakGenerator (with jitter/RC augmentation) and keeps
    only examples labeled as peaks (label==1). Returns an array shaped
    (N_peaks, 4, in_window).
    """
    if _dl is None or not hasattr(_dl, "DistillerPeakGenerator"):
        raise ImportError("npp8_files/data_loader.DistillerPeakGenerator not available")

    loader = _dl.DistillerPeakGenerator(
        peaks=peak_path,
        negatives=negatives_path,
        sequences=genome_path,
    )

    seqs = []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            X, y = batch
        elif isinstance(batch, (list, tuple)) and len(batch) == 3:
            X, _Xctl, y = batch
        else:
            # Unexpected format
            raise RuntimeError("Unexpected batch format from DistillerPeakGenerator")

        # keep only peaks (label==1)
        if hasattr(y, 'numpy'):
            y_np = y.numpy()
        else:
            y_np = np.asarray(y)
        mask = y_np == 1
        if np.any(mask):
            X_sel = X[mask]
            seqs.append(X_sel.numpy() if hasattr(X_sel, 'numpy') else np.asarray(X_sel))

    if not seqs:
        return np.zeros((0, 4, in_window), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(seqs, axis=0))


def main(cell_type="K562", data_type="procap", in_window=2114, verbose=True):
    global proj_dir
    genome_path = os.path.join(proj_dir, "genomes", "hg38.withrDNA.fasta")
    peak_path = os.path.join(proj_dir, "data", data_type, "processed", cell_type, "peaks.bed.gz")
    negatives_path = os.path.join(
        proj_dir, "data", data_type, "processed", cell_type,
        f"dnase_peaks_no_{data_type}_overlap.bed.gz",
    )
    if not os.path.exists(negatives_path):
        # Fallback to a common fold filename if the unsplit file is not present
        negatives_path = os.path.join(
            proj_dir, "data", data_type, "processed", cell_type,
            f"dnase_peaks_no_{data_type}_overlap_fold1_train.bed.gz",
        )

    if not os.path.exists(peak_path):
        raise FileNotFoundError(f"Peak file not found: {peak_path}")
    if not os.path.exists(negatives_path):
        raise FileNotFoundError(f"Negatives (DNase-no-overlap) not found: {negatives_path}")

    # Build one-hot sequences using DistillerPeakGenerator (peaks only)
    seqs = extract_sequences(genome_path, peak_path, negatives_path)
    print(f"Loaded one-hot sequences with shape {seqs.shape}")

    save_dir = os.path.join(proj_dir, "data", data_type, "processed", cell_type)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"onehot_seqs_{in_window}.npy")

    np.save(save_path, seqs)
    print(f"Saved one-hot sequences to {save_path}")


if __name__ == "__main__":
    main()




# _REVCOMP_IDX = np.array([3, 2, 1, 0], dtype=np.int64)


# def _shift_array(arr, shift, axis=-1):
#     """Shift along ``axis`` with zero padding instead of wraparound."""

#     if shift == 0:
#         return arr

#     result = np.zeros_like(arr)
#     if shift > 0:
#         slicer_src = [slice(None)] * arr.ndim
#         slicer_dst = [slice(None)] * arr.ndim
#         slicer_src[axis] = slice(None, -shift)
#         slicer_dst[axis] = slice(shift, None)
#         result[tuple(slicer_dst)] = arr[tuple(slicer_src)]
#     else:
#         slicer_src = [slice(None)] * arr.ndim
#         slicer_dst = [slice(None)] * arr.ndim
#         slicer_src[axis] = slice(-shift, None)
#         slicer_dst[axis] = slice(None, shift)
#         result[tuple(slicer_dst)] = arr[tuple(slicer_src)]
#     return result


# def augment_onehot_sequences(seqs, outputs=None, shift_range=(-1024, 1024), rc_prob=0.5,
#                              seed=None, save_path=None, output_save_path=None):
#     """Apply shift jitter and reverse-complement augmentation.

#     Parameters
#     ----------
#     seqs : numpy.ndarray
#         Array of shape ``(N, 4, L)`` containing one-hot encoded sequences.

#     outputs : numpy.ndarray or None, optional
#         Optional array of corresponding targets to transform alongside ``seqs``.
#         Expected shape ``(N, ..., L)`` where the last dimension matches ``L``.
#         For two-strand outputs, the first axis should index strands to allow
#         strand swapping on reverse complement.

#     shift_range : tuple or int, optional
#         Inclusive jitter range. If an int ``k`` is supplied, shifts are drawn
#         uniformly from ``[-k, k]``. Default draws from [-1024, 1024].

#     rc_prob : float, optional
#         Probability of applying reverse complementation. Default is 0.5.

#     seed : int or numpy.random.Generator, optional
#         Seed or generator for reproducibility.

#     Returns
#     -------
#     tuple
#         ``(aug_seqs, aug_outputs)`` where ``aug_outputs`` is ``None`` when
#         no outputs were provided.
#     """

#     if isinstance(shift_range, int):
#         shift_low, shift_high = -shift_range, shift_range
#     else:
#         shift_low, shift_high = shift_range

#     rng = np.random.default_rng(seed)

#     aug_seqs = np.empty_like(seqs)
#     aug_outputs = None if outputs is None else np.empty_like(outputs)

#     for idx, seq in enumerate(seqs):
#         shift = rng.integers(shift_low, shift_high + 1)
#         shifted_seq = _shift_array(seq, shift)

#         out_shifted = None
#         if outputs is not None:
#             out_shifted = _shift_array(outputs[idx], shift)

#         if rng.random() < rc_prob:
#             shifted_seq = shifted_seq[_REVCOMP_IDX][:, ::-1]
#             if out_shifted is not None:
#                 out_shifted = np.flip(out_shifted, axis=-1)
#                 if out_shifted.ndim >= 2 and out_shifted.shape[0] == 2:
#                     out_shifted = out_shifted[::-1]

#         aug_seqs[idx] = shifted_seq
#         if aug_outputs is not None:
#             aug_outputs[idx] = out_shifted

#     if save_path is not None:
#         np.save(save_path, aug_seqs)
#     if aug_outputs is not None and output_save_path is not None:
#         np.save(output_save_path, aug_outputs)

#     return aug_seqs, aug_outputs

