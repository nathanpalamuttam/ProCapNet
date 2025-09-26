import os
import sys
import numpy as np

proj_dir = os.path.abspath("ProCapNet")
train_src = os.path.join(proj_dir, "src", "2_train_models")
utils_src = os.path.join(proj_dir, "src", "utils")

for path in (train_src, utils_src):
    if path not in sys.path:
        sys.path.append(path)

from data_loading import extract_sequences


_REVCOMP_IDX = np.array([3, 2, 1, 0], dtype=np.int64)


def _shift_array(arr, shift, axis=-1):
    """Shift along ``axis`` with zero padding instead of wraparound."""

    if shift == 0:
        return arr

    result = np.zeros_like(arr)
    if shift > 0:
        slicer_src = [slice(None)] * arr.ndim
        slicer_dst = [slice(None)] * arr.ndim
        slicer_src[axis] = slice(None, -shift)
        slicer_dst[axis] = slice(shift, None)
        result[tuple(slicer_dst)] = arr[tuple(slicer_src)]
    else:
        slicer_src = [slice(None)] * arr.ndim
        slicer_dst = [slice(None)] * arr.ndim
        slicer_src[axis] = slice(-shift, None)
        slicer_dst[axis] = slice(None, shift)
        result[tuple(slicer_dst)] = arr[tuple(slicer_src)]
    return result


def augment_onehot_sequences(seqs, outputs=None, shift_range=(-1024, 1024), rc_prob=0.5,
                             seed=None, save_path=None, output_save_path=None):
    """Apply shift jitter and reverse-complement augmentation.

    Parameters
    ----------
    seqs : numpy.ndarray
        Array of shape ``(N, 4, L)`` containing one-hot encoded sequences.

    outputs : numpy.ndarray or None, optional
        Optional array of corresponding targets to transform alongside ``seqs``.
        Expected shape ``(N, ..., L)`` where the last dimension matches ``L``.
        For two-strand outputs, the first axis should index strands to allow
        strand swapping on reverse complement.

    shift_range : tuple or int, optional
        Inclusive jitter range. If an int ``k`` is supplied, shifts are drawn
        uniformly from ``[-k, k]``. Default draws from [-1024, 1024].

    rc_prob : float, optional
        Probability of applying reverse complementation. Default is 0.5.

    seed : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    tuple
        ``(aug_seqs, aug_outputs)`` where ``aug_outputs`` is ``None`` when
        no outputs were provided.
    """

    if isinstance(shift_range, int):
        shift_low, shift_high = -shift_range, shift_range
    else:
        shift_low, shift_high = shift_range

    rng = np.random.default_rng(seed)

    aug_seqs = np.empty_like(seqs)
    aug_outputs = None if outputs is None else np.empty_like(outputs)

    for idx, seq in enumerate(seqs):
        shift = rng.integers(shift_low, shift_high + 1)
        shifted_seq = _shift_array(seq, shift)

        out_shifted = None
        if outputs is not None:
            out_shifted = _shift_array(outputs[idx], shift)

        if rng.random() < rc_prob:
            shifted_seq = shifted_seq[_REVCOMP_IDX][:, ::-1]
            if out_shifted is not None:
                out_shifted = np.flip(out_shifted, axis=-1)
                if out_shifted.ndim >= 2 and out_shifted.shape[0] == 2:
                    out_shifted = out_shifted[::-1]

        aug_seqs[idx] = shifted_seq
        if aug_outputs is not None:
            aug_outputs[idx] = out_shifted

    if save_path is not None:
        np.save(save_path, aug_seqs)
    if aug_outputs is not None and output_save_path is not None:
        np.save(output_save_path, aug_outputs)

    return aug_seqs, aug_outputs


def main(cell_type="K562", data_type="procap", in_window=2114, verbose=True):
    global proj_dir
    # genome_path = os.path.join(proj_dir, "genomes", "hg38.withrDNA.fasta")
    # chrom_sizes = os.path.join(proj_dir, "genomes", "hg38.withrDNA.chrom.sizes")
    # peak_path = os.path.join(proj_dir, "data", data_type, "processed", cell_type, "peaks.bed.gz")

    # if not os.path.exists(peak_path):
    #     raise FileNotFoundError(f"Peak file not found: {peak_path}")

    # seqs = extract_sequences(genome_path, chrom_sizes, peak_path,
    #                          in_window=in_window, verbose=verbose)
    # print(f"Loaded one-hot sequences with shape {seqs.shape}")

    save_dir = os.path.join(proj_dir, "data", data_type, "processed", cell_type)
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, f"onehot_seqs_{in_window}.npy")

    # np.save(save_path, seqs)
    # print(f"Saved one-hot sequences to {save_path}")
    seqs = np.load("ProCapNet/data/procap/processed/K562/onehot_seqs_2114.npy")
    aug_seqs, _ = augment_onehot_sequences(seqs, shift_range=1024, rc_prob=0.5, seed=42)
    aug_save_path = os.path.join(save_dir, f"onehot_seqs_{in_window}_augmented.npy")
    augment_onehot_sequences(seqs, shift_range=1024, rc_prob=0.5,
                             seed=0, save_path=aug_save_path)
    print(f"Saved augmented one-hot sequences to {aug_save_path}")


if __name__ == "__main__":
    main()
