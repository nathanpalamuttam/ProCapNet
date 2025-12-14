# dataloader.py
# Author: adamyhe@gmail.com
# Code adapted from bpnet-lite by Jacob Schreiber
# https://github.com/jmschrei/bpnet-lite/blob/master/bpnetlite/io.py

import numpy
import torch
from tangermeme.io import extract_loci

# from bpnetlite.io import PeakNegativeSampler


class DistillerPeakNegativeSampler(torch.utils.data.Dataset):
    """A data generator mimicking the BPNet data loading procedure.

    Here, a set of peaks and negatives are separately loaded. These sets can be
    any size. From these sets, batches of given size are sampled that are a
    mixture of peaks and negatives. The peaks are sampled by randomly iterating
    over the entire set such that one epoch means one pass over the entire data
    set. The negatives are sampled by randomly choosing regions from the
    negatives at the given ratio to peaks, without considering whether they have
    been selected before.

    Because peaks and negatives are provided as separate tensors, different
    jittering can be used on them. This means that, for instance, jittering can
    be used on peaks but not used on negatives.

    In the documentation below, `mj` = max_jitter.

    Note that, although the data is passed in as PyTorch tensors, they are saved
    as numpy arrays for faster slicing during training.


    Parameters
    ----------
    peak_sequences: torch.tensor, shape=(n_peaks, 4, in_window+2*mj)
            A tensor of peak sequences that are one-hot encoded. See above for the
            connection between the length here and jitter.

    peak_controls: torch.tensor, shape=(n, t, out_window+2*mj) or None, optional
            A tenso of the control signal to take as input, usually base-pair counts, for `n`
            examples with `t` strands and output length `out_window`. If
            None, does not return controls.

    negative_sequences: torch.tensor, shape=(n, 4, in_window+2*max_jitter)
            A one-hot encoded tensor of `n` example sequences, each of input
            length `in_window`. See description above for connection with jitter.

    negative_controls: torch.tensor, shape=(n, t, out_window+2*max_jitter) or None, optional
            The control signal to take as input, usually counts, for `n`
            examples with `t` strands and output length `out_window`. If
            None, does not return controls.

    p: torch.tensor or None, shape=(n,)
            A vector of probabilities that sum to 1 containing the sampling probability
            of each sequence. If None, use a uniform distribution.

    in_window: int, optional
            The input window size. Default is 2114.

    out_window: int, optional
            The output window size. Default is 1000.

    max_jitter: int, optional
            The maximum amount of jitter to add, in either direction, to the
            midpoints that are passed in. Default is 0.

    reverse_complement: bool, optional
            Whether to reverse complement-augment half of the data. Default is False.

    mutation_rate: float, optional
            Probability of mutating each nucleotide. Default is 0.04 (4%).

    sv_rate: float, optional
            Poisson lambda parameter for number of structural variations per sequence.
            Default is 1.0.

    sv_min_length: int, optional
            Minimum length of structural variations. Default is 1.

    sv_max_length: int, optional
            Maximum length of structural variations. Default is 20.

    random_state: int or None, optional
            Whether to use a deterministic seed or not.
    """

    def __init__(
        self,
        peak_sequences,
        negative_sequences,
        peak_controls=None,
        negative_controls=None,
        negative_ratio=0.1,
        in_window=2114,
        out_window=1000,
        max_jitter=0,
        reverse_complement=False,
        mutation_rate=0.04,
        sv_rate=1.0,
        sv_min_length=1,
        sv_max_length=20,
        shuffle=True,
        random_state=None,
    ):
        self.peak_sequences = peak_sequences.numpy(force=True)
        self.peak_controls = (
            peak_controls.numpy(force=True) if peak_controls is not None else None
        )
        self.n_peaks = len(self.peak_sequences)

        self.negative_sequences = negative_sequences.numpy(force=True)
        self.negative_controls = (
            negative_controls.numpy(force=True) if negative_controls is not None else None
        )
        self.n_negatives = len(self.negative_sequences)

        self.negative_ratio = negative_ratio
        self.negative_likelihood = 1 / (1 + 1 / negative_ratio)

        self.in_window = in_window
        self.out_window = out_window
        self.max_jitter = max_jitter
        self.reverse_complement = reverse_complement
        self.mutation_rate = mutation_rate
        self.sv_rate = sv_rate
        self.sv_min_length = sv_min_length
        self.sv_max_length = sv_max_length
        self.shuffle = shuffle

        self.random_state = numpy.random.RandomState(random_state)
        self.n_peaks_seen = 0
        self.peak_ordering = None

    def __len__(self):
        return self.n_peaks + int(self.n_peaks * self.negative_ratio)

    def _apply_point_mutations(self, Xi):
        """Apply random point mutations to the one-hot encoded sequence.
        
        Parameters
        ----------
        Xi: torch.Tensor, shape=(4, length)
            One-hot encoded sequence.
            
        Returns
        -------
        Xi: torch.Tensor
            Mutated sequence.
        """
        if self.mutation_rate <= 0:
            return Xi
            
        # Generate mutation mask
        mutation_mask = self.random_state.uniform(size=Xi.shape[1]) < self.mutation_rate
        n_mutations = mutation_mask.sum()
        
        if n_mutations > 0:
            # Convert to numpy for easier manipulation
            Xi_np = Xi.numpy()
            
            # For each position to mutate, choose a random nucleotide
            for pos in numpy.where(mutation_mask)[0]:
                # Zero out current nucleotide
                Xi_np[:, pos] = 0
                # Set random nucleotide to 1
                new_nuc = self.random_state.randint(0, 4)
                Xi_np[new_nuc, pos] = 1
            
            Xi = torch.from_numpy(Xi_np)
        
        return Xi

    def _apply_structural_variations(self, Xi):
        """Apply random structural variations (insertions, deletions, inversions).
        
        Parameters
        ----------
        Xi: torch.Tensor, shape=(4, length)
            One-hot encoded sequence.
            
        Returns
        -------
        Xi: torch.Tensor
            Sequence with structural variations applied.
        """
        if self.sv_rate <= 0:
            return Xi
            
        # Sample number of SVs from Poisson distribution
        n_svs = self.random_state.poisson(self.sv_rate)
        
        if n_svs == 0:
            return Xi
        
        Xi_np = Xi.numpy()
        seq_length = Xi_np.shape[1]
        
        for _ in range(n_svs):
            # Choose SV type
            sv_type = self.random_state.choice(['insertion', 'deletion', 'inversion'])
            
            # Choose SV length
            sv_length = self.random_state.randint(self.sv_min_length, self.sv_max_length + 1)
            
            # Choose random position (ensure we don't go out of bounds)
            if sv_type == 'insertion':
                pos = self.random_state.randint(0, seq_length)
                # Generate random sequence for insertion
                random_seq = numpy.zeros((4, sv_length), dtype=Xi_np.dtype)
                random_nucs = self.random_state.randint(0, 4, size=sv_length)
                random_seq[random_nucs, numpy.arange(sv_length)] = 1
                
                # Insert the random sequence
                Xi_np = numpy.concatenate([Xi_np[:, :pos], random_seq, Xi_np[:, pos:]], axis=1)
                seq_length += sv_length
                
            elif sv_type == 'deletion':
                if seq_length <= sv_length:
                    continue  # Skip if deletion would remove entire sequence
                pos = self.random_state.randint(0, seq_length - sv_length + 1)
                # Delete the region
                Xi_np = numpy.concatenate([Xi_np[:, :pos], Xi_np[:, pos + sv_length:]], axis=1)
                seq_length -= sv_length
                
            elif sv_type == 'inversion':
                if seq_length <= sv_length:
                    continue  # Skip if inversion region is larger than sequence
                pos = self.random_state.randint(0, seq_length - sv_length + 1)
                # Invert the region (flip along position axis and complement)
                region = Xi_np[:, pos:pos + sv_length]
                # Flip positions
                region = numpy.flip(region, axis=1).copy()
                # Complement: A<->T (indices 0<->3), C<->G (indices 1<->2)
                region = region[[3, 2, 1, 0], :]
                Xi_np[:, pos:pos + sv_length] = region
        
        # Ensure we maintain the correct window size by cropping or padding
        if Xi_np.shape[1] > self.in_window:
            # Crop from center
            excess = Xi_np.shape[1] - self.in_window
            start = excess // 2
            Xi_np = Xi_np[:, start:start + self.in_window]
        elif Xi_np.shape[1] < self.in_window:
            # Pad with zeros (representing ambiguous nucleotides)
            deficit = self.in_window - Xi_np.shape[1]
            pad_left = deficit // 2
            pad_right = deficit - pad_left
            Xi_np = numpy.pad(Xi_np, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
        
        return torch.from_numpy(Xi_np)

    def __getitem__(self, idx):
        if idx == 0:
            self.peak_ordering = numpy.arange(self.n_peaks)
            if self.shuffle:
                self.random_state.shuffle(self.peak_ordering)

        if self.random_state.uniform() >= self.negative_likelihood:
            idx = self.peak_ordering[self.n_peaks_seen % self.n_peaks]
            jitter = 0 if self.max_jitter <= 0 else self.random_state.randint(self.max_jitter * 2)
            label = 1

            X, X_ctl = self.peak_sequences, self.peak_controls
            self.n_peaks_seen += 1

        else:
            idx = self.random_state.randint(self.n_negatives)
            jitter = 0
            label = 0

            X, X_ctl = self.negative_sequences, self.negative_controls

        Xi = torch.from_numpy(X[idx][:, jitter : jitter + self.in_window])
        if self.peak_controls is not None:
            Xi_ctl = torch.from_numpy(X_ctl[idx][:, jitter : jitter + self.in_window])

        # Apply point mutations
        Xi = self._apply_point_mutations(Xi)
        
        # Apply structural variations
        Xi = self._apply_structural_variations(Xi)

        if self.reverse_complement and self.random_state.randint(2) == 1:
            Xi = torch.flip(Xi, [0, 1])

            if self.peak_controls is not None:
                Xi_ctl = torch.flip(Xi_ctl, [0, 1])

        if self.peak_controls is not None:
            return Xi, Xi_ctl, label

        return Xi, label


class DistillerDataGenerator(torch.utils.data.Dataset):
    """A data generator for BPNet inputs.

    This generator takes in an extracted set of sequences,
    and control signals, and will return a single element with random
    jitter and reverse-complement augmentation applied. Jitter is implemented
    efficiently by taking in data that is wider than the in/out windows by
    two times the maximum jitter and windows are extracted from that.
    Essentially, if an input window is 1000 and the maximum jitter is 128, one
    would pass in data with a length of 1256 and a length 1000 window would be
    extracted starting between position 0 and 256. This  generator must be
    wrapped by a PyTorch generator object.

    Parameters
    ----------
    sequences: torch.tensor, shape=(n, 4, in_window+2*max_jitter)
            A one-hot encoded tensor of `n` example sequences, each of input
            length `in_window`. See description above for connection with jitter.

    signals: torch.tensor, shape=(n, t, out_window+2*max_jitter)
            The signals to predict, usually counts, for `n` examples with
            `t` output tasks (usually 2 if stranded, 1 otherwise), each of
            output length `out_window`. See description above for connection
            with jitter.

    controls: torch.tensor, shape=(n, t, out_window+2*max_jitter) or None, optional
            The control signal to take as input, usually counts, for `n`
            examples with `t` strands and output length `out_window`. If
            None, does not return controls.

    p: torch.tensor or None, shape=(n,)
            A vector of probabilities that sum to 1 containing the sampling probability
            of each sequence. If None, use a uniform distribution.

    in_window: int, optional
            The input window size. Default is 2114.

    out_window: int, optional
            The output window size. Default is 1000.

    max_jitter: int, optional
            The maximum amount of jitter to add, in either direction, to the
            midpoints that are passed in. Default is 0.

    reverse_complement: bool, optional
            Whether to reverse complement-augment half of the data. Default is False.

    mutation_rate: float, optional
            Probability of mutating each nucleotide. Default is 0.04 (4%).

    sv_rate: float, optional
            Poisson lambda parameter for number of structural variations per sequence.
            Default is 1.0.

    sv_min_length: int, optional
            Minimum length of structural variations. Default is 1.

    sv_max_length: int, optional
            Maximum length of structural variations. Default is 20.

    random_state: int or None, optional
            Whether to use a deterministic seed or not.
    """

    def __init__(
        self,
        sequences,
        controls=None,
        p=None,
        in_window=2114,
        out_window=1000,
        max_jitter=0,
        reverse_complement=False,
        mutation_rate=0.04,
        sv_rate=1.0,
        sv_min_length=1,
        sv_max_length=20,
        random_state=None,
    ):
        self.p = p
        self.in_window = in_window
        self.out_window = out_window
        self.max_jitter = max_jitter

        self.reverse_complement = reverse_complement
        self.mutation_rate = mutation_rate
        self.sv_rate = sv_rate
        self.sv_min_length = sv_min_length
        self.sv_max_length = sv_max_length
        self.random_state = numpy.random.RandomState(random_state)

        self.controls = controls
        self.sequences = sequences

        self.random_idxs = None
        self.n_random = 1000000

    def __len__(self):
        return len(self.sequences)

    def _apply_point_mutations(self, X):
        """Apply random point mutations to the one-hot encoded sequence."""
        if self.mutation_rate <= 0:
            return X
            
        mutation_mask = self.random_state.uniform(size=X.shape[1]) < self.mutation_rate
        n_mutations = mutation_mask.sum()
        
        if n_mutations > 0:
            X_np = X.numpy()
            for pos in numpy.where(mutation_mask)[0]:
                X_np[:, pos] = 0
                new_nuc = self.random_state.randint(0, 4)
                X_np[new_nuc, pos] = 1
            X = torch.from_numpy(X_np)
        
        return X

    def _apply_structural_variations(self, X):
        """Apply random structural variations (insertions, deletions, inversions)."""
        if self.sv_rate <= 0:
            return X
            
        n_svs = self.random_state.poisson(self.sv_rate)
        
        if n_svs == 0:
            return X
        
        X_np = X.numpy()
        seq_length = X_np.shape[1]
        
        for _ in range(n_svs):
            sv_type = self.random_state.choice(['insertion', 'deletion', 'inversion'])
            sv_length = self.random_state.randint(self.sv_min_length, self.sv_max_length + 1)
            
            if sv_type == 'insertion':
                pos = self.random_state.randint(0, seq_length)
                random_seq = numpy.zeros((4, sv_length), dtype=X_np.dtype)
                random_nucs = self.random_state.randint(0, 4, size=sv_length)
                random_seq[random_nucs, numpy.arange(sv_length)] = 1
                X_np = numpy.concatenate([X_np[:, :pos], random_seq, X_np[:, pos:]], axis=1)
                seq_length += sv_length
                
            elif sv_type == 'deletion':
                if seq_length <= sv_length:
                    continue
                pos = self.random_state.randint(0, seq_length - sv_length + 1)
                X_np = numpy.concatenate([X_np[:, :pos], X_np[:, pos + sv_length:]], axis=1)
                seq_length -= sv_length
                
            elif sv_type == 'inversion':
                if seq_length <= sv_length:
                    continue
                pos = self.random_state.randint(0, seq_length - sv_length + 1)
                region = X_np[:, pos:pos + sv_length]
                region = numpy.flip(region, axis=1).copy()
                region = region[[3, 2, 1, 0], :]
                X_np[:, pos:pos + sv_length] = region
        
        # Maintain correct window size
        if X_np.shape[1] > self.in_window:
            excess = X_np.shape[1] - self.in_window
            start = excess // 2
            X_np = X_np[:, start:start + self.in_window]
        elif X_np.shape[1] < self.in_window:
            deficit = self.in_window - X_np.shape[1]
            pad_left = deficit // 2
            pad_right = deficit - pad_left
            X_np = numpy.pad(X_np, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
        
        return torch.from_numpy(X_np)

    def __getitem__(self, idx):
        if idx % self.n_random == 0:
            self.random_idxs = self.random_state.choice(
                len(self), p=self.p, size=self.n_random
            )

        i = self.random_idxs[idx % self.n_random]
        j = (
            0
            if self.max_jitter == 0
            else self.random_state.randint(self.max_jitter * 2)
        )

        X = self.sequences[i][:, j : j + self.in_window]

        if self.controls is not None:
            X_ctl = self.controls[i][:, j : j + self.in_window]

        # Apply point mutations
        X = self._apply_point_mutations(X)
        
        # Apply structural variations
        X = self._apply_structural_variations(X)

        if self.reverse_complement and self.random_state.choice(2) == 1:
            X = torch.flip(X, [0, 1])

            if self.controls is not None:
                X_ctl = torch.flip(X_ctl, [0, 1])

        if self.controls is not None:
            return X, X_ctl

        return X


def DistillerPeakGenerator(
    peaks,
    negatives,
    sequences,
    controls=None,
    chroms=None,
    in_window=2114,
    out_window=1000,
    max_jitter=128,
    negative_ratio=0.1,
    reverse_complement=True,
    mutation_rate=0.04,
    sv_rate=1.0,
    sv_min_length=1,
    sv_max_length=20,
    shuffle=True,
    min_counts=None,
    max_counts=None,
    summits=False,
    exclusion_lists=None,
    random_state=None,
    pin_memory=True,
    num_workers=0,
    batch_size=32,
    verbose=False,
):
    """This is a constructor function that handles all IO.

    This function will extract signal from all signal and control files,
    pass that into a DataGenerator, and wrap that using a PyTorch data
    loader. This is the only function that needs to be used.


    Parameters
    ----------
    peaks: str or pandas.DataFrame or list/tuple of such
            A BED-formatted file containing peak coordinates. This can be either
            the string path to the BED file or a pandas DataFrame object containing
            three columns: chrom, start, and end. Alternatively, this can be a list
            of such objects whose coordinates will be interleaved.

    negatives: str or pandas.DataFrame or list/tuple of such
            A BED-formatted file containing negative coordinates. This can be either
            the string path to the BED file or a pandas DataFrame object containing
            three columns: chrom, start, and end. Alternatively, this can be a list
            of such objects whose coordinates will be interleaved.

    sequences: str or dictionary
            Either the path to a fasta file to read from or a dictionary where the
            keys are the unique set of chromosoms and the values are one-hot
            encoded sequences as numpy arrays or memory maps.

    signals: list of strs or list of dictionaries
            A list of filepaths to bigwig files, where each filepath will be read
            using pyBigWig, or a list of dictionaries where the keys are the same
            set of unique chromosomes and the values are numpy arrays or memory
            maps.

    controls: list of strs or list of dictionaries or None, optional
            A list of filepaths to bigwig files, where each filepath will be read
            using pyBigWig, or a list of dictionaries where the keys are the same
            set of unique chromosomes and the values are numpy arrays or memory
            maps. If None, no control tensor is returned. Default is None.

    chroms: list or None, optional
            A set of chromosomes to extact loci from. Loci in other chromosomes
            in the locus file are ignored. If None, all loci are used. Default is
            None.

    in_window: int, optional
            The input window size. Default is 2114.

    out_window: int, optional
            The output window size. Default is 1000.

    max_jitter: int, optional
            The maximum amount of jitter to add, in either direction, to the
            midpoints that are passed in. Default is 128.

    negative_ratio: float, optional
            The ratio of negatives compared to peaks in each batch. A value of 1 means
            that each batch is balanced, and a value of 10 means that there would be 10
            negatives for each positive. Note that this is independent of the number of
            peaks and negatives provided. Even if the `peaks` input has 10x the number
            of coordinates as the `negatives` one, if the ratio is 1 each batch during
            training will be balanced (on average).

    reverse_complement: bool, optional
            Whether to reverse complement-augment half of the data. Default is True.

    mutation_rate: float, optional
            Probability of mutating each nucleotide. Default is 0.04 (4%).

    sv_rate: float, optional
            Poisson lambda parameter for number of structural variations per sequence.
            Default is 1.0.

    sv_min_length: int, optional
            Minimum length of structural variations. Default is 1.

    sv_max_length: int, optional
            Maximum length of structural variations. Default is 20.

    shuffle: bool, optional
            Whether to randomly sample peaks, if True, or to proceed sequentially
            through them, if False. Negatives are always randomly sampled. Default
            is True.

    min_counts: float or None, optional
            The minimum number of counts, summed across the length of each example
            and across all tasks, needed to be kept. If None, no minimum. Default
            is None.

    max_counts: float or None, optional
            The maximum number of counts, summed across the length of each example
            and across all tasks, needed to be kept. If None, no maximum. Default
            is None.
    exclusion_lists: list or None, optional
            A list of strings of filenames to BED-formatted files containing exclusion
            lists, i.e., regions where overlapping loci should be filtered out. If None,
            no filtering is performed based on exclusion zones. Default is None.

    random_state: int or None, optional
            Whether to use a deterministic seed or not.

    pin_memory: bool, optional
            Whether to pin page memory to make data loading onto a GPU easier.
            Default is True.

    num_workers: int, optional
            The number of processes fetching data at a time to feed into a model.
            If 0, data is fetched from the main process. Default is 0.

    batch_size: int, optional
            The number of data elements per batch. Default is 32.

    verbose: bool, optional
            Whether to display a progress bar while loading. Default is False.

    Returns
    -------
    X: torch.utils.data.DataLoader
            A PyTorch DataLoader wrapped DataGenerator object.
    """

    # Wrapper to tolerate tangermeme.extract_loci signature differences
    def _safe_extract_loci(loci, max_j, ret_mask=True):
        try:
            # Call with a conservative argument set; leave out optional kwargs
            # (exclusion_lists, min/max_counts, return_mask) that older
            # extract_loci implementations may not support. The ret_mask arg is
            # accepted by the wrapper for API parity but is intentionally unused.
            return extract_loci(
                loci=loci,
                sequences=sequences,
                in_signals=controls,
                chroms=chroms,
                in_window=in_window,
                out_window=out_window,
                max_jitter=max_j,
                min_counts=min_counts,
                max_counts=max_counts,
                summits=False,
                exclusion_lists=exclusion_lists,
                ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
                return_mask=ret_mask,
                verbose=verbose,
            )
        except TypeError:
            # Fallback: minimal arg set
            return extract_loci(
                loci=loci,
                sequences=sequences,
                in_signals=controls,
                chroms=chroms,
                in_window=in_window,
                out_window=out_window,
                max_jitter=max_j,
                verbose=verbose,
            )

    X_peaks = _safe_extract_loci(peaks, max_j=max_jitter, ret_mask=True)
    X_bg = _safe_extract_loci(negatives, max_j=0, ret_mask=True)

    # Determine outliers only if a signal tensor with >=3 dims is present
    try:
        signals_peaks = X_peaks[1]
        if hasattr(signals_peaks, "dim") and signals_peaks.dim() >= 3:
            loci_counts = signals_peaks.sum(dim=(1, 2))
            outlier_threshold = torch.quantile(loci_counts, 0.99) * 1.2
            outlier_idxs = loci_counts > outlier_threshold
        else:
            raise AttributeError
    except Exception:
        outlier_idxs = torch.zeros(X_peaks[0].shape[0], dtype=torch.bool)

    if verbose:
        # Masks may be missing depending on extract_loci; guard accordingly
        try:
            n_filtered_peaks = len(X_peaks[-1]) - X_peaks[-1].sum() + outlier_idxs.sum()
        except Exception:
            n_filtered_peaks = int(outlier_idxs.sum().item()) if hasattr(outlier_idxs, 'sum') else 0
        try:
            n_filtered_negatives = len(X_bg[-1]) - X_bg[-1].sum()
        except Exception:
            n_filtered_negatives = 0

        print("\nFiltered Peaks: {}".format(n_filtered_peaks))
        print("Filtered Negatives: {}".format(n_filtered_negatives))

    ###

    X_gen = DistillerPeakNegativeSampler(
        peak_sequences=X_peaks[0][~outlier_idxs],
        peak_controls=None if (not isinstance(X_peaks, (list, tuple)) or controls is None or len(X_peaks) < 3) else X_peaks[2][~outlier_idxs],
        negative_sequences=X_bg[0],
        negative_controls=None if (not isinstance(X_bg, (list, tuple)) or controls is None or len(X_bg) < 3) else X_bg[2],
        negative_ratio=negative_ratio,
        in_window=in_window,
        out_window=out_window,
        max_jitter=max_jitter,
        reverse_complement=reverse_complement,
        mutation_rate=mutation_rate,
        sv_rate=sv_rate,
        sv_min_length=sv_min_length,
        sv_max_length=sv_max_length,
        shuffle=shuffle,
        random_state=random_state,
    )

    X_gen = torch.utils.data.DataLoader(
        X_gen, pin_memory=pin_memory, num_workers=num_workers, batch_size=batch_size
    )

    return X_gen
