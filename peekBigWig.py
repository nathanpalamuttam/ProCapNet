import pyBigWig

# Open the bigWig file
bw = pyBigWig.open("data/procap/processed/K562/5prime.neg.bigWig")

# Print genome summary info
print("Chromosomes:", list(bw.chroms().keys())[:5])
print("File summary:", bw.header())

# Peek at the first few intervals
chrom = list(bw.chroms().keys())[0]  # e.g. "chr1"
entries = bw.intervals(chrom, 0, 100000)
print(entries[:10])  # first 10 intervals with (start, end, value)