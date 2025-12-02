import numpy as np

def swap_mutation(chrom):
    a, b = np.random.choice(len(chrom), 2, replace=False)
    chrom[a], chrom[b] = chrom[b], chrom[a]
    return chrom


def inversion_mutation(chrom):
    a, b = sorted(np.random.choice(len(chrom), 2, replace=False))
    chrom[a:b] = chrom[a:b][::-1]
    return chrom


def scramble_mutation(chrom):
    a, b = sorted(np.random.choice(len(chrom), 2, replace=False))
    segment = chrom[a:b]
    np.random.shuffle(segment)
    chrom[a:b] = segment
    return chrom