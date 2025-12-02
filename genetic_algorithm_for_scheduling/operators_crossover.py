import numpy as np

def order_crossover(p1, p2):
    n = len(p1)
    a, b = sorted(np.random.choice(n, 2, replace=False))
    child = [-1] * n
    child[a:b] = p1[a:b]

    pos = b
    for i in range(n):
        gene = p2[(b+i) % n]
        if gene not in child:
            child[pos % n] = gene
            pos += 1
    return np.array(child, dtype=int)


def pmx(p1, p2):
    n = len(p1)
    a, b = sorted(np.random.choice(n, 2, replace=False))
    child = -np.ones(n)

    child[a:b] = p1[a:b]
    mapping = {p1[i]: p2[i] for i in range(a, b)}

    for i in range(n):
        if child[i] == -1:
            gene = p2[i]
            while gene in mapping:
                gene = mapping[gene]
            child[i] = gene

    return np.array(child, dtype=int)


def cycle_crossover(p1, p2):
    n = len(p1)
    child = np.full(n, -1)
    index = 0
    cycle = True

    while -1 in child:
        start = np.where(child == -1)[0][0]
        idx = start
        while True:
            child[idx] = p1[idx] if cycle else p2[idx]
            val = p2[idx]
            idx = np.where(p1 == val)[0][0]
            if idx == start:
                break
        cycle = not cycle

    return np.array(child, dtype=int)