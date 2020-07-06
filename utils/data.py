from collections import Counter
import numpy as np

def gen(X, y, augment, balanced):
    n = len(X)

    if balanced:
        counter = Counter(y)
        weights = np.array([1 / counter[val] for val in y])
        weights /= weights.sum()
        idxs = np.random.choice(range(n), n, p=weights)
    else:
        idxs = list(range(n))

    offsets = np.random.randint(12, size=n)
    np.random.shuffle(idxs)

    l = list(range(12))

    ls = []
    for i in range(12):
        ls.append(l[i:] + l[:i])

    for idx, i in zip(idxs, offsets):
        if augment:
            yield X[idx].take(ls[i], axis=0), y[idx]
        else:
            yield X[idx], y[idx]


