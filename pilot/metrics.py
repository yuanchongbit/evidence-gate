import numpy as np


def softmax(logits):
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def hellinger(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    bc = np.sum(np.sqrt(np.clip(p, 0, None) * np.clip(q, 0, None)))
    return float(np.sqrt(max(0.0, 1.0 - bc)))


def auroc(scores, labels):
    """labels: 1 = positive class. Ties receive average ranks."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def aurc(scores, outcomes):
    """outcomes: bool array, True = correct answer. Abstention is encoded by
    dropping the item from `scores`; items kept here are answered items."""
    scores = np.asarray(scores, dtype=np.float64)
    outcomes = np.asarray(outcomes).astype(bool)
    if len(scores) == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    errs = (~outcomes[order]).astype(np.float64)
    cum_err = np.cumsum(errs)
    k = np.arange(1, len(scores) + 1, dtype=np.float64)
    return float(np.mean(cum_err / k))


def accuracy_at_coverage(scores, outcomes, target=0.8):
    scores = np.asarray(scores, dtype=np.float64)
    outcomes = np.asarray(outcomes).astype(bool)
    n = len(scores)
    if n == 0:
        return float("nan")
    k = max(1, int(round(target * n)))
    order = np.argsort(-scores, kind="mergesort")
    return float(outcomes[order[:k]].mean())


def ece(conf, correct, bins=10):
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def bootstrap_ci(values, stat=np.mean, n_boot=2000, alpha=0.05, seed=0):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    stats = np.array([
        stat(values[rng.integers(0, len(values), len(values))]) for _ in range(n_boot)
    ])
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def paired_auroc_diff_ci(scores_a, scores_b, labels, n_boot=2000, alpha=0.05, seed=0):
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    point = auroc(scores_a, labels) - auroc(scores_b, labels)
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(labels)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d = auroc(scores_a[idx], labels[idx]) - auroc(scores_b[idx], labels[idx])
        if not np.isnan(d):
            diffs.append(d)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)
