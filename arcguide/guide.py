"""The learned guide: task features -> P(primitive is used in the solution).

Trained only on tasks generated from the DSL itself (random programs on random grids). It never sees ARC.
"""
from __future__ import annotations

import numpy as np

from . import dsl

N_FEAT = 26


def _sym_lr(g):
    return float((g == np.fliplr(g)).all())


def _sym_ud(g):
    return float((g == np.flipud(g)).all())


def _is_subgrid(small, big):
    h, w = small.shape
    H, W = big.shape
    if h > H or w > W or (h, w) == (H, W):
        return 0.0
    for y in range(H - h + 1):
        for x in range(W - w + 1):
            if (big[y:y + h, x:x + w] == small).all():
                return 1.0
    return 0.0


def _has_separator(g):
    for a in (g, g.T):
        n = a.shape[0]
        if n % 2 == 1 and (a[n // 2] == a[n // 2][0]).all():
            return 1.0
    return 0.0


def _geo(i, o):
    for f in (lambda g: np.rot90(g, 1), lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 3), np.fliplr, np.flipud, lambda g: g.T):
        r = f(i)
        if r.shape == o.shape and (r == o).all():
            return 1.0
    return 0.0


def features(pairs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    rows = []
    for i, o in pairs:
        bi, bo = dsl.bg(i), dsl.bg(o)
        ci, co = set(np.unique(i).tolist()), set(np.unique(o).tolist())
        same = i.shape == o.shape
        rh, rw = o.shape[0] / i.shape[0], o.shape[1] / i.shape[1]
        ni, no = len(dsl.objects(i, bi)), len(dsl.objects(o, bo))
        rows.append([
            float(same), np.log2(rh), np.log2(rw),
            float(o.shape == i.shape[::-1] and i.shape[0] != i.shape[1]),
            float(o.size < i.size), float(o.size > i.size),
            float(rh in (2.0, 3.0) or rw in (2.0, 3.0)), float(rh in (0.5, 1 / 3) or rw in (0.5, 1 / 3)),
            float(co <= ci), float(bool(co - ci)), len(ci) / 10, len(co) / 10,
            float((i != o).mean()) if same else 0.0,
            _sym_lr(o), _sym_ud(o), _sym_lr(i),
            float((i == bi).mean()), float((o == bo).mean()),
            np.log1p(ni) / 4, np.log1p(no) / 4,
            _is_subgrid(o, i), _geo(i, o), _has_separator(i), float(len(co) <= 2), float(o.size == 1),
            float(same and sorted(np.unique(i, return_counts=True)[1].tolist()) == sorted(np.unique(o, return_counts=True)[1].tolist())),
        ])
    return np.array(rows, dtype=np.float32).mean(axis=0)


# ------------------------------------------------------------------ synthetic training data

def random_grid(rng, r: float | None = None) -> np.ndarray:
    h, w = rng.integers(3, 13), rng.integers(3, 13)
    r = rng.random() if r is None else r
    if r < 0.2:                       # small grids (kron_self, upscale, tiling)
        h, w = rng.integers(2, 6), rng.integers(2, 6)
    elif r < 0.35:                    # sizes divisible by 2 or 3 (downscale)
        k = rng.choice([2, 3])
        h, w = k * rng.integers(2, 5), k * rng.integers(2, 5)
    if rng.random() < 0.5 and h % 2:   # sometimes a separator layout, so the boolean/halves prims get examples
        w = 2 * rng.integers(2, 7) + 1
    g = np.zeros((h, w), np.int8)
    if rng.random() < 0.25:
        g[:] = rng.integers(0, 10, (h, w)) * (rng.random((h, w)) < 0.4)
    else:
        for _ in range(rng.integers(1, 6)):
            y, x = rng.integers(0, h), rng.integers(0, w)
            g[y:y + rng.integers(1, 4), x:x + rng.integers(1, 4)] = rng.integers(1, 10)
    if rng.random() < 0.3:
        mid = w // 2
        g[:, mid] = rng.integers(1, 10)
    return g


def synth_task(rng, max_depth=3):
    prog = tuple(rng.integers(0, dsl.N, rng.integers(1, max_depth + 1)).tolist())
    pairs = []
    regime = rng.random()             # one size regime per task, like a real task
    for _ in range(3):
        g = random_grid(rng, regime)
        o = g
        for k in prog:
            o = dsl.apply(k, o) if o is not None else None
        if o is None or (o.shape == g.shape and (o == g).all()):
            return None
        o = o.copy()
        o[o == dsl.NEW] = rng.integers(1, 10)
        pairs.append((g, o))
    return prog, pairs


def make_dataset(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    while len(X) < n:
        t = synth_task(rng)
        if t is None:
            continue
        prog, pairs = t
        X.append(features(pairs))
        y = np.zeros(dsl.N, np.float32)
        y[list(prog)] = 1
        Y.append(y)
    return np.stack(X), np.stack(Y)


# ------------------------------------------------------------------ tiny MLP (numpy)

class MLP:
    def __init__(self, n_in=N_FEAT, n_hidden=128, n_out=dsl.N, seed=0):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden)).astype(np.float32)
        self.b1 = np.zeros(n_hidden, np.float32)
        self.W2 = r.normal(0, np.sqrt(1 / n_hidden), (n_hidden, n_out)).astype(np.float32)
        self.b2 = np.zeros(n_out, np.float32)

    def forward(self, X):
        h = np.maximum(X @ self.W1 + self.b1, 0)
        return h, 1 / (1 + np.exp(-(h @ self.W2 + self.b2)))

    def predict(self, x):
        return self.forward(x[None])[1][0]

    def fit(self, X, Y, epochs=40, lr=2e-3, batch=256, seed=0):
        rng = np.random.default_rng(seed)
        params = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t = 0
        for _ in range(epochs):
            for idx in np.array_split(rng.permutation(len(X)), max(1, len(X) // batch)):
                x, y = X[idx], Y[idx]
                h, p = self.forward(x)
                d2 = (p - y) / len(x)
                gW2, gb2 = h.T @ d2, d2.sum(0)
                dh = (d2 @ self.W2.T) * (h > 0)
                gW1, gb1 = x.T @ dh, dh.sum(0)
                t += 1
                for k, g in enumerate((gW1, gb1, gW2, gb2)):
                    m[k] = 0.9 * m[k] + 0.1 * g
                    v[k] = 0.999 * v[k] + 0.001 * g * g
                    params[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
        return self

    def loss(self, X, Y):
        p = np.clip(self.forward(X)[1], 1e-6, 1 - 1e-6)
        return float(-(Y * np.log(p) + (1 - Y) * np.log(1 - p)).mean())

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        m = cls()
        m.W1, m.b1, m.W2, m.b2 = z["W1"], z["b1"], z["W2"], z["b2"]
        return m
