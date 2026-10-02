"""A small grid DSL for ARC. Every primitive is grid -> grid (or None if it doesn't apply).

Colour 10 is a placeholder ("new colour"); the fitted colour map at the end of a program decides what
it becomes. That lets one primitive like `fill_holes` cover tasks that fill with red, blue, ...
"""
from __future__ import annotations

from collections import deque

import numpy as np

G = np.ndarray
NEW = 10
MAX_SIDE = 30


def bg(g: G) -> int:
    """ARC convention: black (0) is background when present; otherwise the most common colour."""
    if (g == 0).any():
        return 0
    v, c = np.unique(g, return_counts=True)
    return int(v[c.argmax()])


def _ok(g: G | None) -> G | None:
    if g is None or g.size == 0 or g.ndim != 2 or max(g.shape) > MAX_SIDE:
        return None
    return g


def objects(g: G, b: int | None = None) -> list[tuple[int, list[tuple[int, int]]]]:
    """Same-colour 4-connected components of non-background cells."""
    b = bg(g) if b is None else b
    seen = np.zeros(g.shape, bool)
    out = []
    H, W = g.shape
    for r in range(H):
        for c in range(W):
            if seen[r, c] or g[r, c] == b:
                continue
            col, cells, q = g[r, c], [], deque([(r, c)])
            seen[r, c] = True
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] == col:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            out.append((int(col), cells))
    return out


def _crop_cells(g: G, cells) -> G:
    ys, xs = zip(*cells)
    return g[min(ys):max(ys) + 1, min(xs):max(xs) + 1]


def crop(g):
    b = bg(g)
    ys, xs = np.nonzero(g != b)
    if len(ys) == 0:
        return None
    return g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def obj_largest(g):
    o = objects(g)
    return _crop_cells(g, max(o, key=lambda x: len(x[1]))[1]) if o else None


def obj_smallest(g):
    o = objects(g)
    return _crop_cells(g, min(o, key=lambda x: len(x[1]))[1]) if o else None


def _down(g, k):
    if g.shape[0] % k or g.shape[1] % k:
        return None
    return g[::k, ::k]


def _gravity(g, axis, rev):
    b = bg(g)
    out = np.full_like(g, b)
    a = g if axis == 0 else g.T
    o = out if axis == 0 else out.T
    for j in range(a.shape[1]):
        col = a[:, j][a[:, j] != b]
        if rev:
            o[:len(col), j] = col
        else:
            o[a.shape[0] - len(col):, j] = col
    return out


def _split(g, horizontal: bool):
    """Two halves, skipping a uniform separator line in the middle if there is one."""
    a = g if horizontal else g.T
    n = a.shape[0]
    if n % 2 == 1:
        mid = a[n // 2]
        if not (mid == mid[0]).all():
            return None
        p, q = a[:n // 2], a[n // 2 + 1:]
    else:
        p, q = a[:n // 2], a[n // 2:]
    return (p, q) if horizontal else (p.T, q.T)


def _bool(op, horizontal):
    def f(g):
        s = _split(g, horizontal)
        if s is None:
            return None
        b = bg(g)
        m = op(s[0] != b, s[1] != b)
        return np.where(m, NEW, 0).astype(g.dtype)
    return f


def _half(which):
    def f(g):
        s = _split(g, which in ("top", "bottom"))
        if s is None:
            return None
        return s[0] if which in ("top", "left") else s[1]
    return f


def fill_holes(g):
    """Background cells not reachable from the border become NEW."""
    b = bg(g)
    H, W = g.shape
    reach = np.zeros(g.shape, bool)
    q = deque([(r, c) for r in range(H) for c in (0, W - 1)] + [(r, c) for c in range(W) for r in (0, H - 1)])
    while q:
        y, x = q.popleft()
        if 0 <= y < H and 0 <= x < W and not reach[y, x] and g[y, x] == b:
            reach[y, x] = True
            q.extend(((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)))
    out = g.copy()
    out[(g == b) & ~reach] = NEW
    return out


def compress(g):
    b = bg(g)
    rows = ~(g == b).all(axis=1)
    cols = ~(g == b).all(axis=0)
    return g[rows][:, cols] if rows.any() and cols.any() else None


def dedupe(g):
    keep_r = [0] + [i for i in range(1, g.shape[0]) if not (g[i] == g[i - 1]).all()]
    g = g[keep_r]
    keep_c = [0] + [j for j in range(1, g.shape[1]) if not (g[:, j] == g[:, j - 1]).all()]
    return g[:, keep_c]


def majority(g):
    b = bg(g)
    v = g[g != b]
    if v.size == 0:
        return None
    vals, cnt = np.unique(v, return_counts=True)
    return np.array([[vals[cnt.argmax()]]], dtype=g.dtype)


def invert(g):
    b = bg(g)
    return np.where(g == b, NEW, b).astype(g.dtype)


def outline(g):
    b = bg(g)
    p = np.pad(g != b, 1)
    inner = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    out = g.copy()
    out[inner] = b
    return out


PRIMS: list[tuple[str, callable]] = [
    ("rot90", lambda g: np.rot90(g, -1)), ("rot180", lambda g: np.rot90(g, 2)), ("rot270", lambda g: np.rot90(g, 1)),
    ("flip_lr", np.fliplr), ("flip_ud", np.flipud), ("transpose", lambda g: g.T), ("antitranspose", lambda g: np.rot90(g, 2).T),
    ("crop", crop), ("obj_largest", obj_largest), ("obj_smallest", obj_smallest),
    ("upscale2", lambda g: np.kron(g, np.ones((2, 2), g.dtype))), ("upscale3", lambda g: np.kron(g, np.ones((3, 3), g.dtype))),
    ("downscale2", lambda g: _down(g, 2)), ("downscale3", lambda g: _down(g, 3)),
    ("tile_h", lambda g: np.hstack([g, g])), ("tile_v", lambda g: np.vstack([g, g])), ("tile_2x2", lambda g: np.tile(g, (2, 2))),
    ("mirror_h", lambda g: np.hstack([g, np.fliplr(g)])), ("mirror_v", lambda g: np.vstack([g, np.flipud(g)])),
    ("mirror_4", lambda g: np.vstack([np.hstack([g, np.fliplr(g)]), np.flipud(np.hstack([g, np.fliplr(g)]))])),
    ("kron_self", lambda g: np.kron((g != bg(g)).astype(g.dtype), g) + np.kron((g == bg(g)).astype(g.dtype), np.full_like(g, bg(g)))),
    ("gravity_down", lambda g: _gravity(g, 0, False)), ("gravity_up", lambda g: _gravity(g, 0, True)),
    ("gravity_right", lambda g: _gravity(g, 1, False)), ("gravity_left", lambda g: _gravity(g, 1, True)),
    ("top_half", _half("top")), ("bottom_half", _half("bottom")), ("left_half", _half("left")), ("right_half", _half("right")),
    ("and_lr", _bool(np.logical_and, False)), ("or_lr", _bool(np.logical_or, False)),
    ("xor_lr", _bool(np.logical_xor, False)), ("nor_lr", _bool(lambda a, b: ~(a | b), False)),
    ("and_tb", _bool(np.logical_and, True)), ("or_tb", _bool(np.logical_or, True)),
    ("xor_tb", _bool(np.logical_xor, True)), ("nor_tb", _bool(lambda a, b: ~(a | b), True)),
    ("fill_holes", fill_holes), ("compress", compress), ("dedupe", dedupe), ("majority", majority),
    ("invert", invert), ("outline", outline),
]
NAMES = [n for n, _ in PRIMS]
N = len(PRIMS)


def apply(i: int, g: G) -> G | None:
    try:
        return _ok(np.ascontiguousarray(PRIMS[i][1](g)))
    except (ValueError, IndexError):
        return None


def fit_color_map(preds: list[G], targets: list[G]) -> dict[int, int] | None:
    """A single colour->colour map that turns every prediction into its target, if one exists."""
    m: dict[int, int] = {}
    for p, t in zip(preds, targets):
        if p.shape != t.shape:
            return None
        for a, b in zip(p.ravel().tolist(), t.ravel().tolist()):
            if m.setdefault(a, b) != b:
                return None
    return m


def recolor(g: G, m: dict[int, int]) -> G | None:
    out = g.copy()
    for a, b in m.items():
        out[g == a] = b
    if (out == NEW).any():
        return None
    return out
