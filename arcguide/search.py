"""Enumerative program search over the DSL, blind or guided by a learned prior."""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from . import dsl

MAX_DEPTH = 3


@dataclass
class Result:
    program: tuple[int, ...] | None
    evals: int                    # candidate programs checked before the first consistent one (or budget)
    color_map: dict | None = None
    test_correct: bool | None = None

    @property
    def name(self) -> str:
        return " → ".join(dsl.NAMES[i] for i in self.program) if self.program else "—"


def all_programs() -> list[tuple[int, ...]]:
    progs = []
    for d in range(1, MAX_DEPTH + 1):
        progs += list(itertools.product(range(dsl.N), repeat=d))
    return progs


PROGRAMS = all_programs()
_LEN = np.array([len(p) for p in PROGRAMS])
_IDX = np.full((len(PROGRAMS), MAX_DEPTH), -1)
for k, p in enumerate(PROGRAMS):
    _IDX[k, :len(p)] = p


def blind_order() -> np.ndarray:
    """Shortest programs first, fixed order inside each depth (breadth-first)."""
    return np.arange(len(PROGRAMS))


def guided_order(p: np.ndarray) -> np.ndarray:
    """Best-first by summed log-probability of the primitives a program uses."""
    lp = np.log(np.clip(p, 1e-4, 1 - 1e-4))
    score = np.where(_IDX >= 0, lp[np.maximum(_IDX, 0)], 0.0).sum(axis=1)
    return np.argsort(-score, kind="stable")


def run_program(prog, grids, cache):
    """Apply a program to every grid, reusing cached prefixes."""
    key = tuple(prog)
    if key in cache:
        return cache[key]
    prev = grids if len(prog) == 1 else run_program(prog[:-1], grids, cache)
    out = None if prev is None else [dsl.apply(prog[-1], g) for g in prev]
    if out is not None and any(o is None for o in out):
        out = None
    cache[key] = out
    return out


def search(task: dict, order: np.ndarray, budget: int) -> Result:
    xs = [np.array(p["input"], dtype=np.int8) for p in task["train"]]
    ys = [np.array(p["output"], dtype=np.int8) for p in task["train"]]
    cache: dict = {}
    for n, k in enumerate(order[:budget], start=1):
        prog = PROGRAMS[k]
        outs = run_program(prog, xs, cache)
        if outs is None:
            continue
        if all(o.shape == y.shape and (o == y).all() for o, y in zip(outs, ys)):
            return _finish(task, prog, None, n)
        m = dsl.fit_color_map(outs, ys)
        if m is not None and any(a != b for a, b in m.items()):
            return _finish(task, prog, m, n)
    return Result(None, min(budget, len(order)))


def _finish(task, prog, m, n) -> Result:
    ok = True
    for t in task["test"]:
        g = np.array(t["input"], dtype=np.int8)
        for i in prog:
            g = dsl.apply(i, g) if g is not None else None
        if g is not None and m:
            g = dsl.recolor(g, m)
        want = np.array(t["output"], dtype=np.int8) if "output" in t else None
        ok &= g is not None and want is not None and g.shape == want.shape and bool((g == want).all())
    return Result(tuple(prog), n, m, ok)
