import json
from pathlib import Path

import numpy as np
import pytest

from arcguide import dsl, guide
from arcguide.search import PROGRAMS, blind_order, guided_order, search

ARC = Path(__file__).resolve().parents[1] / "data" / "arc" / "training"


def task(pairs, test):
    return {"train": [{"input": i, "output": o} for i, o in pairs], "test": [{"input": i, "output": o} for i, o in test]}


def test_every_primitive_is_safe_on_odd_grids():
    rng = np.random.default_rng(0)
    for _ in range(200):
        g = guide.random_grid(rng)
        for k in range(dsl.N):
            out = dsl.apply(k, g)
            assert out is None or (out.ndim == 2 and max(out.shape) <= dsl.MAX_SIDE)


def test_program_space_size():
    assert len(PROGRAMS) == dsl.N + dsl.N ** 2 + dsl.N ** 3


def test_finds_depth2_program_and_checks_test():
    g = [[1, 0], [2, 3]]
    f = lambda x: np.fliplr(np.rot90(np.array(x), -1)).tolist()
    t = task([(g, f(g)), ([[5, 6, 0]], f([[5, 6, 0]]))], [([[4, 4], [0, 7]], f([[4, 4], [0, 7]]))])
    r = search(t, blind_order(), 5000)
    assert r.program is not None and r.test_correct


def test_color_map_fitted_on_placeholder():
    sep = lambda a, b: [x + [5] + y for x, y in zip(a, b)]
    pairs = [(sep([[1, 0], [0, 0]], [[1, 1], [0, 0]]), [[2, 0], [0, 0]]),
             (sep([[0, 1], [1, 1]], [[0, 1], [0, 1]]), [[0, 2], [0, 2]])]
    t = task(pairs, [(sep([[1, 1], [1, 0]], [[1, 0], [1, 1]]), [[2, 0], [2, 0]])])
    r = search(t, blind_order(), 5000)
    assert r.name == "and_lr" and r.color_map[dsl.NEW] == 2 and r.test_correct


def test_unsolvable_task_reports_budget():
    rng = np.random.default_rng(1)
    pairs = [(rng.integers(0, 9, (4, 4)).tolist(), rng.integers(0, 9, (3, 5)).tolist()) for _ in range(2)]
    t = task(pairs, [(pairs[0][0], pairs[0][1])])
    r = search(t, blind_order(), 300)
    assert r.program is None and r.evals == 300


def test_guided_order_puts_likely_primitives_first():
    p = np.full(dsl.N, 0.01)
    k = dsl.NAMES.index("kron_self")
    p[k] = 0.9
    assert PROGRAMS[guided_order(p)[0]] == (k,)


def test_guide_learns_something():
    X, Y = guide.make_dataset(600, seed=3)
    m = guide.MLP().fit(X[:500], Y[:500], epochs=40)
    untrained = guide.MLP().loss(X[500:], Y[500:])
    assert m.loss(X[500:], Y[500:]) < 0.5 * untrained


@pytest.mark.skipif(not ARC.exists(), reason="run fetch_arc.py first")
@pytest.mark.parametrize("tid,prog", [("007bbfb7", "kron_self"), ("0520fde7", "and_lr"), ("3c9b0459", "rot180")])
def test_real_arc_tasks(tid, prog):
    t = json.loads((ARC / f"{tid}.json").read_text())
    r = search(t, blind_order(), 5000)
    assert r.name == prog and r.test_correct
