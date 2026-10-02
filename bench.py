"""Train the guide on self-generated tasks, then compare blind vs guided search on public ARC tasks.

python fetch_arc.py && python bench.py            # full: 400 training + 400 evaluation tasks
python bench.py --quick                           # 60 training tasks, smaller guide (for a fast check)
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # one BLAS thread per worker process
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from arcguide import dsl, guide
from arcguide.search import blind_order, guided_order, search

HERE = Path(__file__).parent
ARC = HERE / "data" / "arc"
OUT = HERE / "out"
BUDGET = 5000
_MODEL = None


def _init(model_path):
    global _MODEL
    _MODEL = guide.MLP.load(model_path)


def _one(path: str) -> dict:
    task = json.loads(Path(path).read_text())
    pairs = [(np.array(p["input"], np.int8), np.array(p["output"], np.int8)) for p in task["train"]]
    prior = _MODEL.predict(guide.features(pairs))
    b = search(task, blind_order(), BUDGET)
    g = search(task, guided_order(prior), BUDGET)
    top = [dsl.NAMES[i] for i in np.argsort(-prior)[:3]]
    return {"task": Path(path).stem, "set": Path(path).parent.name,
            "blind": {"program": b.name, "evals": b.evals, "test": b.test_correct},
            "guided": {"program": g.name, "evals": g.evals, "test": g.test_correct}, "guide_top3": top}


def summarize(rows, split):
    rs = [r for r in rows if r["set"] == split]
    out = {"tasks": len(rs)}
    for k in ("blind", "guided"):
        found = [r for r in rs if r[k]["test"] is not None]
        out[k] = {"consistent": len(found), "test_correct": sum(1 for r in found if r[k]["test"]),
                  "total_evals": int(sum(r[k]["evals"] for r in rs))}
    both = [r for r in rs if r["blind"]["test"] is not None and r["guided"]["test"] is not None]
    if both:
        ratio = [r["blind"]["evals"] / r["guided"]["evals"] for r in both]
        out["both_solved"] = len(both)
        out["median_speedup"] = round(float(np.median(ratio)), 1)
        out["blind_median_evals"] = float(np.median([r["blind"]["evals"] for r in both]))
        out["guided_median_evals"] = float(np.median([r["guided"]["evals"] for r in both]))
    out["guided_only"] = sorted(r["task"] for r in rs if r["guided"]["test"] and not r["blind"]["test"])
    out["blind_only"] = sorted(r["task"] for r in rs if r["blind"]["test"] and not r["guided"]["test"])
    return out


def main():
    quick = "--quick" in sys.argv
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    n = 3000 if quick else 20000
    X, Y = guide.make_dataset(n, seed=0)
    split = int(0.9 * n)
    model = guide.MLP().fit(X[:split], Y[:split], epochs=15 if quick else 40)
    print(f"guide trained on {split} self-generated tasks in {time.time() - t0:.0f}s · "
          f"held-out BCE {model.loss(X[split:], Y[split:]):.3f} (vs {guide.MLP().fit(X[:1], Y[:1], epochs=0).loss(X[split:], Y[split:]):.3f} untrained)")
    model.save(OUT / "guide.npz")

    files = sorted((ARC / "training").glob("*.json"))
    if quick:
        files = files[:60]
    else:
        files += sorted((ARC / "evaluation").glob("*.json"))
    t1 = time.time()
    with Pool(min(8, os.cpu_count() or 1), initializer=_init, initargs=(str(OUT / "guide.npz"),)) as pool:
        rows = pool.map(_one, [str(f) for f in files], chunksize=4)
    secs = time.time() - t1

    summary = {"budget": BUDGET, "primitives": dsl.N, "max_depth": 3, "search_seconds": round(secs),
               "splits": {s: summarize(rows, s) for s in sorted({r["set"] for r in rows})}}
    (OUT / "results.json").write_text(json.dumps({"summary": summary, "tasks": rows}, indent=1))
    print(f"\nsearched {len(rows)} tasks in {secs:.0f}s (budget {BUDGET} programs/task, {dsl.N} primitives, depth ≤ 3)\n")
    print(f"{'split':12} {'tasks':>5} | {'blind solved':>12} {'guided solved':>13} | {'blind evals':>11} {'guided evals':>12} | median speed-up")
    for s, v in summary["splits"].items():
        print(f"{s:12} {v['tasks']:5} | {v['blind']['test_correct']:12} {v['guided']['test_correct']:13} | "
              f"{v['blind']['total_evals']:11,} {v['guided']['total_evals']:12,} | {v.get('median_speedup', '-')}x "
              f"(on {v.get('both_solved', 0)} tasks both solve)")
        if v["guided_only"]:
            print(f"   only guided solved: {', '.join(v['guided_only'])}")
        if v["blind_only"]:
            print(f"   only blind solved:  {', '.join(v['blind_only'])}")


if __name__ == "__main__":
    main()
