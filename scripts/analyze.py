"""Compute the reproduced results from results/<mode>_<site>/webarena.<id>/.

Usage (from the repo root):
    python scripts/analyze.py                                   # no_memory vs reasoningbank
    python scripts/analyze.py --modes no_memory,reasoningbank,reasoningbank_successonly

Outputs (printed, and saved to results/analysis/):
  Table 1  success rate (SR) and average steps per website x method
  Table 4  average steps for successful vs failed tasks
  Figure 1 cumulative SR along the task stream (figure1_cumulative_sr.png, needs matplotlib)
  Judge    accuracy of the LLM judge against the official grader (paper: 72.7%)
"""
import argparse
import csv
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITES = ["shopping", "shopping_admin", "reddit"]
sys.path.insert(0, os.path.join(REPO, "WebArena"))
from pipeline_state import summary_info  # noqa: E402


def load(mode, site, model):
    rows = []
    for d in glob.glob(os.path.join(REPO, "results", f"{mode}_{site}", "webarena.*")):
        s = summary_info(d)
        if s is None:
            continue  # incomplete/infrastructure failure; finished model failures count
        judge = None
        try:
            judge = json.load(open(os.path.join(d, f"{model}_autoeval.json")))[0].get("rm")
        except Exception:
            pass
        rows.append({"tid": int(d.rsplit(".", 1)[1]), "success": float(s["cum_reward"]) > 0,
                     "steps": int(s["n_steps"]), "judge": judge})
    return sorted(rows, key=lambda r: r["tid"])


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="no_memory,reasoningbank")
    ap.add_argument("--model", default="qwen3.7-flash")
    args = ap.parse_args()
    modes = args.modes.split(",")
    out_dir = os.path.join(REPO, "results", "analysis")
    os.makedirs(out_dir, exist_ok=True)

    data = {(m, s): load(m, s, args.model) for m in modes for s in SITES}

    # ---- Table 1 and Table 4 ----
    t1, t4 = [], []
    for m in modes:
        allrows = []
        for s in SITES:
            r = data[(m, s)]
            allrows += r
            t1.append([m, s, len(r), 100 * mean([x["success"] for x in r]), mean([x["steps"] for x in r])])
            t4.append([m, s, mean([x["steps"] for x in r if x["success"]]),
                       mean([x["steps"] for x in r if not x["success"]])])
        t1.append([m, "overall", len(allrows), 100 * mean([x["success"] for x in allrows]),
                   mean([x["steps"] for x in allrows])])
        t4.append([m, "overall", mean([x["steps"] for x in allrows if x["success"]]),
                   mean([x["steps"] for x in allrows if not x["success"]])])

    print("\nTable 1: success rate and average steps")
    print(f"{'method':28s}{'website':16s}{'n':>5s}{'SR(%)':>8s}{'steps':>8s}")
    for m, s, n, sr, st in t1:
        print(f"{m:28s}{s:16s}{n:5d}{sr:8.1f}{st:8.1f}")
    print("\nTable 4: average steps, successful vs failed tasks")
    print(f"{'method':28s}{'website':16s}{'success':>9s}{'failed':>9s}")
    for m, s, a, b in t4:
        print(f"{m:28s}{s:16s}{a:9.1f}{b:9.1f}")

    with open(os.path.join(out_dir, "table1.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["method", "website", "n", "SR", "avg_steps"]] + t1)
    with open(os.path.join(out_dir, "table4.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["method", "website", "steps_success", "steps_failed"]] + t4)

    # ---- Judge accuracy (LLM judge vs official grader) ----
    print("\nJudge accuracy (judge verdict vs official grader)")
    jrows = []
    for m in modes:
        r = [x for s in SITES for x in data[(m, s)] if x["judge"] is not None]
        tp = sum(x["judge"] and x["success"] for x in r)
        tn = sum((not x["judge"]) and (not x["success"]) for x in r)
        fp = sum(x["judge"] and not x["success"] for x in r)
        fn = sum((not x["judge"]) and x["success"] for x in r)
        acc = 100 * (tp + tn) / len(r) if r else float("nan")
        jrows.append([m, len(r), acc, tp, tn, fp, fn])
        print(f"  {m:28s} n={len(r):4d} acc={acc:5.1f}%  "
              f"TP={tp} TN={tn} FP(judge says success, grader fail)={fp} FN={fn}")
    with open(os.path.join(out_dir, "judge_accuracy.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["method", "n", "accuracy", "TP", "TN", "FP", "FN"]] + jrows)

    # ---- Figure 1: cumulative SR along the task stream ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed: run `uv add matplotlib` to draw Figure 1)")
        return
    fig, axes = plt.subplots(1, len(SITES), figsize=(5 * len(SITES), 3.6))
    for ax, s in zip(axes, SITES):
        for m in modes:
            r = data[(m, s)]
            if not r:
                continue
            cum, ok = [], 0
            for i, x in enumerate(r, 1):
                ok += x["success"]
                cum.append(100 * ok / i)
            ax.plot(range(1, len(r) + 1), cum, label=m)
        ax.set_title(s)
        ax.set_xlabel("tasks seen (in order)")
        ax.set_ylabel("cumulative SR (%)")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "figure1_cumulative_sr.png")
    fig.savefig(path, dpi=150)
    print(f"\nFigure saved to {os.path.relpath(path, REPO)}")


if __name__ == "__main__":
    main()
