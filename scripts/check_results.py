"""Read-only check of actor, judge, and memory-extraction completion.

Usage (from the repo root):
    python scripts/check_results.py no_memory
    python scripts/check_results.py no_memory --clean     # also move leftover folders of interrupted tasks away

For every website it reports:
  - done:        every required stage completed (including memory for RB)
  - missing:     no result folder (never started, or interrupted / timed out)
  - agent_error: the agent crashed (e.g. API failure); summary_info.json has an err_msg
  - not_judged:  result exists but the judge verdict is missing or failed
  - not_extracted: judge finished, but this task's memory is missing
  - legacy:     old RB results cannot establish the original memory commit order
  - inconsistent: artifacts or memory disagree with the saved stage history
  - leftovers:   timestamped folders left behind by interrupted tasks (safe to remove)
"""
import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITES = ["shopping", "shopping_admin", "reddit"]
sys.path.insert(0, os.path.join(REPO, "WebArena"))
from pipeline_state import MANIFEST_NAME, read_json, task_status  # noqa: E402


def site_task_ids(site):
    ids = []
    for f in glob.glob(os.path.join(REPO, "WebArena", "config_files", "*.json")):
        name = os.path.basename(f).split(".")[0]
        if not name.isdigit():
            continue
        cfg = json.load(open(f))
        if cfg["sites"] == [site]:
            ids.append(int(cfg["task_id"]))
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", help="no_memory, reasoningbank, reasoningbank_successonly, ...")
    ap.add_argument("--model", default="qwen3.7-flash")
    ap.add_argument("--sites", default=",".join(SITES))
    ap.add_argument("--memory-dir", help="Override the memory directory for this result prefix.")
    ap.add_argument("--clean", action="store_true", help="move leftover folders to results/_incomplete/")
    args = ap.parse_args()

    for site in args.sites.split(","):
        out = os.path.join(REPO, "results", f"{args.mode}_{site}")
        ids = site_task_ids(site)
        manifest_path = Path(out) / MANIFEST_NAME
        manifest = read_json(manifest_path) if manifest_path.exists() else None
        if manifest is not None:
            config = manifest["config"]
            mode = config["mode"]
            success_only = config["success_only"]
            criteria = config["criteria"]
            ids = config["task_ids"]
            memory_dir = config["memory_dir"]
            if args.model != config["model"]:
                print(f"=== {site}: model mismatch; run used {config['model']}, requested {args.model}")
                continue
        else:
            mode = next((m for m in ("no_memory", "reasoningbank", "awm", "synapse")
                         if args.mode == m or args.mode.startswith(m + "_")), None)
            if mode is None:
                ap.error("Unknown result prefix; use no_memory/reasoningbank/awm/synapse[_run_id]")
            success_only = "successonly" in args.mode.split("_")
            criteria = "autoeval"
            memory_dir = os.path.join(REPO, "WebArena", f"memories_{args.mode}")
        memory_dir = args.memory_dir or memory_dir
        memory_path = os.path.join(memory_dir, f"{site}.jsonl") if memory_dir else None
        groups = {k: [] for k in ("done", "missing", "agent_error", "not_judged",
                                  "not_extracted", "not_committed", "legacy", "inconsistent")}
        for tid in ids:
            status = task_status(os.path.join(out, f"webarena.{tid}"), args.model,
                                 memory_path=memory_path, success_only=success_only,
                                 mode=mode, criteria=criteria, manifest=manifest)
            groups[status].append(tid)

        leftovers = [p for p in glob.glob(os.path.join(out, "*")) if os.path.isdir(p)
                     and not os.path.basename(p).startswith("webarena.")
                     and os.path.basename(p) != "pipeline_checkpoints"]

        print(f"=== {site} ({args.mode}): {len(groups['done'])}/{len(ids)} done")
        for k in groups:
            if k == "done":
                continue
            v = groups[k]
            shown = ",".join(map(str, v[:40])) + (" ..." if len(v) > 40 else "")
            print(f"  {k:12s} {len(v):4d}  {shown}")
        if groups["agent_error"]:
            tid = groups["agent_error"][0]
            try:
                msg = read_json(Path(out) / f"webarena.{tid}" / "summary_info.json").get("err_msg")
            except (OSError, ValueError, AttributeError):
                msg = "invalid summary"
            print(f"  e.g. error of task {tid}: {str(msg)[:200]}")
        print(f"  leftovers    {len(leftovers):4d}")
        if args.clean and leftovers:
            if manifest is not None:
                print("  active pipeline history found: preserving interrupted-task folders")
                continue
            dest = os.path.join(REPO, "results", "_incomplete", f"{args.mode}_{site}")
            os.makedirs(dest, exist_ok=True)
            for p in leftovers:
                shutil.move(p, os.path.join(dest, os.path.basename(p)))
            print(f"  moved leftovers to {os.path.relpath(dest, REPO)}")
        prob = os.path.join(out, "problems.jsonl")
        if os.path.exists(prob):
            n = sum(1 for _ in open(prob))
            print(f"  problems.jsonl: {n} entries (timeouts / crashed steps)")


if __name__ == "__main__":
    main()
