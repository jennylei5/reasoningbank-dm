# Copyright 2026 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import json
import signal
import argparse
import datetime
import fcntl
import math
import shutil
import sys
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from subprocess import Popen, TimeoutExpired

from pipeline_state import (
    MANIFEST_NAME, PipelineError, fingerprint, judge_info, memory_record,
    needs_extraction, read_json, read_records, record_fingerprint, summary_info,
    task_status, trajectory_fingerprint, write_json,
)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


@dataclass
class StageResult:
    returncode: int | None
    timed_out: bool = False
    error: str | None = None

    @property
    def ok(self):
        return self.returncode == 0 and not self.timed_out and self.error is None


def run_with_timeout(cmd, timeout_s, tid, step, log_path):
    """Run a subprocess; kill it (and the browser it started) if it exceeds timeout_s.

    Timed-out or failed steps are appended to log_path so they can be re-run later.
    """
    started = now()
    process = None
    try:
        process = Popen(cmd, start_new_session=True)
        result = StageResult(process.wait(timeout=timeout_s))
    except OSError as exc:
        result = StageResult(None, error=str(exc))
    except TimeoutExpired:
        if step == "run" and hasattr(signal, "SIGUSR1"):
            print(f"[pipeline] task {tid}: collecting actor stack before timeout cleanup", flush=True)
            try:
                # Signal only the Python actor, never its browser process group.
                # run.py registers faulthandler before its heavyweight imports.
                os.kill(process.pid, signal.SIGUSR1)
                process.wait(timeout=0.25)
            except (ProcessLookupError, TimeoutExpired):
                pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        result = StageResult(process.wait(), timed_out=True)
    except BaseException:
        # An interrupted parent must not leave an actor writing to a shared bank.
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise
    if not result.ok:
        code = "timeout" if result.timed_out else ("launch_error" if result.error else str(result.returncode))
        print(f"[pipeline] task {tid}: step '{step}' ended with {code}", flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps({"task_id": tid, "step": step, "result": code,
                               "error": result.error, "started": started, "time": now()}) + "\n")
    return result


def is_done(output_dir, tid, model, **kwargs):
    return task_status(Path(output_dir) / f"webarena.{tid}", model, **kwargs) == "done"


@contextmanager
def exclusive_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PipelineError(f"Another pipeline is using {path}") from exc
        try:
            yield f
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def bank_paths(args):
    if args.memory_mode == "no_memory":
        return None
    directory = Path(f"memories_{args.memory_mode}{args.memory_suffix}")
    return {
        "memory": directory / f"{args.website}.jsonl",
        "embeddings": directory / f"{args.website}_embeddings.jsonl",
        "prompt": directory / f"{args.website}.txt",
    }


def selected_reward(args, directory):
    if args.judge == "gt":
        return summary_info(directory)["cum_reward"]
    return judge_info(directory, args.model)["rm"]


def checkpoint_bank(paths, output, tid):
    """Keep evidence for a failed actor; never restore a live website implicitly."""
    checkpoint = output / "pipeline_checkpoints" / str(tid)
    checkpoint.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for name, path in (paths or {}).items():
        metadata[name] = {"path": str(path.resolve()), "existed": path.exists()}
        if path.exists():
            metadata[name]["sha256"] = fingerprint(path)
    if (checkpoint / "files.json").exists():
        if read_json(checkpoint / "files.json") != metadata:
            raise PipelineError(f"Bank changed since task {tid}'s pre-actor checkpoint")
        return
    for name, path in (paths or {}).items():
        if path.exists():
            shutil.copyfile(path, checkpoint / name)
    write_json(checkpoint / "files.json", metadata)


def check_bank(args, manifest, paths, current_tid=None, include_current_embedding=False):
    """Reject local resume pollution without changing upstream retrieval policy."""
    if paths is None:
        return
    expected_memory, expected_embeddings = [], []
    for tid in manifest["config"]["task_ids"]:
        state = manifest["tasks"].get(str(tid), {})
        if state.get("complete"):
            expected_embeddings.append(str(tid))
            if state.get("memory") is not None:
                expected_memory.append(str(tid))
        elif tid != current_tid and state:
            raise PipelineError(f"Task {tid} is unfinished; do not resume out of order")
    if include_current_embedding:
        expected_embeddings.append(str(current_tid))
    memories = read_records(paths["memory"])
    embeddings = read_records(paths["embeddings"])
    actual_memory = [str(r.get("task_id")) for r in memories]
    actual_embeddings = [str(r.get("id")) for r in embeddings]
    # A completed extraction may be present if the parent stopped before it
    # recorded the commit. Only the current task may be this one extra record.
    allowed = [expected_memory]
    if current_tid is not None:
        allowed.append(expected_memory + [str(current_tid)])
    if actual_memory not in allowed or actual_embeddings != expected_embeddings:
        raise PipelineError("Memory/cache order or IDs disagree with this run. "
                            "Do not reuse an old bank or rerun the actor; preserve the files.")
    for record in memories:
        state = manifest["tasks"].get(str(record["task_id"]), {})
        if state.get("complete") and state.get("memory") != record_fingerprint(record):
            raise PipelineError("Previously committed memory was modified")


def run_stream(args, task_ids):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = bank_paths(args)
    manifest_path = output / MANIFEST_NAME
    config = {
        "website": args.website, "mode": args.memory_mode, "model": args.model,
        "criteria": args.judge, "success_only": args.success_only,
        "memory_dir": str(paths["memory"].parent.resolve()) if paths else None,
        "task_ids": task_ids,
        "api_base": os.environ.get("OPENAI_BASE_URL"),
        "embedding_model": os.environ.get("RB_EMBED_MODEL"),
        "grader_model": os.environ.get("RB_GRADER_MODEL"),
        "websites": {k: v for k, v in os.environ.items() if k.startswith("WA_")},
    }
    with ExitStack() as stack:
        stack.enter_context(exclusive_lock(output / ".pipeline.lock"))
        bank_lock = None
        if paths:
            bank_lock = stack.enter_context(exclusive_lock(
                paths["memory"].parent / f".{args.website}.pipeline.lock"))
            bank_lock.seek(0)
            owner = bank_lock.read().strip()
            if owner and owner != str(output):
                raise PipelineError("This bank belongs to another output directory. "
                                    "Use a new RB_RUN_ID or --memory_suffix.")
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest.get("version") != 1 or manifest.get("config") != config:
                raise PipelineError("Run configuration changed. Resume with the original "
                                    "task selection/settings, or use a new run directory and bank.")
        else:
            if list(output.glob("webarena.*")) or list(output.glob("*_GenericAgentArgs_*")):
                raise PipelineError("Legacy results have no stage history. Keep them and use "
                                    "RB_RUN_ID=fixed_v1 for a new run; do not overwrite old results.")
            if paths and any(p.exists() and p.stat().st_size for p in paths.values()):
                raise PipelineError("A new run needs an empty, separate memory bank. "
                                    "Use a new RB_RUN_ID or --memory_suffix.")
            manifest = {"version": 1, "run_id": uuid.uuid4().hex,
                        "created": now(), "config": config, "tasks": {}}
            write_json(manifest_path, manifest)
        if bank_lock:
            bank_lock.seek(0)
            bank_lock.truncate()
            bank_lock.write(str(output))
            bank_lock.flush()

        manifest.setdefault("runtime_history", []).append({
            "started_at": now(), "task_timeout_seconds": args.task_timeout,
            "judge_extract_timeout_seconds": args.step_timeout,
        })
        write_json(manifest_path, manifest)

        problem_log = output / "problems.jsonl"
        for tid in task_ids:
            state = manifest["tasks"].setdefault(str(tid), {})
            directory = output / f"webarena.{tid}"
            if state.get("complete"):
                if not is_done(output, tid, args.model, memory_path=paths["memory"] if paths else None,
                               success_only=args.success_only, mode=args.memory_mode,
                               criteria=args.judge, manifest=manifest):
                    raise PipelineError(f"Completed task {tid}'s artifacts were changed")
                continue

            summary = summary_info(directory)
            if state.get("trajectory"):
                if trajectory_fingerprint(directory) != state["trajectory"]:
                    raise PipelineError(f"Task {tid}'s actor trajectory was changed")
            elif state.get("actor_started"):
                if summary is None or state.get("actor_failed"):
                    raise PipelineError(
                        f"Actor {tid} was interrupted. Browser state cannot be restored by this "
                        "pipeline. Preserve this run/checkpoint and restart with a new run ID "
                        "after restoring the website; do not append a retry to this stream.")
                # Parent may have stopped after run.py finished renaming its output.
                state["trajectory"] = trajectory_fingerprint(directory)
                write_json(manifest_path, manifest)
            else:
                check_bank(args, manifest, paths)
                if directory.exists():
                    raise PipelineError(f"Unexpected result directory; refusing to overwrite {directory}")
                checkpoint_bank(paths, output, tid)
                state["actor_started"] = now()
                write_json(manifest_path, manifest)
                command = [sys.executable, "run.py", "--task_name", f"webarena.{tid}",
                           "--model_name", args.model, "--results_path", str(output)]
                if paths:
                    command += ["--memory_path", str(paths["prompt"])]
                result = run_with_timeout(command, args.task_timeout, tid, "run", problem_log)
                if not result.ok or summary_info(directory) is None:
                    state["actor_failed"] = True
                    write_json(manifest_path, manifest)
                    raise PipelineError(f"Actor {tid} failed. Stopped before judge/extract and later tasks; "
                                        "preserve this run and restore the website before a new run.")
                state["trajectory"] = trajectory_fingerprint(directory)
                write_json(manifest_path, manifest)

            check_bank(args, manifest, paths, tid, include_current_embedding=True)
            judge_path = directory / f"{args.model}_autoeval.json"
            if state.get("judge"):
                if judge_info(directory, args.model) is None or fingerprint(judge_path) != state["judge"]:
                    raise PipelineError(f"Task {tid}'s judge artifact was changed")
            else:
                # A judge cannot change the browser; resume it on the same actor trace.
                if judge_info(directory, args.model) is None:
                    result = run_with_timeout([
                        sys.executable, "-m", "autoeval.evaluate_trajectory",
                        "--result_dir", str(directory), "--model", args.model,
                        "--log_dir", f"autoeval/logs_{args.memory_mode}{args.memory_suffix}_{args.website}",
                    ], args.step_timeout, tid, "judge", problem_log)
                    if not result.ok or judge_info(directory, args.model) is None:
                        raise PipelineError(f"Judge {tid} failed. Resume the same command; actor will be kept.")
                state["judge"] = fingerprint(judge_path)
                write_json(manifest_path, manifest)

            reward = selected_reward(args, directory)
            if needs_extraction(args.memory_mode, args.success_only, reward):
                record = memory_record(paths["memory"], tid, reward)
                if record is None:
                    state["extract_started"] = now()
                    write_json(manifest_path, manifest)
                    result = run_with_timeout([
                        sys.executable, "induce_memory.py", "--result_dir", str(output),
                        "--task", f"webarena.{tid}", "--criteria", args.judge,
                        "--memory_mode", args.memory_mode, "--model", args.model,
                        "--output_path", str(paths["memory"]),
                    ] + (["--success_only"] if args.success_only else []),
                        args.step_timeout, tid, "extract", problem_log)
                    if not result.ok:
                        raise PipelineError(f"Extraction {tid} failed. Resume the same command; actor/judge will be kept.")
                    record = memory_record(paths["memory"], tid, reward)
                    if record is None:
                        raise PipelineError(f"Extraction {tid} produced no memory. No later task was started.")
                elif not state.get("extract_started"):
                    raise PipelineError(f"Unexpected memory for task {tid}; refusing to adopt another attempt")
                state["memory"] = record_fingerprint(record)
            else:
                state["memory"] = None
            check_bank(args, manifest, paths, tid, include_current_embedding=True)
            state["complete"] = True
            state["completed_at"] = now()
            write_json(manifest_path, manifest)
            print(f"[pipeline] task {tid}: all required stages committed", flush=True)
        check_bank(args, manifest, paths)


def main(args):
    config_files = sorted(Path("config_files").glob("*.json"),
                          key=lambda p: int(p.stem) if p.stem.isdigit() else -1)
    task_ids = []
    for path in config_files:
        if not path.stem.isdigit():
            continue
        config = read_json(path)
        sites = config["sites"]
        matches = (len(sites) != 1 and "map" not in sites) if args.website == "multi" else sites == [args.website]
        if matches:
            task_ids.append(int(config["task_id"]))
    if args.task_ids:
        wanted = {int(t) for t in args.task_ids.split(",") if t.strip()}
        task_ids = [t for t in task_ids if t in wanted]
    task_ids = [t for t in task_ids[args.start_index:args.end_index] if t > args.prev_id]
    if not task_ids:
        raise PipelineError("No tasks selected")
    run_stream(args, task_ids)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--website", required=True,
                        choices=["shopping", "shopping_admin", "gitlab", "reddit", "multi"])
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--prev_id", type=int, default=-1)
    parser.add_argument("--memory_mode", default="reasoningbank",
                        choices=["no_memory", "reasoningbank", "awm", "synapse"])
    parser.add_argument("--judge", default="autoeval", choices=["autoeval", "gt"])
    parser.add_argument("--success_only", action="store_true")
    parser.add_argument("--memory_suffix", default="")
    parser.add_argument("--skip_done", action="store_true",
                        help="Compatibility flag; completed stages are always reused safely.")
    parser.add_argument("--task_ids", default="")
    parser.add_argument("--task_timeout", type=float, default=float(os.environ.get("RB_TASK_TIMEOUT", 3600)),
                        help="Actor wall-time guard in seconds; max_steps remains the task's interaction budget.")
    parser.add_argument("--step_timeout", type=float, default=float(os.environ.get("RB_STEP_TIMEOUT", 300)))
    args = parser.parse_args()
    known = ["gemini-2.5-flash", "claude-3-7-sonnet@20250219", "gemini-2.5-pro", "google/gemma-3-12b-it"]
    if args.model not in known and not args.model.startswith("qwen"):
        parser.error(f"--model must be one of {known} or qwen...")
    if any(not math.isfinite(t) or t <= 0 for t in (args.task_timeout, args.step_timeout)):
        parser.error("Timeouts must be finite and positive")
    if any(c in args.memory_suffix for c in ("/", "\\")):
        parser.error("--memory_suffix must not contain directory separators")
    return args


if __name__ == "__main__":
    try:
        main(parse_args())
    except PipelineError as exc:
        print(f"[pipeline] STOPPED: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
