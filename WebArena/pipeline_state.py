"""Local pipeline bookkeeping; no browser, model, or memory-policy changes."""

import hashlib
import json
import math
import os
import re
from pathlib import Path


MANIFEST_NAME = "pipeline_state.json"


class PipelineError(RuntimeError):
    pass


def read_json(path):
    with Path(path).open() as f:
        return json.load(f)


def write_json(path, value):
    """Replace one bookkeeping file atomically."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_fingerprint(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def is_model_parse_failure(summary):
    """An exhausted model-output parser is a finished failed attempt, not a crash.

    GenericAgent catches exactly this ValueError and returns action=None. The
    experiment loop then saves a normal truncated episode with its official
    reward. Keep infrastructure failures (including API errors) fail-closed.
    """
    message = summary.get("err_msg")
    stack = summary.get("stack_trace") or ""
    return (
        isinstance(message, str)
        and re.fullmatch(r"Could not parse a valid value after [1-9][0-9]* retries\.", message) is not None
        and summary.get("truncated") is True
        and summary.get("terminated") is False
        and summary.get("cum_reward") == 0
        and "in get_action" in stack
        and f"ValueError: {message}" in stack
    )


def summary_info(directory):
    try:
        summary = read_json(Path(directory) / "summary_info.json")
        reward = summary["cum_reward"]
        steps = summary["n_steps"]
        if summary.get("err_msg") and not is_model_parse_failure(summary):
            return None
        if type(reward) not in (float, int) or not math.isfinite(reward):
            return None
        if type(steps) is not int or steps < 0:
            return None
        # The terminal observation is a separate step file, even with no action.
        if not all((Path(directory) / f"step_{i}.pkl.gz").is_file()
                   for i in range(steps + 1)):
            return None
        return summary
    except (OSError, ValueError, KeyError, TypeError):
        return None


def trajectory_fingerprint(directory):
    directory = Path(directory)
    summary = summary_info(directory)
    if summary is None:
        raise PipelineError(f"Incomplete or failed actor result: {directory}")
    digest = hashlib.sha256()
    paths = [directory / "summary_info.json"] + [
        directory / f"step_{i}.pkl.gz" for i in range(summary["n_steps"] + 1)
    ]
    for path in paths:
        digest.update(path.name.encode())
        digest.update(fingerprint(path).encode())
    return digest.hexdigest()


def judge_info(directory, model):
    try:
        records = read_json(Path(directory) / f"{model}_autoeval.json")
        if not isinstance(records, list) or len(records) != 1:
            return None
        record = records[0]
        if type(record.get("rm")) is not bool:
            return None
        summary = summary_info(directory)
        if summary is None or record.get("gt") != summary["cum_reward"]:
            return None
        return record
    except (OSError, ValueError, AttributeError, TypeError):
        return None


def read_records(path):
    path = Path(path)
    if not path.exists():
        return []
    try:
        with path.open() as f:
            records = [json.loads(line) for line in f if line.strip()]
        if not all(isinstance(record, dict) for record in records):
            raise ValueError("expected JSON objects")
        return records
    except (OSError, ValueError) as exc:
        raise PipelineError(f"Invalid JSONL; preserve the file for inspection: {path}") from exc


def needs_extraction(mode, success_only, reward):
    return mode != "no_memory" and not (success_only and reward != 1)


def memory_record(memory_path, tid, reward):
    records = [r for r in read_records(memory_path) if str(r.get("task_id")) == str(tid)]
    if len(records) > 1:
        raise PipelineError(f"Duplicate memory for task {tid}: {memory_path}")
    if not records:
        return None
    record = records[0]
    if record.get("status") != ("success" if reward == 1 else "fail"):
        raise PipelineError(f"Memory label does not match task {tid}'s selected verdict")
    # Only check that extraction produced an artifact. Its advice/schema policy
    # remains the upstream extractor's responsibility.
    items = record.get("memory_items")
    if not isinstance(items, list) or not any(isinstance(x, str) and x.strip() for x in items):
        raise PipelineError(f"Empty memory output for task {tid}: {memory_path}")
    return record


def task_status(directory, model, memory_path=None, success_only=False,
                mode="no_memory", criteria="autoeval", manifest=None):
    """Read-only status shared by the runner and the local result checker."""
    directory = Path(directory)
    summary_path = directory / "summary_info.json"
    if not summary_path.exists():
        return "missing"
    summary = summary_info(directory)
    if summary is None:
        return "agent_error"
    judged = judge_info(directory, model)
    if judged is None:
        return "not_judged"
    reward = judged["rm"] if criteria == "autoeval" else summary["cum_reward"]
    tid = directory.name.split(".")[-1]
    try:
        memory = None
        if needs_extraction(mode, success_only, reward):
            if memory_path is None:
                return "not_extracted"
            memory = memory_record(memory_path, tid, reward)
            if memory is None:
                return "not_extracted"
        if manifest is None:
            # Legacy baseline results are still usable. Legacy RB artifacts
            # cannot establish that memory was committed before the next task.
            return "done" if mode == "no_memory" else "legacy"
        state = manifest.get("tasks", {}).get(str(tid), {})
        if state.get("trajectory") != trajectory_fingerprint(directory):
            return "inconsistent"
        judge_path = directory / f"{model}_autoeval.json"
        if state.get("judge") != fingerprint(judge_path):
            return "inconsistent"
        if memory is not None and state.get("memory") != record_fingerprint(memory):
            return "not_extracted" if not state.get("complete") else "inconsistent"
        return "done" if state.get("complete") else "not_committed"
    except (PipelineError, OSError, ValueError):
        return "inconsistent"
