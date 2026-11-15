# Local Changes to ReasoningBank

Based on [google-research/reasoning-bank](https://github.com/google-research/reasoning-bank), commit `ed80611788292ea739f1effd31f16c53823b8a0d`.

This is a Qwen-based partial reproduction on WebArena Shopping, Shopping Admin, and Reddit. The original license and attribution are retained.

## Model and environment adaptation

- Use Python 3.12 instead of 3.13; add Matplotlib for result analysis.
- Add Alibaba Cloud OpenAI-compatible clients for the actor, self-judge, memory extractor, and embedding API.
- Use Qwen for the benchmark's LLM-based fuzzy/UA grading calls as well.
- Disable native Qwen thinking, retain explicit ReAct reasoning, cap actor output at 8,192 tokens, and use an approximate tokenizer.
- Configure local website URLs and restrict browser hostname resolution to avoid unavailable external resources.

## Runtime reliability

- Extend the existing `pipeline_memory.py` with stage-aware recovery, failure handling, locks, per-task checkpoints, and artifact validation.
- Isolate results and memory banks with run IDs; verify memory/cache IDs, order, and committed hashes.
- Add `pipeline_state.py` and three local compatibility/diagnostic modules: bounded navigation-content reads, disabled unused screenshots for text-only agents, lazy computation of hidden prompt diffs, and actor stack dumps.
- Use a 3,600-second actor guard and a 300-second judge/extraction guard. Infrastructure recovery attempts are archived separately from retained evaluation trajectories.

## Analysis and tests

- Add `scripts/analyze.py` for SR, average steps, outcome-conditioned steps, judge agreement, and cumulative SR plots.
- Add `scripts/check_results.py` and regression tests for pipeline recovery and compatibility wrappers.
- Add an optional success-only extraction switch; it is disabled in the standard ReasoningBank run.

## Method boundaries

The main ReAct prompts, query-to-query top-1 retrieval, append-only consolidation, and learning from both judged successes and failures are retained. MaTTS and adaptive consolidation are not implemented. Native thinking, embeddings, fuzzy grading, network behavior, and recovery handling differ from the upstream setup; this is not an exact reproduction of the paper's configuration.

## Excluded local artifacts

Private tracking notes, credentials, virtual environments, experiment trajectories, memory banks, logs, caches, and report drafts are excluded from version control. Upstream README and contribution documents remain included.
