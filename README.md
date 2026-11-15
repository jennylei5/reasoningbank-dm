# ReasoningBank with Qwen

**A partial reproduction and adaptive memory management proposal**

Can a web agent's memory remain useful as experience accumulates? This project reproduces the unscaled [ReasoningBank](https://arxiv.org/abs/2509.25140) baseline with Qwen, analyzes its behavior, and proposes improvements to append-only memory management.

## What is implemented

The baseline learns reusable advice from both self-judged successes and failures:

**Retrieve historical experience → execute a task → self-judge → extract advice → append to memory**

- **Benchmark:** 475 WebArena tasks across Shopping (187), Shopping Admin (182), and Reddit (106), using locally hosted website images and the repository's WebArena evaluation harness.
- **Models:** `qwen3.7-flash` for the actor, judge, extractor, and LLM-based fuzzy grading; `qwen3.7-text-embedding` for retrieval.
- **Settings:** top-1 retrieval, 30 steps per task, separate domain memory banks, and ascending task order. Native thinking is disabled; explicit ReAct reasoning remains enabled.
- **Adaptations:** Qwen integration, stage-aware recovery, memory validation, and browser compatibility fixes. See [code changes](CODE_CHANGES_VS_UPSTREAM.md).

MaTTS and adaptive consolidation are not implemented in this reproduction.

## Running the code

Use Python 3.12, Docker-hosted WebArena websites, and Playwright/BrowserGym.

```bash
uv sync --python 3.12
uv run playwright install chromium
```

Provision websites using the [WebArena setup guide](https://github.com/gasse/webarena-setup/tree/main/webarena), prepare `WebArena/config_files/` and authenticated browser state, and put `DASHSCOPE_API_KEY` in `~/.reasoningbank_env`. The launcher expects Shopping at `localhost:7770`, Admin at `localhost:7780/admin`, and Reddit at `localhost:9999`. Benchmark instances, task configs, and credentials are not bundled.

From the repository root, run a domain with a fresh run ID:

```bash
RB_RUN_ID=demo bash WebArena/run.sh shopping reasoningbank
python3 scripts/check_results.py reasoningbank_demo --sites shopping
```

Replace `shopping` with `shopping_admin` or `reddit`, or use `no_memory` as the method. Use separately initialized websites and fresh banks for comparisons. Inspect interrupted actors before restarting; completed actors can resume missing judge/extraction stages.

| Path | Purpose |
|---|---|
| `WebArena/pipeline_memory.py` | Task execution and stage-aware recovery |
| `WebArena/memory_management.py`, `induce_memory.py` | Retrieval and memory extraction |
| `scripts/analyze.py` | SR, step statistics, judge agreement, and cumulative SR plots |
| `tests/` | Pipeline and compatibility regression tests |

Credentials, private tracking notes, raw trajectories, memory banks, and logs are excluded from version control.

## Proposed next steps

Investigate **ADD / MERGE / UPDATE / ARCHIVE** decisions to reduce redundant, conflicting, or outdated advice. As a separate extension, explore memory-item retrieval and applicability-aware reranking. These ideas remain proposals; the current implementation uses append-only consolidation.

## Attribution

This project builds on [Google Research's ReasoningBank implementation](https://github.com/google-research/reasoning-bank), based on commit `ed80611788292ea739f1effd31f16c53823b8a0d`. The original method is by Ouyang et al., *ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory*, ICLR 2026.

Upstream code also builds on [WebArena](https://github.com/web-arena-x/webarena), [Agent Workflow Memory](https://github.com/zorazrw/agent-workflow-memory), and [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent). Original copyright notices and the [Apache 2.0 license](LICENSE) are retained. This repository is an independent reproduction of ReasoningBank.
