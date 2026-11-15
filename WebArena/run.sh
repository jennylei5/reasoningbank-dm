#!/bin/bash
# Run one ReasoningBank experiment on one WebArena website, with Qwen (Alibaba Cloud Bailian).
#
# Usage:   bash run.sh <website> <memory_mode> [extra options for pipeline_memory.py]
#   website:      shopping | shopping_admin
#   memory_mode:  no_memory | synapse | awm | reasoningbank
# Examples:
#   bash run.sh shopping no_memory --start_index 0 --end_index 3     # smoke test, 3 tasks
#   bash run.sh shopping reasoningbank                                # all 187 Shopping tasks
#   bash run.sh shopping reasoningbank --success_only                 # Figure 7 ablation
#   RB_RUN_ID=fixed_v1 bash run.sh shopping reasoningbank             # new results AND empty bank
# Resume with the same command. Judge/extraction failures resume in place;
# an interrupted actor requires a new run after restoring the website state.
#
# Results go to ~/reasoning-bank/results/<memory_mode>[_successonly]_<website>/
set -e
cd "$(dirname "$0")"                       # pipeline_memory.py expects to run inside WebArena/
REPO_ROOT="$(cd .. && pwd)"

# ---- API key: kept outside the repo, in ~/.reasoningbank_env (defines DASHSCOPE_API_KEY) ----
source "$HOME/.reasoningbank_env"

# ---- Qwen on Bailian, through its OpenAI-compatible endpoint ----
export OPENAI_API_KEY="$DASHSCOPE_API_KEY"
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL="${RB_MODEL:-qwen3.7-flash}"             # agent, judge and memory extraction
export RB_EMBED_MODEL="qwen3.7-text-embedding"      # retrieval embeddings
export RB_GRADER_MODEL="$MODEL"                # official WebArena grader (fuzzy answer matching)
# Leave enough wall time for 30 Qwen steps; this is an infrastructure guard.
export RB_TASK_TIMEOUT="${RB_TASK_TIMEOUT:-3600}"

# ---- WebArena website addresses (only Shopping and Admin are deployed; the rest are placeholders) ----
export WA_SHOPPING="http://localhost:7770"
export WA_SHOPPING_ADMIN="http://localhost:7780/admin"
export WA_REDDIT="http://localhost:9999"
export WA_GITLAB="http://localhost:8023"
export WA_WIKIPEDIA="http://localhost:8888"
export WA_MAP="http://localhost:3000"
export WA_HOMEPAGE="http://localhost:4399"

# ---- the patched WebArena grader in third_party/, and the project's Python environment ----
export PYTHONPATH="$REPO_ROOT/third_party:$PYTHONPATH"
source "$REPO_ROOT/.venv/bin/activate"

WEBSITE="$1"; MODE="$2"; shift 2
SUFFIX=""
for a in "$@"; do [ "$a" = "--success_only" ] && SUFFIX="_successonly"; done
RUN_ID="${RB_RUN_ID:-}"
if [[ -n "$RUN_ID" && ! "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "RB_RUN_ID must contain only letters, digits, underscores, or hyphens." >&2
  exit 2
fi
if [ -n "$RUN_ID" ]; then SUFFIX="${SUFFIX}_${RUN_ID}"; fi

python pipeline_memory.py \
  --website "$WEBSITE" \
  --memory_mode "$MODE" \
  --memory_suffix "$SUFFIX" \
  --model "$MODEL" \
  --output_dir "$REPO_ROOT/results/${MODE}${SUFFIX}_${WEBSITE}" \
  "$@"
