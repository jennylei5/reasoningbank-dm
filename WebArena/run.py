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
import shutil
import argparse
import json
from pathlib import Path
import sys
import multiprocessing

# Local timeout diagnostics: the pipeline can request the current Python stack
# before killing a stuck actor. This does not interrupt an action or retry it.
from local_runtime_diagnostics import install_actor_stack_dump
install_actor_stack_dump()

try:
    from utils import gemini_retry  # noqa: F401 — patches google.genai with backoff
except ImportError:
    pass

for _k in ("SHOPPING", "SHOPPING_ADMIN", "REDDIT", "GITLAB", "WIKIPEDIA", "MAP", "HOMEPAGE"):
    if f"WA_{_k}" in os.environ:
        os.environ[_k] = os.environ[f"WA_{_k}"]

# Only allow the browser to reach locally hosted websites. External resources
# embedded in WebArena pages (e.g. video iframes) can otherwise hang forever,
# because BrowserGym evaluates scripts inside every iframe without a timeout.
# Unresolvable hosts fail immediately instead. IP literals such as 127.0.0.1
# bypass the host resolver and keep working.
from playwright.sync_api import BrowserType as _BrowserType

_original_launch = _BrowserType.launch


def _launch_local_only(self, *args, **kwargs):
    launch_args = list(kwargs.get("args") or [])
    launch_args.append("--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE localhost")
    kwargs["args"] = launch_args
    return _original_launch(self, *args, **kwargs)


_BrowserType.launch = _launch_local_only

from memory_management import select_memory
import webarena_patch  # noqa: F401 — applies MemEvol llm_fuzzy_match prompt

from browsergym.experiments import ExpArgs, EnvArgs

from agents.legacy.agent import GenericAgentArgs
from agents.legacy.dynamic_prompting import Flags
from agents.legacy.utils.chat_api import ChatModelArgs


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def ensure_file(path: str):
    """Ensure that the file and its parent directories exist."""
    if path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        open(path, "w").close()


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment with hyperparameters.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-2.5-flash",
        help="Model name for the chat model.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="openended",
        help="Name of the Browsergym task to run. If 'openended', you need to specify a 'start_url'",
    )
    parser.add_argument(
        "--start_url",
        type=str,
        default="https://www.google.com",
        help="Starting URL (only for the openended task).",
    )
    parser.add_argument(
        "--slow_mo", type=int, default=30, help="Slow motion delay for the playwright actions."
    )
    parser.add_argument(
        "--headless",
        type=str2bool,
        default=True,
        help="Run the experiment in headless mode (hides the browser windows).",
    )
    parser.add_argument(
        "--demo_mode",
        type=str2bool,
        default=True,
        help="Add visual effects when the agents performs actions.",
    )
    parser.add_argument(
        "--use_html", type=str2bool, default=False, help="Use HTML in the agent's observation space."
    )
    parser.add_argument(
        "--use_ax_tree",
        type=str2bool,
        default=True,
        help="Use AX tree in the agent's observation space.",
    )
    parser.add_argument(
        "--use_screenshot",
        type=str2bool,
        default=False,
        help="Use screenshot in the agent's observation space.",
    )
    parser.add_argument(
        "--multi_actions", type=str2bool, default=True, help="Allow multi-actions in the agent."
    )
    parser.add_argument(
        "--action_space",
        type=str,
        default="bid",
        choices=["python", "bid", "coord", "bid+coord", "bid+nav", "coord+nav", "bid+coord+nav"],
        help="",
    )
    parser.add_argument(
        "--use_history",
        type=str2bool,
        default=True,
        help="Use history in the agent's observation space.",
    )
    parser.add_argument(
        "--use_thinking",
        type=str2bool,
        default=True,
        help="Use thinking in the agent (chain-of-thought prompting).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=30,
        help="Maximum number of steps to take for each task.",
    )
    parser.add_argument(
        "--memory_path",
        type=str,
        default=None,
        help="Path to the memory file to load for the agent.",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="./results",
        help="Path to the directory where the results will be saved.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    from local_prompt_compat import install_lazy_prompt_diffs
    install_lazy_prompt_diffs()

    if not args.use_screenshot:
        from local_browser_compat import disable_unused_screenshots
        disable_unused_screenshots()

    # Local timing compatibility only: retry a transient HTML read during a
    # Shopping navigation. No evaluator rules, model calls, or actions change.
    if args.task_name.startswith("webarena."):
        task_id = args.task_name.split(".")[-1]
        with open(f"config_files/{task_id}.json") as f:
            local_task_config = json.load(f)
        if local_task_config.get("sites") == ["shopping"]:
            from local_browser_compat import install_navigation_content_retry
            install_navigation_content_retry()

    if args.memory_path:
        ensure_file(args.memory_path)

        # Agent with memory retrieval

        website = args.memory_path.split("/")[-1].split(".")[0]
        mem_partial_path = args.memory_path.split("/")[0]

        if not os.path.exists(f"./{mem_partial_path}/{website}.jsonl"):
            open(f"./{mem_partial_path}/{website}.jsonl", "w").close()

        with open(f"./{mem_partial_path}/{website}.jsonl", "r") as f:
            reasoning_bank = [json.loads(line) for line in f.readlines()]

        cur_query = json.load(open(f"./config_files/{args.task_name.split('.')[-1]}.json"))["intent"]

        res = select_memory(n=1,
                            reasoning_bank=reasoning_bank,
                            cur_query=cur_query,
                            task_id=args.task_name.split('.')[-1],
                            cache_path=f"./{mem_partial_path}/{website}_embeddings.jsonl",
                            prefer_model="gemini")

        if not res:
            with open(args.memory_path, "w") as f:
                f.write("")
        else:
            mem_items = []
            for item in res:
                for i in item["memory_items"]:
                    mem_items.append(i)
            with open(args.memory_path, "w") as f:
                f.write("\n\n".join(mem_items) + "\n")

    env_args = EnvArgs(
        task_name=args.task_name,
        task_seed=None,
        max_steps=args.max_steps,
        headless=args.headless,
        viewport={"width": 1500, "height": 1280},
        slow_mo=args.slow_mo,
    )

    if args.task_name == "openended":
        env_args.wait_for_user_message = True
        env_args.task_kwargs = {"start_url": args.start_url}

    exp_args = ExpArgs(
        env_args=env_args,
        save_screenshot=args.use_screenshot,
        agent_args=GenericAgentArgs(
            chat_model_args=ChatModelArgs(
                model_name=args.model_name,
                temperature=0.7,
                max_total_tokens=128_000,  # "Maximum total tokens for the chat model."
                max_input_tokens=126_000,  # "Maximum tokens for the input to the chat model."
                max_new_tokens=65_536,  # "Maximum total tokens for the chat model."
            ),
            flags=Flags(
                use_html=args.use_html,
                use_ax_tree=args.use_ax_tree,
                use_thinking=args.use_thinking,  # "Enable the agent with a memory (scratchpad)."
                use_error_logs=True,  # "Prompt the agent with the error logs."
                use_memory=False,  # "Enables the agent with a memory (scratchpad)."
                use_history=args.use_history,
                use_diff=False,  # "Prompt the agent with the difference between the current and past observation."
                use_past_error_logs=True,  # "Prompt the agent with the past error logs."
                use_action_history=True,  # "Prompt the agent with the action history."
                multi_actions=args.multi_actions,
                use_abstract_example=True,  # "Prompt the agent with an abstract example."
                use_concrete_example=True,  # "Prompt the agent with a concrete example."
                use_screenshot=args.use_screenshot,
                enable_chat=True,
                demo_mode="default" if args.demo_mode else "off",
                memory_path=args.memory_path,
            ),
        ),
    )

    exp_args.prepare(Path(args.results_path))
    exp_args.run()

    dest = f"{args.results_path}/{args.task_name}"
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.rename(exp_args.exp_dir, dest)


if __name__ == "__main__":
    main()
