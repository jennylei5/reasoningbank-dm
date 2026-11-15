"""Offline regression tests: simulated artifacts, no browser or model calls."""

import json
import os
import signal
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "WebArena"))
import pipeline_memory as pipeline
from pipeline_state import MANIFEST_NAME, PipelineError, read_json, read_records, summary_info, task_status, write_json


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(self.temp.name)
        self.args = SimpleNamespace(
            output_dir=str(Path("results").resolve()), website="shopping",
            memory_mode="reasoningbank", memory_suffix="_test", model="qwen-test",
            judge="autoeval", success_only=False, task_timeout=1, step_timeout=1,
        )
        self.calls = []
        self.fail = None
        self.invalid_judge = False
        self.after_append = False
        self.duplicate = False
        self.mock = patch.object(pipeline, "run_with_timeout", side_effect=self.fake_stage).start()
        self.addCleanup(patch.stopall)

    def directory(self, tid):
        return Path(self.args.output_dir) / f"webarena.{tid}"

    def append(self, path, record):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def fake_stage(self, command, timeout, tid, stage, log):
        self.calls.append((tid, stage))
        failing = self.fail == (tid, stage)
        if failing:
            self.fail = None
        directory = self.directory(tid)
        paths = pipeline.bank_paths(self.args)
        if stage == "run":
            if paths:
                self.append(paths["embeddings"], {"id": str(tid), "embedding": [1.0]})
            if failing:
                return pipeline.StageResult(-9, timed_out=True)
            directory.mkdir(parents=True)
            write_json(directory / "summary_info.json", {
                "cum_reward": float(tid == 2), "n_steps": 1, "err_msg": None,
            })
            for i in (0, 1):
                (directory / f"step_{i}.pkl.gz").write_bytes(f"fake step {i}".encode())
        elif stage == "judge":
            if failing:
                return pipeline.StageResult(1)
            write_json(directory / f"{self.args.model}_autoeval.json", [{
                "idx": tid, "gt": float(tid == 2),
                "rm": None if self.invalid_judge else tid == 2,
                "thoughts": "simulated verdict",
            }])
        else:
            if failing and not self.after_append:
                return pipeline.StageResult(1)
            record = {"task_id": str(tid), "status": "success" if tid == 2 else "fail",
                      "memory_items": ["A simulated upstream advice item."]}
            self.append(paths["memory"], record)
            if self.duplicate:
                self.append(paths["memory"], record)
            if failing:
                return pipeline.StageResult(1)
        return pipeline.StageResult(0)

    def run_tasks(self, ids=(1, 2)):
        pipeline.run_stream(self.args, list(ids))

    def test_complete_stream_and_idempotent_resume(self):
        self.run_tasks()
        self.assertEqual(self.calls, [(t, s) for t in (1, 2) for s in ("run", "judge", "extract")])
        self.run_tasks()
        self.assertEqual(len(self.calls), 6)
        paths = pipeline.bank_paths(self.args)
        self.assertEqual([r["task_id"] for r in read_records(paths["memory"])], ["1", "2"])
        manifest = read_json(Path(self.args.output_dir) / MANIFEST_NAME)
        self.assertEqual(task_status(self.directory(1), self.args.model,
                                    paths["memory"], manifest=manifest,
                                    mode="reasoningbank"), "done")

    def test_judge_failure_resumes_without_running_actor_again(self):
        self.fail = (1, "judge")
        with self.assertRaisesRegex(PipelineError, "Judge 1 failed"):
            self.run_tasks()
        self.assertEqual(self.calls, [(1, "run"), (1, "judge")])
        self.run_tasks()
        self.assertEqual(self.calls.count((1, "run")), 1)
        self.assertEqual(self.calls.count((1, "judge")), 2)

    def test_null_verdict_is_failure_even_with_zero_exit(self):
        self.invalid_judge = True
        with self.assertRaisesRegex(PipelineError, "Judge 1 failed"):
            self.run_tasks()
        self.assertNotIn((1, "extract"), self.calls)
        self.assertNotIn((2, "run"), self.calls)
        self.invalid_judge = False
        self.run_tasks()
        self.assertEqual(self.calls.count((1, "run")), 1)

    def test_extraction_failure_resumes_only_missing_stage(self):
        self.fail = (1, "extract")
        with self.assertRaisesRegex(PipelineError, "Extraction 1 failed"):
            self.run_tasks()
        self.assertNotIn((2, "run"), self.calls)
        manifest = read_json(Path(self.args.output_dir) / MANIFEST_NAME)
        self.assertEqual(task_status(self.directory(1), self.args.model,
                                    pipeline.bank_paths(self.args)["memory"],
                                    mode="reasoningbank", manifest=manifest), "not_extracted")
        self.run_tasks()
        self.assertEqual(self.calls.count((1, "run")), 1)
        self.assertEqual(self.calls.count((1, "judge")), 1)
        self.assertEqual(self.calls.count((1, "extract")), 2)

    def test_extraction_written_before_failure_is_not_appended_twice(self):
        self.fail, self.after_append = (1, "extract"), True
        with self.assertRaises(PipelineError):
            self.run_tasks()
        self.run_tasks()
        self.assertEqual(self.calls.count((1, "extract")), 1)
        self.assertEqual(len(read_records(pipeline.bank_paths(self.args)["memory"])), 2)

    def test_actor_timeout_stops_and_cannot_silently_retry(self):
        self.fail = (1, "run")
        with self.assertRaisesRegex(PipelineError, "Actor 1 failed"):
            self.run_tasks()
        self.assertEqual(self.calls, [(1, "run")])
        with self.assertRaisesRegex(PipelineError, "Browser state cannot be restored"):
            self.run_tasks()
        self.assertEqual(self.calls, [(1, "run")])
        self.assertTrue((Path(self.args.output_dir) / "pipeline_checkpoints/1/files.json").exists())

    def test_reward_zero_is_a_valid_failed_experience(self):
        self.run_tasks((1,))
        self.run_tasks((1,))
        self.assertEqual(self.calls.count((1, "run")), 1)
        self.assertEqual(read_records(pipeline.bank_paths(self.args)["memory"])[0]["status"], "fail")

    def mark_finished_model_parse_failure(self, tid=1):
        path = self.directory(tid) / 'summary_info.json'
        data = read_json(path)
        message = 'Could not parse a valid value after 6 retries.'
        data.update(err_msg=message, truncated=True, terminated=False,
                    stack_trace='File "agents/legacy/agent.py", line 1, in get_action\nValueError: ' + message)
        write_json(path, data)
        return data

    def test_finished_model_parse_failure_is_committed_without_actor_retry(self):
        def stage(*args):
            result = self.fake_stage(*args)
            if args[2:4] == (1, 'run'):
                self.mark_finished_model_parse_failure()
            return result
        self.mock.side_effect = stage
        self.run_tasks()
        self.run_tasks()
        self.assertEqual(self.calls.count((1, 'run')), 1)
        self.assertIn((1, 'extract'), self.calls)
        self.assertIn((2, 'run'), self.calls)
        self.assertEqual(read_records(pipeline.bank_paths(self.args)['memory'])[0]['status'], 'fail')

    def test_model_failure_exception_does_not_accept_other_errors_or_success(self):
        self.fake_stage([], 1, 1, 'run', Path('unused'))
        original = self.mark_finished_model_parse_failure()
        self.assertIsNotNone(summary_info(self.directory(1)))
        for changed in [{'truncated': False}, {'cum_reward': 1.0},
                        {'err_msg': 'API error'}, {'stack_trace': 'unrelated error'}]:
            with self.subTest(changed=changed):
                write_json(self.directory(1) / 'summary_info.json', original | changed)
                self.assertIsNone(summary_info(self.directory(1)))

    def test_finished_model_parse_failure_still_requires_complete_steps(self):
        self.fake_stage([], 1, 1, 'run', Path('unused'))
        self.mark_finished_model_parse_failure()
        (self.directory(1) / 'step_1.pkl.gz').unlink()
        self.assertIsNone(summary_info(self.directory(1)))

    def test_no_memory_does_not_require_extraction(self):
        self.args.memory_mode = "no_memory"
        self.run_tasks()
        self.run_tasks()
        self.assertEqual(self.calls, [(1, "run"), (1, "judge"), (2, "run"), (2, "judge")])

    def test_success_only_records_an_intentional_skip(self):
        self.args.success_only = True
        self.run_tasks()
        self.run_tasks()
        self.assertNotIn((1, "extract"), self.calls)
        self.assertEqual(self.calls.count((2, "extract")), 1)

    def test_refuses_legacy_output(self):
        self.directory(1).mkdir(parents=True)
        with self.assertRaisesRegex(PipelineError, "Legacy results"):
            self.run_tasks()
        self.assertEqual(self.calls, [])

    def test_refuses_preexisting_bank(self):
        self.append(pipeline.bank_paths(self.args)["memory"], {"task_id": "99"})
        with self.assertRaisesRegex(PipelineError, "empty, separate"):
            self.run_tasks()
        self.assertEqual(self.calls, [])

    def test_refuses_different_output_using_owned_bank(self):
        self.run_tasks()
        self.args.output_dir += "_another"
        with self.assertRaisesRegex(PipelineError, "belongs to another"):
            self.run_tasks()
        self.assertEqual(len(self.calls), 6)

    def test_refuses_changed_model_or_stream(self):
        self.run_tasks()
        with self.assertRaisesRegex(PipelineError, "configuration changed"):
            self.run_tasks((2,))
        self.args.model = "qwen-another"
        with self.assertRaisesRegex(PipelineError, "configuration changed"):
            self.run_tasks()

    def test_detects_changed_completed_trajectory(self):
        self.run_tasks()
        (self.directory(1) / "step_0.pkl.gz").write_bytes(b"changed")
        with self.assertRaisesRegex(PipelineError, "artifacts were changed"):
            self.run_tasks()
        self.assertEqual(len(self.calls), 6)

    def test_duplicate_extraction_blocks_following_task(self):
        self.duplicate = True
        with self.assertRaisesRegex(PipelineError, "Duplicate memory"):
            self.run_tasks()
        self.assertNotIn((2, "run"), self.calls)

    def test_checker_does_not_certify_legacy_rb_as_a_new_clean_run(self):
        self.run_tasks((1,))
        path = pipeline.bank_paths(self.args)["memory"]
        self.assertEqual(task_status(self.directory(1), self.args.model,
                                    memory_path=path, mode="reasoningbank"), "legacy")

    def test_resume_adopts_actor_finished_before_parent_recorded_it(self):
        def interrupt_after_actor(*args):
            self.fake_stage(*args)
            raise KeyboardInterrupt

        self.mock.side_effect = interrupt_after_actor
        with self.assertRaises(KeyboardInterrupt):
            self.run_tasks()
        self.mock.side_effect = self.fake_stage
        self.run_tasks()
        self.assertEqual(self.calls.count((1, "run")), 1)

    def test_resume_before_actor_launch_reuses_unchanged_checkpoint(self):
        output = Path(self.args.output_dir)
        pipeline.checkpoint_bank(pipeline.bank_paths(self.args), output, 1)
        self.run_tasks()
        self.assertEqual(len(self.calls), 6)

    def test_partial_memory_write_is_preserved_and_not_retried(self):
        self.fail, self.after_append = (1, "extract"), True
        with self.assertRaises(PipelineError):
            self.run_tasks()
        path = pipeline.bank_paths(self.args)["memory"]
        with path.open("a") as f:
            f.write('{"task_id":')
        before = path.read_bytes()
        with self.assertRaisesRegex(PipelineError, "Invalid JSONL"):
            self.run_tasks()
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(len(self.calls), 3)

    def test_resume_rejects_actor_modification_after_judge_failure(self):
        self.fail = (1, "judge")
        with self.assertRaises(PipelineError):
            self.run_tasks()
        (self.directory(1) / "step_1.pkl.gz").write_bytes(b"changed")
        with self.assertRaisesRegex(PipelineError, "actor trajectory was changed"):
            self.run_tasks()
        self.assertEqual(len(self.calls), 2)

    def test_output_lock_prevents_a_second_runner(self):
        with pipeline.exclusive_lock(Path(self.args.output_dir) / ".pipeline.lock"):
            with self.assertRaisesRegex(PipelineError, "Another pipeline"):
                self.run_tasks()
        self.assertEqual(self.calls, [])

    def test_real_subprocess_timeout_returns_failure_and_records_it(self):
        patch.stopall()
        result = pipeline.run_with_timeout(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            0.05, 1, "run", Path("timeout.jsonl"),
        )
        self.assertFalse(result.ok)
        self.assertTrue(result.timed_out)
        self.assertEqual(read_records("timeout.jsonl")[0]["result"], "timeout")

    def test_subprocess_launch_failure_is_reported(self):
        patch.stopall()
        result = pipeline.run_with_timeout(
            [str(Path("missing-executable").resolve())], 1, 1, "run", Path("error.jsonl"),
        )
        self.assertFalse(result.ok)
        self.assertIn("No such file", result.error)
        self.assertEqual(read_records("error.jsonl")[0]["error"], result.error)

    @unittest.skipUnless(hasattr(signal, "SIGUSR1"), "Unix actor diagnostics")
    def test_actor_timeout_requests_diagnostics_before_killing(self):
        patch.stopall()
        code = (
            "import signal,time; from pathlib import Path; "
            "signal.signal(signal.SIGUSR1, lambda *_: Path('stack_requested').write_text('yes')); "
            "time.sleep(5)"
        )
        result = pipeline.run_with_timeout(
            [sys.executable, "-c", code], 0.3, 1, "run", Path("timeout.jsonl"),
        )
        self.assertTrue(result.timed_out)
        self.assertEqual(Path('stack_requested').read_text(), 'yes')
        self.assertEqual(result.returncode, -signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
