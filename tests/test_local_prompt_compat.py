import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "WebArena"))
from local_prompt_compat import install_lazy_prompt_diffs


@unittest.skipUnless(importlib.util.find_spec("browsergym") and importlib.util.find_spec("langchain_core"),
                     "Run with project .venv for prompt dependencies")
class LazyDiffTests(unittest.TestCase):
    def setUp(self):
        from agents.legacy import dynamic_prompting
        self.prompts = dynamic_prompting
        original = dynamic_prompting.Diff
        self.addCleanup(setattr, dynamic_prompting, "Diff", original)

    def test_disabled_diffs_never_run_the_expensive_comparison(self):
        install_lazy_prompt_diffs()
        with patch.object(self.prompts, "diff", side_effect=AssertionError("Hidden diff computed")):
            for visible in [False, lambda: False]:
                diff = self.prompts.Diff("previous", "current", visible=visible)
                diff.shrink()
                self.assertEqual(diff.prompt, "")

    def test_visible_and_later_enabled_diffs_preserve_output(self):
        original = self.prompts.Diff
        expected = original("a\nb", "a\nc", prefix="diff:")
        expected.shrink()
        install_lazy_prompt_diffs()
        installed = self.prompts.Diff
        install_lazy_prompt_diffs()
        self.assertIs(self.prompts.Diff, installed)
        visible = [False]
        actual = installed("a\nb", "a\nc", prefix="diff:", visible=lambda: visible[0])
        actual.shrink()
        visible[0] = True
        self.assertEqual(actual.prompt, expected.prompt)
