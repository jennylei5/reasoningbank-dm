"""Avoid computing hidden prompt diffs; preserve visible prompt contents."""


def install_lazy_prompt_diffs():
    from agents.legacy import dynamic_prompting as prompts

    original = prompts.Diff
    if getattr(original, "_rb_lazy_diff", False):
        return

    class LazyDiff(original):
        _rb_lazy_diff = True

        def __init__(self, previous, new, prefix="", max_line_diff=20,
                     shrink_speed=2, visible=True):
            prompts.Shrinkable.__init__(self, visible=visible)
            self.max_line_diff = max_line_diff
            self.shrink_speed = shrink_speed
            self.prefix = prefix
            self.header, self.diff_lines = "", []
            self._previous, self._new = previous, new
            self._computed = False
            if self.is_visible:
                self._ensure_computed()

        def _ensure_computed(self):
            if not self._computed:
                self.header, self.diff_lines = prompts.diff(self._previous, self._new)
                self._computed = True
                self._previous = self._new = None

        @property
        def _prompt(self):
            if not self.is_visible:
                return ""
            self._ensure_computed()
            return original._prompt.fget(self)

    prompts.Diff = LazyDiff
