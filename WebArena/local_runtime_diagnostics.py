"""Local diagnostics for actors waiting inside Playwright's greenlet loop."""

import faulthandler
import signal
import sys
import traceback


def install_actor_stack_dump():
    if not hasattr(signal, "SIGUSR1"):
        return
    from greenlet import getcurrent

    actor = getcurrent()

    def dump_stack(signum, current_frame):
        # A native thread dump alone shows only Playwright's asyncio dispatcher
        # when the actor greenlet is suspended waiting for a browser operation.
        print("[local-diagnostics] Actor call stack (including suspended greenlet):",
              file=sys.stderr, flush=True)
        traceback.print_stack(actor.gr_frame or current_frame, file=sys.stderr)
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        sys.stderr.flush()

    signal.signal(signal.SIGUSR1, dump_stack)
