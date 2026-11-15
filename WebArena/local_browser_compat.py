"""Local browser timing workaround; benchmark evaluation rules stay unchanged."""

import logging


NAVIGATION_CONTENT_ERROR = (
    "Unable to retrieve content because the page is navigating and changing the content"
)


def read_content_with_retry(page, read_content, *, error_type, timeout_error_type,
                            retries=3, wait_timeout_ms=3000, settle_ms=250):
    """Retry only a failed DOM read, never an action or an entire evaluator.

    Page.content() can race a navigation started by the preceding search click.
    Leave successful reads and all unrelated errors exactly as they were.
    """
    for attempt in range(retries + 1):
        try:
            return read_content(page)
        except error_type as exc:
            if NAVIGATION_CONTENT_ERROR not in str(exc) or attempt == retries:
                raise
            logging.getLogger(__name__).warning(
                "[local-browser] navigation raced Page.content; retrying DOM read %s/%s",
                attempt + 1, retries,
            )
            try:
                page.wait_for_load_state("domcontentloaded", timeout=wait_timeout_ms)
            except timeout_error_type:
                # A slow load may still provide readable HTML. The bounded next
                # content() attempt decides; never convert an error into a score.
                pass
            # A load state may refer to the outgoing document. Pump events before
            # trying again so a just-started navigation can advance as well.
            page.wait_for_timeout(settle_ms)


def install_navigation_content_retry():
    """Install once in a Shopping actor process, without editing Playwright."""
    from playwright.sync_api import Error, Page, TimeoutError

    original = Page.content
    if getattr(original, "_rb_navigation_retry", False):
        return

    def content(page):
        return read_content_with_retry(
            page, original, error_type=Error, timeout_error_type=TimeoutError,
        )

    content._rb_navigation_retry = True
    Page.content = content


def disable_unused_screenshots():
    """Skip diagnostic PNG capture when the actor does not consume images.

    BrowserGym captures a screenshot on every observation even for text-only
    agents. CDP Page.captureScreenshot can hang independently of DOM extraction.
    Return None explicitly; do not fabricate images or change textual fields.
    """
    import browsergym.core.env as browser_env

    if getattr(browser_env.extract_screenshot, "_rb_disabled_for_text", False):
        return

    def no_screenshot(page):
        return None

    no_screenshot._rb_disabled_for_text = True
    browser_env.extract_screenshot = no_screenshot
    logging.getLogger(__name__).info(
        "[local-browser] Screenshot capture disabled for text-only actor; no PNG images will be saved."
    )

