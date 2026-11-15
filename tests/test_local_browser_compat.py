"""Offline tests for the narrow navigation/read race workaround."""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "WebArena"))
from local_browser_compat import NAVIGATION_CONTENT_ERROR, read_content_with_retry


class BrowserError(Exception):
    pass


class LoadTimeout(BrowserError):
    pass


class NavigationReadTests(unittest.TestCase):
    def read(self, page, reader):
        return read_content_with_retry(
            page, reader, error_type=BrowserError, timeout_error_type=LoadTimeout,
        )

    def test_success_does_not_wait_or_retry(self):
        page, reader = Mock(), Mock(return_value="original HTML")
        self.assertEqual(self.read(page, reader), "original HTML")
        reader.assert_called_once_with(page)
        page.wait_for_load_state.assert_not_called()
        page.wait_for_timeout.assert_not_called()

    def test_navigation_race_retries_only_the_read(self):
        page = Mock()
        reader = Mock(side_effect=[BrowserError(NAVIGATION_CONTENT_ERROR), "loaded HTML"])
        self.assertEqual(self.read(page, reader), "loaded HTML")
        self.assertEqual(reader.call_count, 2)
        page.wait_for_load_state.assert_called_once_with("domcontentloaded", timeout=3000)
        page.wait_for_timeout.assert_called_once_with(250)
        self.assertEqual({call[0] for call in page.mock_calls},
                         {"wait_for_load_state", "wait_for_timeout"})

    def test_unrelated_browser_error_is_not_retried(self):
        page, error = Mock(), BrowserError("Target page has been closed")
        reader = Mock(side_effect=error)
        with self.assertRaises(BrowserError) as caught:
            self.read(page, reader)
        self.assertIs(caught.exception, error)
        self.assertEqual(reader.call_count, 1)
        self.assertEqual(page.mock_calls, [])

    def test_retries_are_bounded_and_original_error_propagates(self):
        page, error = Mock(), BrowserError(NAVIGATION_CONTENT_ERROR)
        reader = Mock(side_effect=error)
        with self.assertRaises(BrowserError) as caught:
            self.read(page, reader)
        self.assertIs(caught.exception, error)
        self.assertEqual(reader.call_count, 4)
        self.assertEqual(page.wait_for_load_state.call_count, 3)

    def test_load_timeout_still_allows_a_readable_page(self):
        page = Mock()
        page.wait_for_load_state.side_effect = LoadTimeout("loading timed out")
        reader = Mock(side_effect=[BrowserError(NAVIGATION_CONTENT_ERROR), "HTML"])
        self.assertEqual(self.read(page, reader), "HTML")
        self.assertEqual(reader.call_count, 2)

    def test_closed_page_during_wait_stays_an_error(self):
        page = Mock()
        page.wait_for_load_state.side_effect = BrowserError("Target closed")
        reader = Mock(side_effect=BrowserError(NAVIGATION_CONTENT_ERROR))
        with self.assertRaisesRegex(BrowserError, "Target closed"):
            self.read(page, reader)
        self.assertEqual(reader.call_count, 1)


if __name__ == "__main__":
    unittest.main()
