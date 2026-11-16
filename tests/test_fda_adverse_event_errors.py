"""Failed FDA requests must not be reported as successful searches with no events."""

import unittest
from unittest.mock import patch

import requests
from biomni.tool.pharmacology import query_fda_adverse_events


class FDAAdverseEventErrorsTest(unittest.TestCase):
    def test_http_failure_is_reported_as_error(self):
        response = requests.Response()
        response.status_code = 500
        response.url = "https://api.fda.gov/drug/event.json"
        with patch("requests.sessions.Session.request", return_value=response) as request:
            summary = query_fda_adverse_events("aspirin")
        self.assertEqual(request.call_count, 1)
        self.assertIn("Error querying FDA adverse events for aspirin:", summary)
        self.assertIn("500", summary)
        self.assertNotIn("No adverse events found", summary)

    def test_exhausted_timeouts_are_reported_as_error(self):
        with (
            patch("requests.sessions.Session.request", side_effect=requests.exceptions.Timeout) as request,
            patch("time.sleep"),
        ):
            summary = query_fda_adverse_events("aspirin")
        self.assertEqual(request.call_count, 3)
        self.assertIn("Error querying FDA adverse events for aspirin:", summary)
        self.assertIn("timed out", summary)
        self.assertNotIn("No adverse events found", summary)

    def test_no_matches_preserves_empty_result_message(self):
        response = requests.Response()
        response.status_code = 404
        with patch("requests.sessions.Session.request", return_value=response) as request:
            summary = query_fda_adverse_events("aspirin")
        self.assertEqual(request.call_count, 1)
        self.assertEqual(summary, "No adverse events found for aspirin in the FDA database.")


if __name__ == "__main__":
    unittest.main()
