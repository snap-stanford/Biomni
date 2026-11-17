"""Offline checks for timeouts and GET retries in the shared database REST helper."""

import json
import unittest
from unittest.mock import patch

import requests
from biomni.tool.database import _query_rest_api

URL = "https://example.org/records"
PAYLOAD = {"records": [{"id": "P01308"}]}


def response(status=200, headers=None):
    result = requests.Response()
    result.status_code = status
    result.url = URL
    result.headers.update(headers or {})
    result._content = json.dumps(PAYLOAD if status == 200 else {"message": "failure"}).encode()
    result._content_consumed = True  # as after Session.send without stream=True
    return result


class RestApiRetryTest(unittest.TestCase):
    def setUp(self):
        request_patch = patch("requests.sessions.Session.request")
        self.request = request_patch.start()
        self.addCleanup(request_patch.stop)
        sleep_patch = patch("biomni.tool.database.time.sleep")
        self.sleep = sleep_patch.start()
        self.addCleanup(sleep_patch.stop)

    def test_success_uses_timeout_and_single_request(self):
        self.request.return_value = response()
        self.assertEqual(_query_rest_api(URL)["result"], PAYLOAD)
        self.assertEqual(self.request.call_count, 1)
        self.assertEqual(self.request.call_args.kwargs["timeout"], (5, 30))

    def test_transient_get_failures_are_retried(self):
        failures = [response(s) for s in (429, 500, 502, 503, 504)]
        failures += [requests.ConnectionError("reset"), requests.ReadTimeout("read")]
        for failure in failures:
            with self.subTest(failure=failure):
                self.request.reset_mock()
                self.request.side_effect = [failure, response()]
                self.assertEqual(_query_rest_api(URL, params={"gene": "INS"})["result"], PAYLOAD)
                self.assertEqual(self.request.call_count, 2)
                first, second = self.request.call_args_list
                self.assertEqual(first, second)

    def test_persistent_failure_stops_after_three_attempts(self):
        self.request.side_effect = [response(503) for _ in range(3)]
        result = _query_rest_api(URL)
        self.assertFalse(result["success"])
        self.assertIn("503", result["error"])
        self.assertEqual(self.request.call_count, 3)
        self.assertEqual(self.sleep.call_count, 2)

    def test_client_errors_are_not_retried(self):
        self.request.side_effect = [response(404), response()]
        self.assertFalse(_query_rest_api(URL)["success"])
        self.assertEqual(self.request.call_count, 1)

    def test_post_is_not_retried(self):
        self.request.side_effect = [response(503), response()]
        self.assertFalse(_query_rest_api(URL, method="POST", json_data={"q": 1})["success"])
        self.assertEqual(self.request.call_count, 1)
        self.assertEqual(self.request.call_args.kwargs["timeout"], (5, 30))

    def test_retry_after(self):
        self.request.side_effect = [response(429, {"Retry-After": "2"}), response()]
        self.assertTrue(_query_rest_api(URL)["success"])
        self.sleep.assert_called_once_with(2.0)

        self.request.reset_mock()
        self.sleep.reset_mock()
        self.request.side_effect = [response(429, {"Retry-After": "120"}), response()]
        self.assertFalse(_query_rest_api(URL)["success"])
        self.assertEqual(self.request.call_count, 1)
        self.sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
