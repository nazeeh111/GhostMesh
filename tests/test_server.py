"""Exercise the actual HTTP contract and local file boundaries."""

import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest

from ghostmesh.server import make_server


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.replay = Path(self.tmp.name) / "record.json"
        self.replay.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "provenance": {"kind": "synthetic_diagnostic"},
                    "frames": [],
                }
            )
        )
        self.server = make_server(self.replay, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.tmp.cleanup()

    def request(self, path, method="GET", headers=None):
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=2
        )
        connection.request(method, path, headers=headers or {})
        result = connection.getresponse()
        status, response_headers, body = (
            result.status,
            dict(result.getheaders()),
            result.read(),
        )
        connection.close()
        return status, response_headers, body

    def test_static_and_replay_contract(self):
        status, headers, body = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn(b"GhostMesh", body)
        self.assertIn("script-src 'self'", headers["Content-Security-Policy"])
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        status, _, body = self.request("/api/replay")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["provenance"]["kind"], "synthetic_diagnostic")

    def test_no_arbitrary_file_or_upload_endpoint(self):
        for route in [
            "/../ghostmesh/server.py",
            "/%2e%2e/README.md",
            "/ghostmesh/server.py",
        ]:
            status, _, body = self.request(route)
            self.assertNotIn(b"root:", body)
            self.assertEqual(status, 404)
        status, _, body = self.request("/api/replay?path=/etc/passwd")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["schema_version"], 1)
        self.assertEqual(self.request("/api/replay", method="POST")[0], 501)

    def test_foreign_host_is_rejected(self):
        self.assertEqual(
            self.request("/api/replay", headers={"Host": "example.com"})[0], 403
        )

    def test_file_is_snapshotted_and_head_has_no_body(self):
        self.replay.write_text("broken replacement")
        self.assertEqual(self.request("/api/replay")[0], 200)
        status, headers, body = self.request("/app.js", method="HEAD")
        self.assertEqual(status, 200)
        self.assertGreater(int(headers["Content-Length"]), 0)
        self.assertEqual(body, b"")

    def test_nonlocal_binding_rejected(self):
        with self.assertRaisesRegex(ValueError, "loopback"):
            make_server(host="0.0.0.0", port=0)


if __name__ == "__main__":
    unittest.main()
