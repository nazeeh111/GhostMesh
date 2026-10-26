"""Small, local-only replay viewer. It never accepts uploads or executes code."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import ipaddress
import json
from urllib.parse import urlsplit


STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/replay.js": ("replay.js", "text/javascript; charset=utf-8"),
}
MAX_REPLAY_BYTES = 100 * 1024 * 1024


def _read_replay(replay_path: str | Path | None) -> bytes | None:
    if replay_path is None:
        return None
    path = Path(replay_path)
    if path.stat().st_size > MAX_REPLAY_BYTES:
        raise ValueError("Replay exceeds 100 MiB; export a shorter sequence.")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda x: (_ for _ in ()).throw(
            ValueError(f"Nonfinite JSON value: {x}")
        ),
    )
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Expected GhostMesh replay schema_version 1.")
    # Snapshot once: changing a file on disk cannot silently change an open replay.
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")


def make_handler(
    replay_path: str | Path | None = None, *, web_root: str | Path | None = None
) -> type[BaseHTTPRequestHandler]:
    """Return a handler with an immutable replay and an exact static-file allowlist."""
    replay = _read_replay(replay_path)
    root = (
        Path(web_root)
        if web_root is not None
        else Path(__file__).resolve().parent / "web"
    )
    assets = {
        route: ((root / filename).read_bytes(), content_type)
        for route, (filename, content_type) in STATIC_FILES.items()
    }

    class ReplayHandler(BaseHTTPRequestHandler):
        server_version = "GhostMesh/1"
        sys_version = ""

        def _response(
            self, status: int, body: bytes, content_type: str, *, head: bool = False
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
            )
            self.end_headers()
            if not head:
                self.wfile.write(body)

        def _get(self, *, head: bool = False) -> None:
            # Reject cross-site requests and DNS rebinding. A page on another host
            # cannot use this localhost service as a file-reading proxy.
            host = self.headers.get("Host", "").split(":", 1)[0].lower()
            if host not in {"localhost", "127.0.0.1"}:
                self._response(
                    HTTPStatus.FORBIDDEN,
                    b"Local host required.\n",
                    "text/plain; charset=utf-8",
                    head=head,
                )
                return
            path = urlsplit(self.path).path
            if path == "/api/health":
                self._response(
                    HTTPStatus.OK,
                    b'{"status":"ok","mode":"local_replay"}',
                    "application/json",
                    head=head,
                )
            elif path == "/api/replay":
                if replay is None:
                    self._response(
                        HTTPStatus.NOT_FOUND,
                        b'{"error":"No replay configured"}',
                        "application/json",
                        head=head,
                    )
                else:
                    self._response(HTTPStatus.OK, replay, "application/json", head=head)
            elif path in assets:
                body, content_type = assets[path]
                self._response(HTTPStatus.OK, body, content_type, head=head)
            else:
                self._response(
                    HTTPStatus.NOT_FOUND,
                    b"Not found.\n",
                    "text/plain; charset=utf-8",
                    head=head,
                )

        def do_GET(self) -> None:
            self._get()

        def do_HEAD(self) -> None:
            self._get(head=True)

        def log_message(self, format: str, *args: object) -> None:
            # Avoid recording user-supplied replay data or arbitrary request text.
            return

    return ReplayHandler


def make_server(
    replay_path: str | Path | None = None, host: str = "127.0.0.1", port: int = 8765
) -> ThreadingHTTPServer:
    """Construct a testable server. Binding is restricted to IPv4 loopback."""
    if host == "localhost":
        host = "127.0.0.1"
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError(
            "Use localhost or 127.0.0.1 for the local replay viewer."
        ) from exc
    if address.version != 4 or not address.is_loopback:
        raise ValueError("The replay viewer only binds to IPv4 loopback.")
    return ThreadingHTTPServer((host, port), make_handler(replay_path))


def serve(
    replay_path: str | Path | None = None, host: str = "127.0.0.1", port: int = 8765
) -> None:
    """Serve until interrupted. No browser is opened automatically."""
    with make_server(replay_path, host, port) as server:
        print(f"GhostMesh replay: http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
