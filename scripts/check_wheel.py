"""Check an installed wheel outside the checkout, including its default replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from urllib.request import build_opener, ProxyHandler


PROBE = r"""
import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path
import sys

import ghostmesh

site = Path(sys.argv[1]).resolve()
package = Path(ghostmesh.__file__).resolve().parent
if not package.is_relative_to(site):
    raise RuntimeError(f"Imported checkout instead of installed wheel: {package}")
dist = distribution("ghostmesh")
if Path(dist.locate_file("")).resolve() != site:
    raise RuntimeError("Package metadata did not come from the installed wheel")
if (dist.metadata.get("Name") != "ghostmesh" or not dist.version
        or not dist.metadata.get("Requires-Python")
        or dist.metadata.get("Description-Content-Type") != "text/markdown"
        or "GhostMesh" not in dist.metadata.get_payload()):
    raise RuntimeError("Wheel metadata lacks its identity, Python requirements or README")
if dist.metadata.get("License-Expression") != "MIT AND CC-BY-NC-4.0":
    raise RuntimeError("Wheel metadata lacks the declared software/data license")
licenses = {"LICENSE", "NOTICE.md", "data/LICENSE.md"}
if not licenses <= set(dist.metadata.get_all("License-File") or []):
    raise RuntimeError("Wheel metadata omits a required license file")
for name in licenses:
    matches = [p for p in dist.files or [] if str(p).endswith("/licenses/" + name)]
    if len(matches) != 1 or not dist.locate_file(matches[0]).read_bytes():
        raise RuntimeError(f"Missing packaged license: {name}")
if not any(p.name == "ghostmesh" and p.value == "ghostmesh.cli:main"
           for p in dist.entry_points):
    raise RuntimeError("Wheel lacks its ghostmesh command entry point")
static = {}
for name in ("index.html", "app.js", "replay.js", "style.css"):
    content = (package / "web" / name).read_bytes()
    if not content:
        raise RuntimeError(f"Empty packaged web asset: {name}")
    static[name] = hashlib.sha256(content).hexdigest()
replay_path = package / "examples" / "heldout-replay.json"
if not replay_path.is_file():
    raise RuntimeError("Wheel lacks the real default examples/heldout-replay.json")
replay = json.loads(replay_path.read_text())
if (replay.get("schema_version") != 1
        or replay.get("provenance", {}).get("kind") != "dataset_inference"
        or not replay.get("frames")):
    raise RuntimeError("Default replay must contain actual dataset inference frames")
replay_bytes = json.dumps(replay, allow_nan=False, separators=(",", ":")).encode()
print("WHEEL_PROBE " + json.dumps({
    "package": str(package), "version": dist.version, "static_sha256": static,
    "replay_sha256": hashlib.sha256(replay_bytes).hexdigest(),
}), flush=True)
from ghostmesh.cli import main
main(["serve", "--port", "0"])
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    args = parser.parse_args(argv)
    site = args.site.resolve(strict=True)
    if not site.is_dir():
        raise ValueError("--site must be an installed-wheel directory")
    environment = {
        **os.environ,
        "PYTHONPATH": str(site),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    with tempfile.TemporaryDirectory(prefix="ghostmesh-wheel-check-") as temporary:
        log_path = Path(temporary) / "server.log"
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", "-c", PROBE, str(site)],
                cwd=temporary,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 30
                while True:
                    output = log_path.read_text()
                    address = re.search(r"http://127\.0\.0\.1:\d+", output)
                    proof = re.search(r"^WHEEL_PROBE (.+)$", output, re.MULTILINE)
                    if process.poll() is not None:
                        raise RuntimeError(f"Installed-wheel startup failed:\n{output}")
                    if address and proof:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            f"Installed-wheel startup timed out:\n{output}"
                        )
                    time.sleep(0.05)
                evidence = json.loads(proof[1])
                # Ignore proxy environment variables for this owned loopback server.
                opener = build_opener(ProxyHandler({}))

                def get(route):
                    with opener.open(address[0] + route, timeout=5) as response:
                        if response.status != 200:
                            raise RuntimeError(f"Unexpected HTTP status for {route}")
                        return response.read()

                if json.loads(get("/api/health")) != {
                    "status": "ok",
                    "mode": "local_replay",
                }:
                    raise RuntimeError("Installed server health response differs")
                if (
                    hashlib.sha256(get("/api/replay")).hexdigest()
                    != evidence["replay_sha256"]
                ):
                    raise RuntimeError(
                        "Server did not serve its packaged default replay"
                    )
                for name, digest in evidence["static_sha256"].items():
                    route = "/" if name == "index.html" else "/" + name
                    if hashlib.sha256(get(route)).hexdigest() != digest:
                        raise RuntimeError(f"Installed HTTP asset differs: {name}")
                print(
                    json.dumps(
                        {"status": "installed_wheel_verified", **evidence}, indent=2
                    )
                )
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)


if __name__ == "__main__":
    main()
