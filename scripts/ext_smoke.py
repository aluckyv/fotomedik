"""CI check of FotoMedik's Claude extension with the built program: unpack the extension,
run its server with Node.js the way Claude does, and make a tool call through
``<exe> --mcp-call``.

    python scripts/ext_smoke.py <FotoMedik.exe>
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    exe = str(Path(sys.argv[1]).resolve())
    from fotomedik import claude_connect

    tmp = Path(tempfile.mkdtemp())
    # the extension of the installed program: Claude runs its command directly
    with zipfile.ZipFile(claude_connect.build_extension(tmp / "bin.mcpb", Path(exe))) as z:
        server = json.loads(z.read("manifest.json"))["server"]
    assert server["type"] == "binary", server
    cfg = server["mcp_config"]
    init = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}}
    call = {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "app_info", "arguments": {}}}
    p = subprocess.run([cfg["command"], *cfg["args"]], timeout=120, capture_output=True,
                       input=(json.dumps(init) + "\n" + json.dumps(call) + "\n").encode())
    replies = {r["id"]: r for r in map(json.loads, p.stdout.decode().splitlines())}
    if not replies.get(2, {}).get("result", {}).get("structuredContent", {}).get("packaged"):
        print("binary extension:", p.stdout[-1000:], p.stderr[-1000:])
        return 1
    print("::notice title=Extension smoke::the binary extension's command answered")
    pkg = claude_connect.build_extension(tmp / "FotoMedik.mcpb")
    with zipfile.ZipFile(pkg) as z:
        z.extractall(tmp / "ext")
    node = shutil.which("node")
    if node is None:
        print("no Node.js here")
        return 1
    msgs = [{"jsonrpc": "2.0", "id": 0, "method": "server/discover"},
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "app_info", "arguments": {}}}]
    # Claude may hand the script only a few variables (Node.js itself needs SystemRoot):
    # the script has to fill in the rest for the program
    env = {k: os.environ[k] for k in ("PATH", "USERPROFILE", "SYSTEMROOT", "FOTOMEDIK_HOME") if k in os.environ}
    env["FOTOMEDIK_EXE"] = exe
    p = subprocess.run([node, str(tmp / "ext" / "server" / "index.js")], env=env, timeout=180,
                       capture_output=True, input="".join(json.dumps(m) + "\n" for m in msgs).encode())
    print(p.stdout.decode("utf-8", "replace")[-2000:], p.stderr.decode("utf-8", "replace")[-2000:])
    replies = {r["id"]: r for r in map(json.loads, p.stdout.decode().splitlines())}
    if 2 not in replies:
        return 1
    info = replies[2]["result"]
    if info.get("isError") or not info["structuredContent"]["packaged"]:
        return 1
    print(f"::notice title=Extension smoke::build {info['structuredContent']['build']} answered through the extension")
    return 0


if __name__ == "__main__":
    sys.exit(main())
