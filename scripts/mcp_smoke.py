"""CI check of FotoMedik for Claude in the built program: talk MCP to ``<exe> --mcp`` over
pipes (the windowed program too), queue a repair of a folder, follow it to the end.

    python scripts/mcp_smoke.py <FotoMedik.exe> <photos folder> <output folder>

FOTOMEDIK_HOME should point at the installed app (the jobs are kept there).
"""

import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    exe, src, out = sys.argv[1], Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
    p = subprocess.Popen([exe, "--mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    n = 0

    def ask(method, params=None, notify=False):
        nonlocal n
        msg = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if not notify:
            n += 1
            msg["id"] = n
        p.stdin.write((json.dumps(msg) + "\n").encode())
        p.stdin.flush()
        if notify:
            return None
        line = p.stdout.readline()
        if not line:
            raise SystemExit(f"no answer to {method} (exit {p.poll()})")
        reply = json.loads(line)
        if "error" in reply:
            raise SystemExit(f"{method}: {reply['error']}")
        return reply["result"]

    def tool(name, **args):
        res = ask("tools/call", {"name": name, "arguments": args})
        print(f"{name}: {res['content'][0]['text'][:600]}")
        if res.get("isError"):
            raise SystemExit(f"{name} failed")
        return res["structuredContent"]

    init = ask("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                              "clientInfo": {"name": "ci", "version": "1"}})
    print("server:", init["serverInfo"])
    ask("notifications/initialized", notify=True)
    names = [t["name"] for t in ask("tools/list")["tools"]]
    print("tools:", names)
    info = tool("app_info")
    assert info["packaged"], "not the packaged program"
    job = tool("start_job", source=str(src), output=str(out), steps="")
    t = time.time()
    while True:
        st = tool("job_status", job_id=job["id"], lines=5)
        if st["state"] not in ("queued", "running"):
            break
        if time.time() - t > 600:
            raise SystemExit("the job did not finish in 10 minutes")
        time.sleep(3)
    p.stdin.close()
    p.wait(timeout=30)
    made = sorted(x.name for x in out.glob("*.jpg"))
    print(f"job {st['state']}, exit code {st['returncode']}, outputs: {made}")
    if st["state"] != "done" or not made:
        return 1
    print(f"::notice title=MCP smoke::{len(names)} tools, job done, {len(made)} photo(s) repaired by the packaged program")
    return 0


if __name__ == "__main__":
    sys.exit(main())
