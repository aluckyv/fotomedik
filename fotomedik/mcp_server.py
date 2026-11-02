"""FotoMedik for Claude: a small MCP server (``FotoMedik.exe --mcp``), stdio JSON-RPC.

Claude's desktop app starts it (see fotomedik.claude_connect) and Claude can then queue
folders for repair on this computer, follow them and stop them. The repairs run in
processes of their own (fotomedik.jobs), on this computer's graphics card, so they go
on when the conversation ends. Only the standard library is used here: the server
starts fast and holds no files of the program open (an update can still replace it).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from . import jobs

PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")

INSTRUCTIONS = (
    "FotoMedik repairs damaged JPEG photos on this computer. start_job queues a folder; jobs run one "
    "after another in the background (minutes per photo for the enhanced copies), so check them "
    "with job_status now and then instead of waiting. Originals are never changed: the output must "
    "be another folder, and protected places set by the person are refused. update_app brings the program "
    "up to the newest build between jobs."
)

TOOLS = [
    {
        "name": "start_job",
        "description": (
            "Queue a folder of damaged photos for repair into an output folder (subfolders too). "
            "Every photo gets a repaired copy; with steps, photos with holes also get <name>_enhanced.jpg. "
            "Jobs run one at a time in the background; returns the job id at once."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "folder with the damaged photos (only read)"},
                "output": {"type": "string", "description": "folder for the results; never the source, "
                           "inside it or around it; made when missing"},
                "steps": {"type": "string", "description": "enhanced-copy steps, comma separated, from "
                          f"{', '.join(jobs.STEPS)}; \"\" for the repair only (rows, rowsnear, holemap are experiments). "
                          f"Default: {jobs.DEFAULT_STEPS}"},
                "only": {"type": "array", "items": {"type": "string"},
                         "description": "only these photos (paths relative to source)"},
                "skip_done": {"type": "boolean", "description": "photos that already have their enhanced "
                              "copy in output (or repaired copy without steps) get no new one; the others are "
                              "done - a stopped or interrupted job continues this way", "default": False},
            },
            "required": ["source", "output"],
        },
    },
    {
        "name": "job_status",
        "description": ("How a job is going: its state, copies written since it started, seconds per enhanced "
                        "copy, errors, and its last messages. Without job_id: the running job, or the last one."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "lines": {"type": "integer", "description": "how many last messages (default 12)", "default": 12},
            },
        },
    },
    {
        "name": "list_jobs",
        "description": "All jobs, oldest first: queued, running and finished.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "stop_job",
        "description": ("Stop a running job at once, or take a queued one off the queue. Copies finished so far "
                        "stay; start_job with skip_done continues later."),
        "inputSchema": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]},
    },
    {
        "name": "update_app",
        "description": ("Bring FotoMedik up to the newest build on GitHub, between jobs (refused while one runs). "
                        "With install false only says whether a newer build exists. New jobs use the update at "
                        "once; app_info shows installed_build."),
        "inputSchema": {"type": "object", "properties": {
            "install": {"type": "boolean", "description": "download and put it in (default true)", "default": True}}},
    },
    {
        "name": "app_info",
        "description": ("The FotoMedik build, where it keeps its files, the AI models and runtimes it has, "
                        "whether an NVIDIA driver is present, the protected places, and whether a job runs."),
        "inputSchema": {"type": "object", "properties": {}},
    },
]


# ----------------------------------------------------------------------------- the tools


def _short(job: dict) -> dict:
    keys = ("id", "state", "source", "output", "steps", "photos", "to_do", "created", "started", "finished",
            "returncode")
    return {k: job.get(k) for k in keys}


def start_job(source: str, output: str, steps: str | None = None, only: list | None = None,
              skip_done: bool = False) -> dict:
    job = jobs.new_job(source, output, steps, only, skip_done)
    ahead = [j["id"] for j in jobs.all_jobs() if j.get("state") in ("queued", "running") and j["id"] != job["id"]]
    return {**_short(job), "jobs_ahead": ahead}


def job_status(job_id: str | None = None, lines: int = 12) -> dict:
    job = jobs.read(job_id) if job_id else jobs.latest()
    if job is None:
        return {"jobs": 0, "note": "no jobs yet"}
    return jobs.status(job, max(1, min(int(lines or 12), 200)))


def list_jobs() -> dict:
    return {"jobs": [_short(jobs.status(j, 0)) for j in jobs.all_jobs()]}


def stop_job(job_id: str) -> dict:
    return {"result": jobs.stop(job_id)}


def update_app(install: bool = True) -> dict:
    return jobs.update_program(bool(install))


def app_info() -> dict:
    from . import _build, install
    from .ai.cuda import nvidia_present
    from .ai.models import model_dir

    md = model_dir()
    models = sorted(f"{p.name} ({p.stat().st_size >> 20} MB)" for p in md.glob("*.onnx")) if md.is_dir() else []
    rt = md.parent / "runtime"
    runtimes = sorted(p.name for p in rt.iterdir() if p.is_dir()) if rt.is_dir() else []
    have = install.installed() or {}
    return {
        "build": _build.BUILD, "commit": _build.COMMIT[:7], "flavor": getattr(_build, "FLAVOR", "") or "",
        "installed_build": have.get("build"),  # on disk: what new jobs run (the build above is this server's)
        "packaged": bool(getattr(sys, "frozen", False)), "home": str(install.home()),
        "models_dir": str(md), "models": models, "ai_runtimes_downloaded": runtimes,
        "nvidia_driver": nvidia_present(), "protected": jobs.settings()["protected"],
        "runner_working": jobs.runner_alive(), "python": sys.version.split()[0],
    }


HANDLERS = {"start_job": start_job, "job_status": job_status, "list_jobs": list_jobs, "stop_job": stop_job,
            "update_app": update_app, "app_info": app_info}


def call_tool(name: str, args: dict) -> dict:
    """The MCP result of one tool call (errors come back as text with isError)."""
    fn = HANDLERS.get(name)
    if fn is None:
        return {"content": [{"type": "text", "text": f"unknown tool: {name}"}], "isError": True}
    try:
        res = fn(**(args or {}))
    except jobs.JobError as e:
        return {"content": [{"type": "text", "text": f"refused: {e}"}], "isError": True}
    except TypeError as e:
        return {"content": [{"type": "text", "text": f"wrong arguments: {e}"}], "isError": True}
    except Exception as e:  # noqa: BLE001  (the server keeps going)
        return {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}], "isError": True}
    return {"content": [{"type": "text", "text": json.dumps(res, indent=1, ensure_ascii=False)}],
            "structuredContent": res, "isError": False}


# ----------------------------------------------------------------------------- the protocol


def handle(msg: dict) -> dict | None:
    """The answer to one JSON-RPC message (None for notifications)."""
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:  # a notification (initialized, cancelled...): nothing to answer
        return None
    params = msg.get("params") or {}
    if method == "initialize":
        from . import _build

        asked = params.get("protocolVersion")
        result = {"protocolVersion": asked if asked in PROTOCOLS else PROTOCOLS[0],
                  "capabilities": {"tools": {"listChanged": False}},
                  "serverInfo": {"name": "fotomedik", "version": str(_build.BUILD)},
                  "instructions": INSTRUCTIONS}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        result = call_tool(params.get("name", ""), params.get("arguments") or {})
    elif method in ("resources/list", "prompts/list"):
        result = {method.split("/")[0]: []}
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _std_streams():
    """Binary stdin/stdout, also in the windowed program (its sys.stdin can be None even
    when the desktop app handed it pipes)."""
    inp = getattr(sys.stdin, "buffer", None)
    out = getattr(sys.stdout, "buffer", None)
    if (inp is None or out is None) and sys.platform == "win32":
        import ctypes
        import msvcrt

        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        k32.GetStdHandle.restype = ctypes.c_void_p
        if inp is None:
            h = k32.GetStdHandle(-10)
            inp = os.fdopen(msvcrt.open_osfhandle(h, os.O_RDONLY | os.O_BINARY), "rb", buffering=0)
        if out is None:
            h = k32.GetStdHandle(-11)
            out = os.fdopen(msvcrt.open_osfhandle(h, os.O_WRONLY | os.O_BINARY), "wb", buffering=0)
    if inp is None or out is None:
        raise SystemExit("no standard input/output to talk over")
    return inp, out


def serve() -> int:
    from . import install

    try:  # hold no folder of the program open (an update renames them)
        install.home().mkdir(parents=True, exist_ok=True)
        os.chdir(install.home())
    except OSError:
        pass
    inp, out = _std_streams()
    while True:
        line = inp.readline()
        if not line:
            return 0  # the desktop app closed the connection
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line.decode("utf-8"))
        except ValueError:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        else:
            batch = msg if isinstance(msg, list) else [msg]
            replies = [r for r in (handle(m) for m in batch if isinstance(m, dict)) if r is not None]
            if not replies:
                continue
            reply = replies if isinstance(msg, list) else replies[0]
        out.write(json.dumps(reply, ensure_ascii=False).encode("utf-8") + b"\n")
        out.flush()


def call_once(name: str) -> int:
    """One tool call for the extension (FotoMedik.exe --mcp-call <tool>): the arguments come
    as JSON on the input, the MCP result goes out as one JSON line."""
    raw = sys.stdin.buffer.read() if sys.stdin is not None else b""
    try:
        args = json.loads(raw.decode("utf-8-sig") or "{}")
    except ValueError:
        args = None
    res = call_tool(name, args) if isinstance(args, dict) else \
        {"content": [{"type": "text", "text": "the arguments are not a JSON object"}], "isError": True}
    sys.stdout.buffer.write(json.dumps(res, ensure_ascii=False).encode("utf-8") + b"\n")
    sys.stdout.flush()
    return 0


def main() -> None:
    sys.exit(serve())


if __name__ == "__main__":  # pragma: no cover
    main()
