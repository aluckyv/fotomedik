"""The program's other ways of running, besides the window (FotoMedik.exe <flag> ...):

  --mcp            the MCP server Claude's desktop app talks to (fotomedik.mcp_server)
  --mcp-call <t>   one tool call of that server, for the Claude extension (JSON in, JSON out)
  --jobs           the runner of the queued repair jobs (fotomedik.jobs)
  --cli <args>     the command line (fotomedik.cli), e.g. --cli batch <folder> -o <out>

They are picked here, inside the code part of the app, so adding one needs no new base.
"""

from __future__ import annotations

import os
import sys


def _std_streams() -> None:
    """A windowed program has no sys.stdout/stderr unless it was handed files or pipes:
    take the handed ones when Python did not, and a sink otherwise (prints must not fail)."""
    for name, num in (("stdout", -11), ("stderr", -12)):
        if getattr(sys, name) is not None:
            continue
        stream = None
        if sys.platform == "win32":
            try:
                import ctypes
                import msvcrt

                k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
                k32.GetStdHandle.restype = ctypes.c_void_p
                h = k32.GetStdHandle(num)
                if h and h != ctypes.c_void_p(-1).value:
                    fd = msvcrt.open_osfhandle(h, os.O_WRONLY | os.O_TEXT)
                    stream = os.fdopen(fd, "w", encoding="utf-8", errors="replace", buffering=1)
            except Exception:  # noqa: BLE001
                stream = None
        setattr(sys, name, stream or open(os.devnull, "w", encoding="utf-8"))  # noqa: SIM115
    for name in ("stdout", "stderr"):  # the reports hold č, ć, š...: the packaged program ignores PYTHONIOENCODING
        try:
            getattr(sys, name).reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def _run(mode: str, args: list[str]) -> int:
    if mode == "--mcp":
        from .mcp_server import serve

        return serve()
    if mode == "--mcp-call":
        from .mcp_server import call_once

        return call_once(args[0] if args else "")
    if mode == "--jobs":
        from .jobs import run_queue

        return run_queue()
    from .cli import main

    try:
        main(args=args, prog_name="FotoMedik")
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    return 0


def dispatch(args: list[str]) -> int | None:
    """Run the mode the first argument asks for and return its exit code; None for the window.
    An error ends the mode with its story on stderr (in the packaged program an error left
    unhandled would open a message box and the process would wait for a click that never comes)."""
    if not args or args[0] not in ("--mcp", "--mcp-call", "--jobs", "--cli"):
        return None
    if args[0] in ("--cli", "--mcp-call"):
        _std_streams()
    try:
        return _run(args[0], args[1:])
    except KeyboardInterrupt:
        return 130
    except BaseException:  # noqa: BLE001
        import traceback

        try:
            traceback.print_exc()
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
        return 1
