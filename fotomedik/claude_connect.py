"""Connect FotoMedik to Claude's desktop app as a desktop extension (FotoMedik.mcpb).

Claude reads its list of MCP servers only when it starts, and while it runs it writes its
settings file back from memory, so a server written into that file by someone else gets
lost. An extension is installed by Claude itself: FotoMedik makes the package and opens
it, Claude shows its Install window, one click, and FotoMedik is in Claude's tools at once,
without closing Claude. The package holds a few lines of JavaScript that Claude runs with
its own Node.js; they start this program as the MCP server (FotoMedik.exe --mcp, see
fotomedik.mcp_server) and hand it Claude's connection as it is.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from . import install

NAME = "fotomedik"
PACKAGE = "FotoMedik.mcpb"

INDEX_JS = r"""// FotoMedik for Claude: start the FotoMedik program of this computer as the MCP server
// (FotoMedik.exe --mcp) and hand it Claude's connection (stdin/stdout) as it is.
const { spawn } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const local = process.env.LOCALAPPDATA || path.join(os.homedir(), "AppData", "Local");
const exe = process.env.FOTOMEDIK_EXE || path.join(local, "FotoMedik", "app", "base", "FotoMedik.exe");
const args = process.env.FOTOMEDIK_ARGS ? JSON.parse(process.env.FOTOMEDIK_ARGS) : ["--mcp"];
if (!fs.existsSync(exe)) {
  process.stderr.write("FotoMedik is not installed here (" + exe + "): start FotoMedik once.\n");
  process.exit(1);
}
const child = spawn(exe, args, { stdio: "inherit", windowsHide: true });
child.on("error", (e) => {
  process.stderr.write("FotoMedik did not start: " + e.message + "\n");
  process.exit(1);
});
child.on("exit", (code) => process.exit(code === null ? 1 : code));
for (const sig of ["SIGINT", "SIGTERM"]) process.on(sig, () => child.kill());
"""


def manifest() -> dict:
    from . import _build
    from .mcp_server import TOOLS

    env = {}
    if getattr(sys, "frozen", False) and install.is_folder_app():
        env["FOTOMEDIK_EXE"] = str(install.app_exe())  # also when FotoMedik lives somewhere else
    return {
        "manifest_version": "0.3",
        "name": NAME,
        "display_name": "FotoMedik",
        "version": f"1.0.{_build.BUILD}",
        "description": "Repair damaged JPEG photos on this computer: Claude queues folders, follows "
                       "the repairs and stops them. Originals are never changed.",
        "author": {"name": "FotoMedik"},
        "server": {
            "type": "node",
            "entry_point": "server/index.js",
            "mcp_config": {"command": "node", "args": ["${__dirname}/server/index.js"], "env": env},
        },
        "tools": [{"name": t["name"], "description": t["description"]} for t in TOOLS],
        "compatibility": {"platforms": ["win32"]},
    }


def build_extension(dest: Path | None = None) -> Path:
    """Write the extension package (a zip: manifest.json + server/index.js)."""
    dest = dest or install.home() / PACKAGE
    dest.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest(), indent=2, ensure_ascii=False))
        z.writestr("server/index.js", INDEX_JS)
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(buf.getvalue())
    os.replace(tmp, dest)
    return dest


def _open(path: Path) -> bool:
    """Open the package with the program Windows has for it (Claude); False when none."""
    if sys.platform != "win32":
        return False
    try:
        os.startfile(str(path))  # type: ignore[attr-defined]
        return True
    except OSError:
        try:  # no program for .mcpb: show the file, to be dragged into Claude
            subprocess.Popen(["explorer", "/select,", str(path)])
        except OSError:
            pass
        return False


def connect(open_it: bool = True) -> list[str]:
    """Make the extension and open it in Claude; returns what was done, line by line."""
    path = build_extension()
    lines = [f"extension: {path}"]
    if open_it and _open(path):
        lines.append("Claude shows the FotoMedik extension: click Install. FotoMedik is in Claude's "
                     "tools right away, Claude keeps running.")
    else:
        lines.append("In Claude open Settings > Extensions and drag this file there (or double-click "
                     "the file), then click Install.")
    return lines
