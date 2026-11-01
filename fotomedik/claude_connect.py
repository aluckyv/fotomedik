"""Connect FotoMedik to Claude's desktop app, with one click and no questions.

FotoMedik's MCP server (FotoMedik.exe --mcp, see fotomedik.mcp_server) reaches Claude as a
desktop extension (FotoMedik.mcpb): a few lines of JavaScript that Claude runs with its own
Node.js and that start this program, passing Claude's messages to it and its answers back.
"Connect to Claude" brings an installed extension up to date in place (Claude uses it the
next time it starts the server), sees a FotoMedik entry in Claude's settings file as
connected too, and otherwise writes the package and shows it, to be dragged into Claude's
Settings > Extensions once. It opens no window of its own.
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
// (FotoMedik.exe --mcp) and pass Claude's messages to it and its answers back. The streams
// are passed on by hand: Claude's own Node.js gives this script streams, not handles that a
// child process could inherit.
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
const child = spawn(exe, args, { stdio: ["pipe", "pipe", "pipe"], windowsHide: true });
child.on("error", (e) => {
  process.stderr.write("FotoMedik did not start: " + e.message + "\n");
  process.exit(1);
});
process.stdin.pipe(child.stdin);
child.stdout.pipe(process.stdout);
child.stderr.on("data", (d) => process.stderr.write(d));
child.stdin.on("error", () => {});
process.stdout.on("error", () => {});
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


def _claude_dirs() -> list[Path]:
    """Claude's data folders: the usual install and the Microsoft Store one."""
    appdata = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    local = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    dirs = [appdata / "Claude"]
    pk = local / "Packages"
    if pk.is_dir():
        dirs += [p / "LocalCache" / "Roaming" / "Claude" for p in sorted(pk.glob("Claude_*"))]
    return [d for d in dirs if d.is_dir()]


def installed_extensions() -> list[Path]:
    """Folders of the FotoMedik extension installed in Claude."""
    found = []
    for d in _claude_dirs():
        for man in (d / "Claude Extensions").glob("*/manifest.json"):
            try:
                if json.loads(man.read_text(encoding="utf-8-sig")).get("name") == NAME:
                    found.append(man.parent)
            except (OSError, ValueError):
                continue
    return found


def in_settings() -> bool:
    """Whether Claude's settings file names a FotoMedik server (written by hand)."""
    for d in _claude_dirs():
        try:
            data = json.loads((d / "claude_desktop_config.json").read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and any(k.lower() == NAME for k in (data.get("mcpServers") or {})):
            return True
    return False


def connect() -> tuple[bool, str]:
    """(connected, one line for the status bar). Nothing is asked and no window opens; only
    when FotoMedik is not in Claude at all the package is written and shown in Explorer."""
    ext = installed_extensions()
    for d in ext:  # bring the installed script up to date in place
        (d / "server").mkdir(exist_ok=True)
        (d / "server" / "index.js").write_text(INDEX_JS, encoding="utf-8")
    if ext:
        return True, "Connected to Claude (FotoMedik extension, up to date)."
    if in_settings():
        return True, "Connected to Claude (Claude's settings)."
    path = build_extension()
    if sys.platform == "win32":
        try:
            subprocess.Popen(["explorer", "/select,", str(path)])
        except OSError:
            pass
    return False, f"Not in Claude yet: drag {path.name} into Claude > Settings > Extensions once."
