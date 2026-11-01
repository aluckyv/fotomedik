"""Connect FotoMedik to Claude's desktop app, with one click and no questions.

FotoMedik's MCP server (FotoMedik.exe --mcp, see fotomedik.mcp_server) reaches Claude as a
desktop extension (FotoMedik.mcpb): a small MCP server in JavaScript that Claude runs with
its own Node.js; it answers Claude itself and does each tool call with this program
(FotoMedik.exe --mcp-call <tool>).
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

INDEX_JS_TEMPLATE = r"""// FotoMedik for Claude: the MCP server of the FotoMedik extension, run by Claude's own Node.js.
// It talks to Claude itself (the handshake never waits for the program) and does each tool
// call with the FotoMedik program of this computer: FotoMedik.exe --mcp-call <tool>, the
// arguments on its input, the result on its output. When the program cannot run, the
// call says why instead of the connection dropping.
const { spawn } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");
const readline = require("readline");

const TOOLS = __TOOLS__;
const INSTRUCTIONS = __INSTRUCTIONS__;
const VERSION = __VERSION__;
const PROTOCOLS = ["2025-06-18", "2025-03-26", "2024-11-05"];

const home = os.homedir();
const local = process.env.LOCALAPPDATA || path.join(home, "AppData", "Local");
const exe = process.env.FOTOMEDIK_EXE || path.join(local, "FotoMedik", "app", "base", "FotoMedik.exe");
const prefix = process.env.FOTOMEDIK_ARGS ? JSON.parse(process.env.FOTOMEDIK_ARGS) : [];

function childEnv() {
  // Claude may start this script with only a few variables; Windows programs need these
  const env = Object.assign({}, process.env);
  const root = env.SystemRoot || env.SYSTEMROOT || env.windir || "C:\\Windows";
  const put = (k, v) => { if (!env[k]) env[k] = v; };
  put("SystemRoot", root); put("windir", root);
  put("USERPROFILE", home); put("LOCALAPPDATA", local);
  put("APPDATA", path.join(home, "AppData", "Roaming"));
  put("TEMP", path.join(local, "Temp")); put("TMP", env.TEMP);
  put("HOMEDRIVE", home.slice(0, 2)); put("HOMEPATH", home.slice(2));
  const sys32 = path.join(root, "System32");
  env.PATH = (env.PATH || env.Path || "") + path.delimiter + sys32 + path.delimiter + root;
  env.PYTHONIOENCODING = "utf-8";
  return env;
}

function fail(text) {
  return { content: [{ type: "text", text: text }], isError: true };
}

function callTool(name, args) {
  return new Promise((resolve) => {
    if (!fs.existsSync(exe)) return resolve(fail("FotoMedik is not installed here (" + exe + "): start FotoMedik once."));
    const out = [], err = [];
    let child;
    try {
      child = spawn(exe, prefix.concat(["--mcp-call", name]),
                    { env: childEnv(), windowsHide: true, stdio: ["pipe", "pipe", "pipe"] });
    } catch (e) {
      return resolve(fail("FotoMedik did not start (" + exe + "): " + e.message));
    }
    const timer = setTimeout(() => child.kill(), 120000);
    child.stdout.on("data", (d) => out.push(d));
    child.stderr.on("data", (d) => err.push(d));
    child.stdin.on("error", () => {});
    child.on("error", (e) => { clearTimeout(timer); resolve(fail("FotoMedik did not start (" + exe + "): " + e.message)); });
    child.on("close", (code) => {
      clearTimeout(timer);
      const text = Buffer.concat(out).toString("utf8").trim();
      const last = text.split("\n").pop();
      try {
        resolve(JSON.parse(last));
      } catch (e) {
        const why = (Buffer.concat(err).toString("utf8") || text).slice(-3000);
        resolve(fail("FotoMedik ended with code " + code + " without an answer:\n" + why));
      }
    });
    child.stdin.end(JSON.stringify(args || {}));
  });
}

function send(msg) {
  process.stdout.write(JSON.stringify(msg) + "\n");
}

async function handle(msg) {
  if (msg.id === undefined || msg.id === null) return;  // a notification
  const p = msg.params || {};
  let result;
  if (msg.method === "initialize") {
    result = { protocolVersion: PROTOCOLS.includes(p.protocolVersion) ? p.protocolVersion : PROTOCOLS[0],
               capabilities: { tools: { listChanged: false } },
               serverInfo: { name: "fotomedik", version: VERSION }, instructions: INSTRUCTIONS };
  } else if (msg.method === "ping") {
    result = {};
  } else if (msg.method === "tools/list") {
    result = { tools: TOOLS };
  } else if (msg.method === "tools/call") {
    result = await callTool(p.name || "", p.arguments || {});
  } else if (msg.method === "resources/list" || msg.method === "prompts/list") {
    result = {}; result[msg.method.split("/")[0]] = [];
  } else {
    return send({ jsonrpc: "2.0", id: msg.id, error: { code: -32601, message: "method not found: " + msg.method } });
  }
  send({ jsonrpc: "2.0", id: msg.id, result: result });
}

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
let pending = 0, closed = false;
rl.on("line", (line) => {
  line = line.trim();
  if (!line) return;
  let msg;
  try { msg = JSON.parse(line); } catch (e) {
    return send({ jsonrpc: "2.0", id: null, error: { code: -32700, message: "parse error" } });
  }
  pending++;
  handle(msg).catch((e) => send({ jsonrpc: "2.0", id: msg.id, error: { code: -32603, message: String(e) } }))
    .finally(() => { pending--; if (closed && pending === 0) process.exit(0); });
});
rl.on("close", () => { closed = true; if (pending === 0) process.exit(0); });
"""


def index_js() -> str:
    from . import _build
    from .mcp_server import INSTRUCTIONS, TOOLS

    return (INDEX_JS_TEMPLATE.replace("__TOOLS__", json.dumps(TOOLS))
            .replace("__INSTRUCTIONS__", json.dumps(INSTRUCTIONS))
            .replace("__VERSION__", json.dumps(str(_build.BUILD))))


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
        z.writestr("server/index.js", index_js())
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
        (d / "server" / "index.js").write_text(index_js(), encoding="utf-8")
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
