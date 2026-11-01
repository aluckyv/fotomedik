"""Connect FotoMedik to Claude's desktop app: add the FotoMedik MCP server to its settings
(claude_desktop_config.json, a backup copy is kept) and restart the app on request, so
Claude can repair folders on this computer (see fotomedik.mcp_server)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from . import install

SERVER = "fotomedik"
CONFIG = "claude_desktop_config.json"


def config_paths() -> list[Path]:
    """Where Claude's desktop app keeps its settings: the usual install and the Microsoft
    Store one (its own folder under Packages); the ones that exist, else the likely one."""
    appdata = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    local = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    places = [appdata / "Claude"]
    pk = local / "Packages"
    if pk.is_dir():
        places += [p / "LocalCache" / "Roaming" / "Claude" for p in sorted(pk.glob("Claude_*"))]
    found = [p / CONFIG for p in places if (p / CONFIG).exists()]
    if found:
        return found
    dirs = [p for p in places if p.is_dir()]
    if dirs:
        return [d / CONFIG for d in dirs]
    store = [p for p in places[1:] if p.parent.parent.is_dir()]  # the Store app, its settings not made yet
    return [(store or places)[0] / CONFIG]


def server_entry() -> dict:
    if getattr(sys, "frozen", False):
        exe = install.app_exe() if install.is_folder_app() else Path(sys.executable)
        return {"command": str(exe), "args": ["--mcp"]}
    return {"command": sys.executable, "args": ["-m", "fotomedik.mcp_server"],
            "env": {"PYTHONPATH": str(Path(__file__).resolve().parent.parent)}}


def connect(paths: list[Path] | None = None) -> list[str]:
    """Write the FotoMedik server into the settings; returns what was done, line by line.
    Other servers and settings stay as they are; the file before the change is copied
    to claude_desktop_config.json.bak-fotomedik-<time>."""
    done = []
    for path in paths or config_paths():
        data: dict = {}
        if path.exists():
            text = path.read_text(encoding="utf-8-sig")
            try:
                data = json.loads(text) if text.strip() else {}
            except ValueError as e:
                raise ValueError(f"{path} is not valid JSON ({e}); fix it or remove it first") from None
            if not isinstance(data, dict):
                raise ValueError(f"{path} does not hold settings (an object)")
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup = path.with_name(f"{CONFIG}.bak-fotomedik-{stamp}")
            shutil.copy2(path, backup)
            done.append(f"backup: {backup}")
        servers = data.get("mcpServers")
        if not isinstance(servers, dict):
            servers = {}
        servers[SERVER] = server_entry()
        data["mcpServers"] = servers
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        done.append(f"connected: {path}")
    return done


def is_connected() -> bool:
    for path in config_paths():
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and SERVER in (data.get("mcpServers") or {}):
            return True
    return False


# Claude's desktop app is claude.exe; so is Claude Code in a terminal, which must not be touched:
# only processes of the desktop install (the one with a window, or a known install folder) are restarted.
RESTART_PS = r"""
$ErrorActionPreference = 'SilentlyContinue'
Start-Sleep -Seconds 2
$all = Get-Process -Name claude
$main = $all | Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1
$paths = @()
if ($main) { $paths += $main.Path }
$paths += ($all | Where-Object { $_.Path -like '*\AnthropicClaude\*' -or $_.Path -like '*\WindowsApps\Claude_*' -or $_.Path -like '*\Programs\Claude\*' } | ForEach-Object { $_.Path })
$paths = $paths | Where-Object { $_ } | Select-Object -Unique
if (-not $paths) { exit 1 }
$all | Where-Object { $paths -contains $_.Path } | Stop-Process -Force
Start-Sleep -Seconds 4
$app = Get-StartApps | Where-Object { $_.Name -eq 'Claude' } | Select-Object -First 1
if ($app) { Start-Process ('shell:AppsFolder\' + $app.AppID) } else { Start-Process $paths[0] }
"""


def restart_claude() -> bool:
    """Close Claude's desktop app and start it again (in a process of its own, a couple
    of seconds later, so this program can answer first). False when not on Windows."""
    if sys.platform != "win32":
        return False
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    cmd = ["powershell", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", RESTART_PS]
    try:
        subprocess.Popen(cmd, creationflags=flags | 0x01000000, close_fds=True)  # out of any job object
    except OSError:
        subprocess.Popen(cmd, creationflags=flags, close_fds=True)
    return True
