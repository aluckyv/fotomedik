"""FotoMedik for Claude: the MCP server, the job queue and "Connect to Claude"."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from fotomedik import claude_connect, jobs, mcp_server
from fotomedik.cli import main as cli
from fotomedik.entry import dispatch
from synth import flip_bits

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    monkeypatch.setenv("FOTOMEDIK_HOME", str(h))
    monkeypatch.delenv("FOTOMEDIK_JOB_CMD", raising=False)
    yield h
    for j in jobs.all_jobs():  # leave nothing running
        if j.get("state") in ("queued", "running"):
            jobs.stop(j["id"])


@pytest.fixture
def photos(tmp_path, healthy_jpeg_rst):
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "a.jpg").write_bytes(healthy_jpeg_rst)
    data = bytearray(healthy_jpeg_rst)
    data[len(data) // 2] ^= 0x10
    (src / "b.jpg").write_bytes(bytes(data))
    (src / "sub" / "c.JPG").write_bytes(flip_bits(healthy_jpeg_rst, len(healthy_jpeg_rst) // 3, 0x04))
    (src / "notes.txt").write_text("not a photo")
    return src


def _hashes(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob("*") if p.is_file()}


def _wait(job_id: str, states=jobs.FINISHED, timeout: float = 120.0) -> dict:
    t = time.time()
    while time.time() - t < timeout:
        st = jobs.status(jobs.read(job_id))
        if st["state"] in states:
            return st
        time.sleep(0.3)
    raise AssertionError(f"job {job_id} still {jobs.read(job_id)['state']}: {jobs.status(jobs.read(job_id))}")


def test_constants_follow_the_program():
    from fotomedik.ai.run import STEPS
    from fotomedik.cli import IMAGE_SUFFIXES

    assert jobs.STEPS == STEPS
    assert jobs.IMAGE_SUFFIXES == IMAGE_SUFFIXES


def test_paths_that_could_harm_a_file_are_refused(home, photos, tmp_path):
    with pytest.raises(jobs.JobError, match="inside"):
        jobs.check_paths(str(photos), str(photos))
    with pytest.raises(jobs.JobError, match="inside"):
        jobs.check_paths(str(photos), str(photos / "sub" / "out"))
    with pytest.raises(jobs.JobError, match="contain"):
        jobs.check_paths(str(photos / "sub"), str(photos))
    with pytest.raises(jobs.JobError, match="does not exist"):
        jobs.check_paths(str(tmp_path / "nope"), str(tmp_path / "out"))
    (tmp_path / "file.jpg").write_bytes(b"x")
    with pytest.raises(jobs.JobError, match="is a file"):
        jobs.check_paths(str(photos), str(tmp_path / "file.jpg"))
    jobs.save_settings([str(tmp_path / "prot"), "  "])
    assert jobs.settings()["protected"] == [str(tmp_path / "prot")]
    with pytest.raises(jobs.JobError, match="protected"):
        jobs.check_paths(str(photos), str(tmp_path / "prot" / "deep" / "out"))
    src, out = jobs.check_paths(str(photos), str(tmp_path / "out"))
    assert src == photos.resolve() and out == (tmp_path / "out").resolve()
    with pytest.raises(jobs.JobError, match="unknown step"):
        jobs.new_job(str(photos), str(tmp_path / "out"), "puzzle,magic")
    with pytest.raises(jobs.JobError, match="not in the source"):
        jobs.new_job(str(photos), str(tmp_path / "out"), "", only=["a.jpg", "zz.jpg"])
    assert jobs.all_jobs() == []  # nothing was queued


def test_server_speaks_mcp_over_stdio(home, photos, tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    p = subprocess.Popen([sys.executable, "-m", "fotomedik.mcp_server"], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, env=env, cwd=str(tmp_path))
    msgs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "app_info", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
         "params": {"name": "start_job", "arguments": {"source": str(photos), "output": str(photos / "out")}}},
        {"jsonrpc": "2.0", "id": 5, "method": "nope"},
        {"jsonrpc": "2.0", "id": 6, "method": "ping"},
    ]
    out, _ = p.communicate("".join(json.dumps(m) + "\n" for m in msgs).encode(), timeout=60)
    assert p.returncode == 0
    replies = {r["id"]: r for r in map(json.loads, out.decode().splitlines())}
    assert sorted(replies) == [1, 2, 3, 4, 5, 6]  # the notification got no answer
    init = replies[1]["result"]
    assert init["protocolVersion"] == "2025-06-18" and init["serverInfo"]["name"] == "fotomedik"
    assert "tools" in init["capabilities"]
    names = {t["name"] for t in replies[2]["result"]["tools"]}
    assert names == {"start_job", "job_status", "list_jobs", "stop_job", "app_info"}
    for t in replies[2]["result"]["tools"]:
        assert t["inputSchema"]["type"] == "object" and t["description"]
    info = replies[3]["result"]
    assert not info["isError"] and info["structuredContent"]["home"] == str(home)
    refused = replies[4]["result"]
    assert refused["isError"] and "inside" in refused["content"][0]["text"]
    assert replies[5]["error"]["code"] == -32601
    assert replies[6]["result"] == {}


def test_a_job_repairs_a_folder_in_the_background(home, photos, tmp_path):
    before = _hashes(photos)
    out = tmp_path / "out"
    res = mcp_server.call_tool("start_job", {"source": str(photos), "output": str(out), "steps": ""})
    assert not res["isError"], res
    job = res["structuredContent"]
    assert job["state"] == "queued" and job["photos"] == 3 and job["to_do"] == 3
    st = _wait(job["id"])
    assert st["state"] == "done", st
    assert st["returncode"] == 0 and st["repaired_since_start"] >= 3
    assert (out / "a.jpg").exists() and (out / "b.jpg").exists() and (out / "sub" / "c.JPG").exists()
    assert _hashes(photos) == before  # originals untouched
    assert any("file(s)" in ln for ln in st["last_lines"])
    listed = mcp_server.call_tool("list_jobs", {})["structuredContent"]["jobs"]
    assert [j["id"] for j in listed] == [job["id"]]
    latest = mcp_server.call_tool("job_status", {})["structuredContent"]
    assert latest["id"] == job["id"]
    # everything has its result: a job that would do nothing is refused
    res = mcp_server.call_tool("start_job", {"source": str(photos), "output": str(out), "steps": "",
                                             "skip_done": True})
    assert res["isError"] and "already" in res["content"][0]["text"]
    (out / "b.jpg").unlink()
    res = mcp_server.call_tool("start_job", {"source": str(photos), "output": str(out), "steps": "",
                                             "skip_done": True})
    job2 = res["structuredContent"]
    assert job2["to_do"] == 1
    st = _wait(job2["id"])
    assert st["state"] == "done" and (out / "b.jpg").exists()
    assert any("2 photo(s) done already" in ln for ln in jobs._tail(jobs.jobs_dir() / f"{job2['id']}.out", 200))


def test_jobs_run_one_after_another_and_stop(home, photos, tmp_path, monkeypatch):
    monkeypatch.setenv("FOTOMEDIK_JOB_CMD", json.dumps([sys.executable, "-c", "import time; time.sleep(60)"]))
    a = jobs.new_job(str(photos), str(tmp_path / "o1"), "")
    b = jobs.new_job(str(photos), str(tmp_path / "o2"), "")
    c = jobs.new_job(str(photos), str(tmp_path / "o3"), "")
    st = _wait(a["id"], states=("running",), timeout=30)
    t = time.time()
    while not jobs.read(a["id"]).get("pid") and time.time() - t < 10:
        time.sleep(0.1)
    pid = jobs.read(a["id"])["pid"]
    assert jobs.pid_alive(pid) and jobs.runner_alive()
    assert jobs.read(b["id"])["state"] == "queued"
    assert "taken off" in jobs.stop(c["id"])
    assert "stopped" in mcp_server.call_tool("stop_job", {"job_id": a["id"]})["structuredContent"]["result"]
    t = time.time()
    while jobs.pid_alive(pid) and time.time() - t < 10:
        time.sleep(0.1)
    assert not jobs.pid_alive(pid)
    assert jobs.read(a["id"])["state"] == "stopped"
    _wait(b["id"], states=("running",), timeout=30)  # the queue goes on with the next job
    assert jobs.read(c["id"])["state"] == "cancelled"
    jobs.stop(b["id"])
    assert _wait(b["id"], timeout=30)["state"] == "stopped"
    assert "not running" in jobs.stop(a["id"])
    res = mcp_server.call_tool("job_status", {"job_id": "../etc"})
    assert res["isError"] and "no such job" in res["content"][0]["text"]


def test_batch_only_some_photos_and_skip_done(tmp_path, photos):
    out = tmp_path / "out"
    lst = tmp_path / "only.txt"
    lst.write_text("sub\\c.JPG\n\nA.jpg\n", encoding="utf-8")
    r = CliRunner().invoke(cli, ["batch", str(photos), "-o", str(out), "--only-file", str(lst), "--jobs", "1"])
    assert r.exit_code == 0, r.output
    assert (out / "a.jpg").exists() and (out / "sub" / "c.JPG").exists() and not (out / "b.jpg").exists()
    r = CliRunner().invoke(cli, ["batch", str(photos), "-o", str(out), "--skip-done", "--jobs", "1"])
    assert r.exit_code == 0, r.output
    assert "2 photo(s) done already" in r.output and (out / "b.jpg").exists()


def test_skipped_photos_still_lend_their_pixels(tmp_path, photos, monkeypatch):
    """With --skip-done and steps, every photo is repaired again (burst donors) but only
    the ones without an enhanced copy get one."""
    seen = {}

    def fake_run_extras(items, out, steps, say, on_done=None, quality=None, prompt=None, enlarger=None, skip=None):
        seen["names"] = [Path(src).name for _rep, src in items]
        seen["skip"] = sorted(Path(items[i][1]).name for i in skip)
        return ""

    import fotomedik.ai.run as run

    monkeypatch.setattr(run, "run_extras", fake_run_extras)
    out = tmp_path / "out"
    out.mkdir()
    (out / "a_enhanced.jpg").write_bytes(b"x")
    r = CliRunner().invoke(cli, ["batch", str(photos), "-o", str(out), "--skip-done", "--enhance", "burst",
                                 "--jobs", "1"])
    assert r.exit_code == 0, r.output
    assert sorted(seen["names"]) == ["a.jpg", "b.jpg", "c.JPG"] and seen["skip"] == ["a.jpg"]


def test_connect_to_claude_keeps_the_other_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    # no Claude settings anywhere yet: they are made where the usual install keeps them
    assert claude_connect.config_paths() == [tmp_path / "Roaming" / "Claude" / "claude_desktop_config.json"]
    cfg = tmp_path / "Roaming" / "Claude" / "claude_desktop_config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text('﻿{"mcpServers": {"bridge-after": {"command": "x.exe"}}, "theme": "dark"}', encoding="utf-8")
    assert not claude_connect.is_connected()
    done = claude_connect.connect()
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["theme"] == "dark" and data["mcpServers"]["bridge-after"] == {"command": "x.exe"}
    entry = data["mcpServers"]["fotomedik"]
    assert entry["args"][-1] in ("--mcp", "fotomedik.mcp_server")
    backups = list(cfg.parent.glob("claude_desktop_config.json.bak-fotomedik-*"))
    assert len(backups) == 1 and "bridge-after" in backups[0].read_text(encoding="utf-8")
    assert any(d.startswith("connected:") for d in done) and claude_connect.is_connected()
    # the Store app keeps its settings in its own package folder
    store = tmp_path / "Local" / "Packages" / "Claude_abc123" / "LocalCache" / "Roaming" / "Claude"
    store.mkdir(parents=True)
    (store / "claude_desktop_config.json").write_text("{}", encoding="utf-8")
    assert len(claude_connect.config_paths()) == 2
    cfg.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        claude_connect.connect()


def test_the_program_picks_its_mode(monkeypatch, capsys):
    assert dispatch([]) is None and dispatch(["photo.jpg"]) is None
    assert dispatch(["--cli", "--help"]) == 0
    assert dispatch(["--cli", "batch", "/no/such/folder", "-o", "/tmp/x"]) == 2

    def boom(**_kw):
        raise UnicodeEncodeError("charmap", "č", 0, 1, "can't encode")

    import fotomedik.cli

    monkeypatch.setattr(fotomedik.cli, "main", boom)
    assert dispatch(["--cli", "batch"]) == 1  # an error ends the mode (no message box in the packaged program)
    assert "UnicodeEncodeError" in capsys.readouterr().err
