import json

from fotomedik.gui import updater


def _fake(release_build: int, cuda: bool = False):
    rel = {"assets": [{"name": "version.json", "url": "V"},
                      {"name": "FotoMedik.exe", "url": "E", "size": 123}]}
    if cuda:
        rel["assets"] += [{"name": "FotoMedik-CUDA.exe", "url": "C", "size": 456}, {"name": "version-cuda.json", "url": "V"}]

    def get(url, accept="application/vnd.github+json"):
        if url == updater.API:
            return json.dumps(rel).encode()
        if url == "V":
            return json.dumps({"build": release_build, "commit": "abc"}).encode()
        raise AssertionError(url)

    return get


def _nvidia(monkeypatch, present: bool):
    monkeypatch.setattr(updater, "wanted_flavor", lambda: "cuda" if present else "")


def test_newer_build_is_offered(monkeypatch):
    _nvidia(monkeypatch, False)
    monkeypatch.setattr(updater, "BUILD", 5)
    monkeypatch.setattr(updater, "_get", _fake(7))
    upd = updater.check()
    assert upd is not None and upd.build == 7 and upd.exe_url == "E" and upd.size == 123


def test_same_build_is_not_offered(monkeypatch):
    _nvidia(monkeypatch, False)
    monkeypatch.setattr(updater, "BUILD", 7)
    monkeypatch.setattr(updater, "_get", _fake(7))
    assert updater.check() is None


def test_restart_does_not_inherit_packaged_app_variables(monkeypatch):
    from fotomedik.gui import updater

    monkeypatch.setenv("_PYI_APPLICATION_HOME_DIR", r"C:\Temp\_MEI123")
    monkeypatch.setenv("_PYI_PARENT_PROCESS_LEVEL", "1")
    monkeypatch.setenv("_MEIPASS2", r"C:\Temp\_MEI123")
    monkeypatch.setenv("PATH", "/bin")
    env = updater.fresh_env()
    assert not any(k.startswith("_PYI_") or k == "_MEIPASS2" for k in env)
    assert env["PYINSTALLER_RESET_ENVIRONMENT"] == "1" and env["PATH"] == "/bin"


def test_nvidia_card_gets_the_cuda_build_once(monkeypatch):
    from fotomedik import _build

    _nvidia(monkeypatch, True)
    monkeypatch.setattr(updater, "BUILD", 7)
    monkeypatch.setattr(updater, "_get", _fake(7, cuda=True))
    monkeypatch.setattr(_build, "FLAVOR", "")
    upd = updater.check()
    assert upd is not None and upd.exe_url == "C" and upd.size == 456  # same build, other kind
    monkeypatch.setattr(_build, "FLAVOR", "cuda")
    assert updater.check() is None  # already the CUDA build


def test_no_cuda_build_published_keeps_the_plain_one(monkeypatch):
    _nvidia(monkeypatch, True)
    monkeypatch.setattr(updater, "BUILD", 5)
    monkeypatch.setattr(updater, "_get", _fake(7, cuda=False))
    upd = updater.check()
    assert upd is not None and upd.exe_url == "E"
