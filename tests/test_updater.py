import json

from fotomedik.gui import updater


def _fake(release_build: int):
    rel = {"assets": [{"name": "version.json", "url": "V"},
                      {"name": "FotoMedik.exe", "url": "E", "size": 123}]}

    def get(url, accept="application/vnd.github+json"):
        if url == updater.API:
            return json.dumps(rel).encode()
        if url == "V":
            return json.dumps({"build": release_build, "commit": "abc"}).encode()
        raise AssertionError(url)

    return get


def test_newer_build_is_offered(monkeypatch):
    monkeypatch.setattr(updater, "BUILD", 5)
    monkeypatch.setattr(updater, "_get", _fake(7))
    upd = updater.check()
    assert upd is not None and upd.build == 7 and upd.exe_url == "E" and upd.size == 123


def test_same_build_is_not_offered(monkeypatch):
    monkeypatch.setattr(updater, "BUILD", 7)
    monkeypatch.setattr(updater, "_get", _fake(7))
    assert updater.check() is None
