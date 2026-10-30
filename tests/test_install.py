"""The installed folder app: first install, part-by-part updates, the swap."""

import json
import zipfile

import pytest

from fotomedik import install


def _zip(path, files: dict):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)


def _release(tmp_path, build, base_id, code_text="v1"):
    a = tmp_path / f"assets{build}"
    a.mkdir()
    _zip(a / install.BASE_ZIP, {"FotoMedik.exe": b"exe" + base_id.encode(), "_internal/python312.dll": b"py"})
    _zip(a / install.CODE_ZIP, {"fotomedik/__init__.py": code_text, "fotomedik/jpeg/_fastdec.dll": b"dll"})
    import hashlib

    info = {"build": build, "commit": "c", "base_id": base_id,
            "base_sha256": hashlib.sha256((a / install.BASE_ZIP).read_bytes()).hexdigest(),
            "code_sha256": hashlib.sha256((a / install.CODE_ZIP).read_bytes()).hexdigest()}
    (a / install.VERSION).write_text("﻿" + json.dumps(info), encoding="utf-8")  # PowerShell may add a BOM
    return a


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("FOTOMEDIK_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def test_first_install_then_code_only_update(tmp_path, monkeypatch, home):
    monkeypatch.setenv("FOTOMEDIK_ASSETS", str(_release(tmp_path, 10, "B1")))
    rel = install.Release()
    assert install.needs_update(rel, install.installed())
    assert install.stage(rel) == ["base", "code"]
    assert install.installed() is None and install.pending()  # nothing touched before the swap
    assert install.apply_pending()
    have = install.installed()
    assert have["build"] == 10 and install.app_exe().read_bytes() == b"exeB1"
    assert (install.app_dir() / "code" / "fotomedik" / "jpeg" / "_fastdec.dll").exists()

    monkeypatch.setenv("FOTOMEDIK_ASSETS", str(_release(tmp_path, 11, "B1", "v2")))
    rel = install.Release()
    assert install.needs_update(rel, have)
    assert install.stage(rel) == ["code"]  # same base: only the code comes
    assert (install.app_dir() / "code" / "fotomedik" / "__init__.py").read_text() == "v1"  # until the swap
    install.apply_pending()
    assert (install.app_dir() / "code" / "fotomedik" / "__init__.py").read_text() == "v2"
    assert install.installed()["build"] == 11 and not install.pending()
    assert not any(p.name.endswith((".old", ".new")) for p in install.app_dir().iterdir())


def test_new_base_comes_when_its_id_changes(tmp_path, monkeypatch, home):
    monkeypatch.setenv("FOTOMEDIK_ASSETS", str(_release(tmp_path, 10, "B1")))
    install.stage(install.Release())
    install.apply_pending()
    monkeypatch.setenv("FOTOMEDIK_ASSETS", str(_release(tmp_path, 12, "B2")))
    assert install.stage(install.Release()) == ["base", "code"]
    install.apply_pending()
    assert install.app_exe().read_bytes() == b"exeB2"


def test_damaged_download_leaves_the_app_alone(tmp_path, monkeypatch, home):
    monkeypatch.setenv("FOTOMEDIK_ASSETS", str(_release(tmp_path, 10, "B1")))
    install.stage(install.Release())
    install.apply_pending()
    a = _release(tmp_path, 13, "B1", "v3")
    info = json.loads((a / install.VERSION).read_text(encoding="utf-8-sig"))
    info["code_sha256"] = "0" * 64
    (a / install.VERSION).write_text(json.dumps(info), encoding="utf-8")
    monkeypatch.setenv("FOTOMEDIK_ASSETS", str(a))
    with pytest.raises(OSError):
        install.stage(install.Release())
    assert not install.pending() and install.installed()["build"] == 10


def test_runtime_kinds_follow_the_graphics_card(monkeypatch):
    import sys

    from fotomedik.ai import cuda, runtime

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(cuda, "nvidia_present", lambda: True)
    assert runtime.candidates() == ["cuda", "directml", "cpu"]
    monkeypatch.setattr(cuda, "nvidia_present", lambda: False)
    assert runtime.candidates() == ["directml", "cpu"]
