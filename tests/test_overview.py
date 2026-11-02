"""The quick look at a folder before the repair (overview.py)."""

from pathlib import Path

from synth import flip_bits, make_image, make_jpeg
from test_dcfield import _with_thumbnail

from fotomedik import jobs
from fotomedik.overview import Overview, PhotoFacts, overview


def _zeroed(data: bytes) -> bytes:
    """A stretch of the scan overwritten: holes in the picture."""
    out = bytearray(data)
    a = len(out) * 2 // 3
    out[a : a + 400] = b"\x00" * 400
    return bytes(out)


def _folder(tmp_path) -> Path:
    img = make_image()
    plain = make_jpeg(img, restart_rows=1)
    thumbed = _with_thumbnail(plain, img, size=(128, 64))
    # a black-and-white photo without a thumbnail: its scan header differs from the set's
    # thumbnails, so none is "rescued" from it (a colour one's would match: synthetic files
    # share Pillow's tables, real cameras' thumbnails and main scans do not)
    grey = make_jpeg(img.convert("L"), restart_rows=1)
    src = tmp_path / "src"
    src.mkdir()
    (src / "healthy.jpg").write_bytes(thumbed)
    (src / "holes_thumb.jpg").write_bytes(_zeroed(thumbed))
    (src / "holes_plain.jpg").write_bytes(_zeroed(grey))
    (src / "notes.txt").write_text("not a photo")
    return src


def test_overview_tells_the_damage_and_the_thumbnails_apart(tmp_path):
    src = _folder(tmp_path)
    files = [(p, Path(p.name)) for p in sorted(src.glob("*.jpg"))]
    ov = overview(files, jobs=1, faces=False)
    by = {p.name: p for p in ov.photos}
    assert by["healthy.jpg"].state == "healthy" and by["healthy.jpg"].thumb == "whole"
    assert by["holes_thumb.jpg"].damaged and by["holes_thumb.jpg"].thumb == "whole"
    assert by["holes_plain.jpg"].damaged and by["holes_plain.jpg"].thumb == "none"
    assert all(p.faces is None for p in ov.photos)  # not looked for
    preset, why = ov.suggest()
    assert preset == "standard" and why[0].startswith("2 of 3 photos damaged")
    assert any("1 with a whole thumbnail" in w for w in why) and any("1 without a thumbnail" in w for w in why)
    assert "Suggested: Standard." in ov.text()


def test_overview_without_damage_suggests_the_repair_only():
    ov = Overview([PhotoFacts("a.jpg", state="healthy", thumb="whole"), PhotoFacts("b.jpg", state="healthy")])
    assert ov.suggest() == ("faithful", ["no damaged photo found: only the repaired copies are made"])
    assert Overview([]).suggest()[0] == "faithful"


def test_look_at_folder_gives_claude_the_numbers(tmp_path, monkeypatch):
    monkeypatch.setenv("FOTOMEDIK_HOME", str(tmp_path / "home"))
    src = _folder(tmp_path)
    out = jobs.look_at_folder(str(src), photos=True)
    assert out["photos"] == 3 and out["damaged"] == 2 and out["thumbnail_whole"] == 1 and out["thumbnail_none"] == 1
    assert out["suggested_preset"] == "standard" and out["suggested_steps"] == jobs.DEFAULT_STEPS
    assert {p["photo"] for p in out["list"]} == {"healthy.jpg", "holes_thumb.jpg", "holes_plain.jpg"}
