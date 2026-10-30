"""Finding the NVIDIA CUDA libraries already on the computer."""

from pathlib import Path

from fotomedik.ai import cuda


def test_libraries_are_found_across_folders(tmp_path):
    a, b = tmp_path / "toolkit", tmp_path / "cudnn"
    a.mkdir()
    b.mkdir()
    for n in ("cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll", "cufft64_11.dll"):
        (a / n).write_bytes(b"x")
    (b / "cudnn64_9.dll").write_bytes(b"x")
    found = cuda.find([tmp_path / "nothing", a, b])
    assert set(found) == set(cuda.NEEDED)
    assert found["cudnn64_9.dll"] == b / "cudnn64_9.dll" and found["cudart64_12.dll"].parent == a


def test_missing_libraries_are_named(tmp_path):
    (tmp_path / "cudart64_12.dll").write_bytes(b"x")
    found = cuda.find([tmp_path])
    missing = [n for n in cuda.NEEDED if n not in found]
    assert "cudnn64_9.dll" in missing and "cudart64_12.dll" not in missing
    assert {cuda.NEEDED[n][0] for n in missing} == {"nvidia-cublas-cu12", "nvidia-cufft-cu12", "nvidia-cudnn-cu12"}


def test_search_includes_the_toolkit_and_python_packages(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_PATH", str(tmp_path / "CUDA" / "v12.4"))
    dirs = [str(d) for d in cuda.search_dirs()]
    assert str(Path(tmp_path / "CUDA" / "v12.4" / "bin")) in dirs
    assert dirs[0] == str(cuda.own_dir())


def test_no_nvidia_card_means_no_cuda(monkeypatch):
    monkeypatch.setattr(cuda, "_ready", None)
    monkeypatch.setattr(cuda, "nvidia_present", lambda: False)
    assert cuda.prepare() is False
