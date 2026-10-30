

def test_summary_says_how_the_run_went(tmp_path):
    from fotomedik.report.report import FileReport
    from fotomedik.runner import Result, summary_text

    def res(pct, findings=True):
        rep = FileReport("a.jpg", "a.jpg", 10, 10, pct, findings=[{"kind": "scan_error"}] if findings else [])
        return Result(tmp_path / "a.jpg", tmp_path / "out" / "a.jpg", rep, 1.0, {})

    text = summary_text([res(100.0, False), res(60.0), res(10.0)], 3, 125.0, "", tmp_path / "out", True, errors=2)
    assert "Photos: 3 of 3" in text and "partly recovered: 1" in text and "lost: 1" in text
    assert "Recovered on average: 57 %" in text and "2 min 5 s" in text
    assert "Errors: 2" in text and "FotoMedik-greske.txt" in text and "Enhanced copies made: 0" in text
