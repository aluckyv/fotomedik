# Plan: FotoMedik jezgro v0.1

**Cilj:** Python biblioteka + CLI koja dijagnostikuje oštećene JPEG fajlove, popravlja ih receptima R1–R9 i daje izveštaj sa procentom vraćene slike.

**Arhitektura:** `fotomedik.jpeg` (parser, Huffman dekoder sa bit-pozicijama, mini-enkoder za rupe, pisač) je temelj; `scan` pravi `Diagnosis`; `recipes` su mali detect/fix moduli u registru koje pipeline primenjuje po prioritetu nad `Context`-om; `learn` pravi profil kamere iz zdrave slike; `report` i `cli` su tanki slojevi.

**Stack:** Python 3.11+, numpy, Pillow (render/verifikacija), click, pytest. Bez C ekstenzija.

**Spec:** `docs/superpowers/specs/2026-10-29-fotomedik-design.md`

**Globalna ograničenja (iz speca):**
- Nikad ne prepisivati original — izlaz uvek u zaseban folder.
- Svaki rezultat nosi procenat vraćenih MCU-ova i listu rupa.
- `samples/` je u `.gitignore`; u repo idu samo sintetički testovi.
- Kod i komentari na engleskom; poruke korisniku sr/en.
- Izlaz je uvek standardan JPEG koji otvara bilo koji viewer.

**Fokus za review (ono što spec podrazumeva, a nijedan test direktno ne pokriva):**
- Brute-force ne sme da „popravi" interval u nešto što dekodira, a vizuelno je smeće (mera glatkoće mora da odbije takve kandidate).
- Resinhronizacija na RST posle greške ne sme da pomeri sliku (pogrešan indeks intervala).
- Dekodovani podaci iz „smeće" klastera ne smeju da se računaju kao vraćeni.
- Performanse: fajl od 3 MB sa 500 grešaka mora da prođe za < 60 s na Ryzen 7.

## Struktura fajlova

| Fajl | Odgovornost |
|---|---|
| `pyproject.toml`, `.gitignore`, `README.md` | paket `fotomedik`, CLI entry point, ignorisanje `samples/` |
| `fotomedik/jpeg/markers.py` | `Segment`, `parse_segments(data) -> ParseResult` (tolerantan prolaz kroz markere) |
| `fotomedik/jpeg/tables.py` | `QuantTable`, `HuffmanTable`, `FrameInfo`, `ScanInfo`; parseri DQT/DHT/SOF/SOS/DRI; standardne Annex K tabele |
| `fotomedik/jpeg/bitreader.py` | `BitReader` — bitovi iz entropijskog toka, byte-stuffing, detekcija markera, pozicija u bitovima |
| `fotomedik/jpeg/decoder.py` | `decode_scan(...) -> ScanResult` — MCU po MCU, intervali, prva greška, DC vrednosti, resync na RST |
| `fotomedik/jpeg/encoder.py` | `BitWriter`, `encode_dc_only_mcus(...)` — minimalni validni entropijski podaci za rupe |
| `fotomedik/jpeg/writer.py` | `assemble(segments, scan_bytes) -> bytes` |
| `fotomedik/scan/diagnosis.py` | `Finding`, `Diagnosis`, `to_dict()` |
| `fotomedik/scan/garbage.py` | `cluster_stats`, `classify_clusters` (JPEG/garbage/clean) |
| `fotomedik/scan/scanner.py` | `scan_bytes(data, name) -> Diagnosis` — poziva sve detektore |
| `fotomedik/recipes/base.py` | `Recipe`, `Context`, `LogEntry` |
| `fotomedik/recipes/registry.py` | `register`, `all_recipes()`, `run_pipeline(ctx) -> Context` |
| `fotomedik/recipes/r1_cardrecovery.py` | XOR `0x76` prvih 512 B |
| `fotomedik/recipes/r2_header.py` | R2+R3: markeri i dužine segmenata, tabele vs. standardne |
| `fotomedik/recipes/r6_garbage.py` | odsecanje smeća, označavanje MCU-ova |
| `fotomedik/recipes/r5_bitflip.py` | R4+R5+R9: brute-force bitova po intervalu, greedy, budžet |
| `fotomedik/recipes/r9_holes.py` | zamena nepopravljivih intervala DC-only blokovima |
| `fotomedik/recipes/r7_reference.py` | presađivanje header-a iz `CameraProfile` |
| `fotomedik/recipes/r8_thumbnail.py` | Exif thumbnail kao rezervni izlaz |
| `fotomedik/learn/profile.py` | `CameraProfile.from_jpeg(data)`, `to_json/from_json` |
| `fotomedik/report/report.py` | `FileReport`, `write_json`, `format_text(lang)` |
| `fotomedik/cli.py` | `scan`, `fix`, `learn`, `batch` |
| `tests/synth.py` | generator zdrave slike + oštećenja sa poznatim odgovorom |
| `tests/conftest.py` | fixture-i: `healthy_jpeg`, `healthy_jpeg_rst`, `healthy_progressive` |
| `tests/test_*.py` | po jedan fajl po modulu |

## Zadaci

### T1 — Skelet projekta i sintetički fixture-i
1. `pyproject.toml` (name `fotomedik`, deps numpy/Pillow/click, `[project.scripts] fotomedik = "fotomedik.cli:main"`), `.gitignore` (`samples/`, `out/`, `.venv`, `__pycache__`), `README.md` (dve rečenice).
2. `tests/synth.py`: `make_image(w=256,h=192) -> PIL.Image` (gradijenti + geometrija, deterministički seed), `make_jpeg(img, restart_rows=None, progressive=False, quality=90) -> bytes`, `flip_bits(data, byte_off, mask) -> bytes`, `truncate(data, n)`, `xor_prefix(data, key=0x76, n=512)`, `append_garbage(data, n, seed)`.
3. Test `tests/test_synth.py`: `make_jpeg` daje bajtove koji počinju `FFD8`, završavaju `FFD9`, Pillow ih otvara; sa `restart_rows=1` sadrži `FFDD` i bar jedan `FFD0`. RED → GREEN → commit.

### T2 — Parser markera
1. Test: `parse_segments(healthy)` vraća listu segmenata sa `marker`, `offset`, `length`, `payload`; poslednji je SOS; `scan_start` tačan; `eoi_offset` tačan.
2. Test: posle `flip_bits` na bajtu markera, `parse_segments` ne baca izuzetak nego vraća `errors=[ParseError(offset, reason)]` i staje.
3. Implementacija `markers.py` (`Segment(marker:int, offset:int, length:int, payload:bytes)`, `ParseResult(segments, scan_start, eoi_offset, errors)`). Commit.

### T3 — Tabele
1. Testovi `test_tables.py`: `parse_dqt` daje 64 vrednosti u zig-zag redu; `parse_dht` daje `HuffmanTable.lookup[(length, code)] -> symbol` i `max_code`, `min_code`, `val_ptr` nizove; `parse_sof` daje `FrameInfo(width,height,precision,components[(id,h,v,tq)], progressive)`; `parse_sos` daje `ScanInfo(components[(id,td,ta)], ss,se,ah,al)`; `parse_dri` daje interval.
2. `STANDARD_DC_LUMA`, `STANDARD_AC_LUMA`, `STANDARD_DC_CHROMA`, `STANDARD_AC_CHROMA` (Annex K) + test da se iz njih gradi tabela sa 12/162 simbola.
3. Implementacija. Commit.

### T4 — BitReader
1. Testovi: čita bitove MSB-first; `FF 00` daje bajt `FF`; na `FF D0` vraća `MarkerHit(marker, byte_offset)` umesto bitova; `bit_pos` tačan; `align_to_byte()`.
2. Implementacija. Commit.

### T5 — Dekoder (srce)
1. Test: `decode_scan(healthy)` → `ScanResult.error is None`, `len(mcus) == mcu_count(frame)`, svaki `MCUInfo.bit_start` rastući, `intervals==[1 interval]`.
2. Test sa `healthy_jpeg_rst`: `len(intervals) == broj redova MCU`, svaki interval `ok=True`, `interval.end_marker` je ispravan RSTn.
3. Test: `flip_bits` u sredini skena bez RST → `error.mcu_index` ≥ MCU koji sadrži taj bajt i ≤ +2, `error.byte_offset` u ±16 B, `error.reason in {"bad_huffman","coef_overflow","dc_range","unexpected_marker","eof"}`.
4. Test sa RST + flip u intervalu k → interval k `ok=False`, intervali k+1.. `ok=True` (resync), `mcus` indeksi tačni (nema pomeranja).
5. Test: `ScanResult.dc[component][block_row][block_col]` postoji za dekodovane blokove (numpy int32, NaN/None za nedekodovane).
6. Implementacija `decoder.py`: `decode_scan(data, parse: ParseResult, tables, start_byte=None, end_byte=None, max_mcus=None) -> ScanResult`. DC range provera: `abs(dc * q[0]) > 1024 + 8*q[0]` → `dc_range`. Resync: posle greške traži sledeći `FF Dx`, indeks intervala bira po `(x+1) mod 8` i proceni bajtova/intervalu.
7. Performanse: test da 1 MP sken dekodira < 5 s (skip ako sporo, ali meri). Commit.

### T6 — Dijagnoza i smeće
1. Test `cluster_stats(data, 32768)` na `append_garbage(healthy, 65536)` → poslednja 2 klastera `kind=="garbage"` (bad-FF ratio > 0.5 ili entropija ~8 bita), ostali `jpeg`.
2. Test `scan_bytes` na zdravom → `Diagnosis.findings == []`, `score == 1.0`; na `.~CR` → finding `cardrecovery_xor`; na flip markera → `bad_marker`; na flip u skenu → `scan_error` sa pozicijom; na odsečenom → `truncated`.
3. Implementacija `diagnosis.py`, `garbage.py`, `scanner.py`. Commit.

### T7 — Okvir za recepte + R1
1. Test: `register` dodaje recept; `run_pipeline(ctx)` primenjuje recepte po `priority`, svaki `fix` upisuje `LogEntry(recipe, message, before, after)`; recept čiji `detect` vrati `None` se preskače.
2. Test R1: `xor_prefix(healthy)` → posle pipeline-a `ctx.data == healthy`.
3. Implementacija `base.py`, `registry.py`, `r1_cardrecovery.py`. Commit.

### T8 — R2/R3 header
1. Test: flip 1 bita u bajtu markera `DQT` → popravljeno, `parse_segments` čist.
2. Test: flip bita u polju dužine `DHT` → popravljeno (sledeći marker nađen u ±64 B).
3. Test: flip bita u DHT counts tako da suma > 256 → zamena standardnom tabelom ako je Hamming ≤ 4, log kaže `replaced_with_standard`.
4. Implementacija `r2_header.py`. Commit.

### T9 — R6 smeće + R9 rupe (enkoder)
1. Test enkodera: `encode_dc_only_mcus(frame, scan, tables, n_mcus, dc_values)` daje bajtove koje `decode_scan` dekodira bez greške i sa očekivanim DC vrednostima.
2. Test R6: `append_garbage` → posle pipeline-a fajl završava `FFD9`, `report.garbage_bytes == n`.
3. Test R9: interval sa `ok=False` (bez popravke) → zamenjen DC-only blokovima, ceo fajl dekodira bez greške, `report.holes` sadrži taj interval, Pillow otvara sliku pune visine.
4. Implementacija `encoder.py`, `writer.py`, `r6_garbage.py`, `r9_holes.py`. Commit.

### T10 — R5 brute-force bitova
1. Test: `healthy_jpeg_rst` + 1 flip u intervalu 3 → posle R5 ceo fajl dekodira čisto i `ctx.data == healthy` (tačan bit vraćen).
2. Test: 2 flipa u istom intervalu (različiti bajtovi) → oba vraćena (greedy iteracija).
3. Test: bez RST, 1 flip → vraćen.
4. Test: 20 flipova u jednom intervalu → R5 odustaje (`budget_exhausted`), interval ostaje za R9.
5. Test glatkoće: kandidat koji dekodira do kraja intervala ali sa DC skokom > prag se rangira ispod kandidata sa manjim skokom (konstruisan slučaj).
6. Implementacija `r5_bitflip.py`: `W=48` bajtova unazad, 1-bit pa 2-bit u istom bajtu, redosled 0→1 prvo; `MAX_ERRORS_PER_INTERVAL=8`; skor = (stigao_do_RST, broj_MCU, -DC_skok). Commit.

### T11 — R7 referenca + R8 thumbnail
1. Test `CameraProfile.from_jpeg(healthy)` sadrži DQT/DHT/SOF-sampling/DRI; `to_json`/`from_json` round-trip.
2. Test R7: obriši (nulama) DQT+DHT u kopiji → sa `ctx.ref=profile` pipeline vraća fajl koji dekodira čisto i jednak je originalu u skenu.
3. Test R8: sken potpuno uništen (random) ali Exif thumbnail prisutan (fixture sa ugrađenim thumbnail-om) → `report.fallback == "thumbnail"`, izlaz je validan JPEG dimenzija thumbnail-a.
4. Implementacija. Commit.

### T12 — Izveštaj i CLI
1. Test `FileReport` iz `Context`: `recovered_pct`, `holes`, `findings`, `log`; `format_text("sr")` i `("en")`.
2. Test CLI preko `click.testing.CliRunner`: `scan` ispisuje nalaze i ne piše fajlove; `fix -o out` pravi `out/<ime>.jpg` + `out/<ime>.report.json` i **ne dira** ulaz; `learn` pravi profil; `batch` pravi `summary.csv`.
3. Implementacija `report.py`, `cli.py`. Commit.

### T13 — Realni test set (ručna provera, nije unit test)
1. Stage 10 fajlova u `samples/` (3 Meteori, 2 Krit, 2 Češka, 1 Berlin, 2 `.~CR`).
2. `fotomedik batch samples -o out`; pregledati `summary.csv` i kontaktnu tablu slika.
3. Zabeležiti procenat po fajlu u `docs/superpowers/plans/results-v0.1.md`; ono što ne radi → novi recepti ili pragovi (novi plan).
