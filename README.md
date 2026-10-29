# FotoMedik


## The app (Windows)

Download `FotoMedik-windows` from the latest successful run under **Actions** on
GitHub, unzip, and start `FotoMedik.exe` — no Python or other installs needed.
Drop a folder or photos onto the window (or use the buttons), check where the
repaired copies go, press **Repair**. Click any photo in the list to compare
before and after and read what was found and repaired. Originals are never
changed.

From source: `pip install -e . pyside6`, then `python -m fotomedik.gui`.
Dijagnoza i popravka oštećenih fotografija (JPEG). Program kaže *šta* je pokvareno, popravi
ono što se popraviti može i pošteno prijavi koliko je slike vraćeno. Originali se nikad ne diraju.

Diagnose and repair corrupted photos (JPEG). Tells you *what* is broken, repairs what can be
repaired, and reports honestly how much of the image came back. Originals are never touched.

## Instalacija (Windows, Python 3.11+)

```
git clone https://github.com/aluckyv/fotomedik
cd fotomedik
py -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

### Brzi dekoder (preporučeno)

Jezgro dekodera postoji i kao C kernel (`fotomedik/jpeg/_fastdec.c`), ~15× brže od čistog
Pythona — bez njega sve radi, samo sporije. Kompajliranje: `python build_fastdec.py`
(treba gcc/clang/MSVC). Na Windows-u bez kompajlera: skini `_fastdec.dll` iz
GitHub Actions artefakata (workflow *build*) i stavi ga u `fotomedik/jpeg/`.

## Upotreba

```
fotomedik scan  D:\slike                 # samo dijagnoza, ništa se ne piše
fotomedik fix   slika.jpg -o D:\popravljeno
fotomedik batch D:\slike -o D:\popravljeno   # ceo folder + summary.csv
fotomedik learn zdrava.jpg -o kamera.json    # profil kamere za --ref
fotomedik fix   pokvarena.jpg -o out --ref kamera.json
```

Opcije: `--mark-holes` (rupe magenta umesto stapanja), `--keep-damaged` (zadrži i sumnjive
blokove), `--time-budget 120` (sekundi po fajlu za pogađanje bitova), `--lang sr|en`.

`batch` sam pravi referentne profile iz zdravih slika u folderu i koristi ih za one sa
uništenim headerom. Uz svaki izlazni fajl ide `<ime>.report.json`.

## Šta ume (recepti)

| | oštećenje | popravka |
|---|---|---|
| R1 | CardRecovery `.~CR` temp fajl | XOR dekodiranje prvih 512 B |
| R2/R3 | izvrnuti bitovi u markerima, dužinama, tabelama | heuristička popravka headera |
| R7 | uništen header | presađivanje iz profila kamere (zdrava slika) |
| R6 | tuđ sadržaj na kraju fajla | odsecanje + EOI |
| R5 | izvrnuti bitovi u podacima slike | pogađanje bita dekodovanjem do restart markera |
| R9 | nepopravljivi intervali | popunjavanje bojom suseda, fajl se otvara ceo |
| R8 | ništa od slike | ugrađeni Exif thumbnail |

Dizajn: `docs/superpowers/specs/`, plan i rezultati: `docs/superpowers/plans/`.
Testovi: `pytest` (sintetička oštećenja sa poznatim odgovorom; `samples/` je privatno, van Git-a).
