# FotoMedik

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
