# FotoMedik


Dijagnoza i popravka oštećenih fotografija (JPEG). Program kaže *šta* je pokvareno, popravi
ono što se popraviti može i pošteno prijavi koliko je slike vraćeno. Originali se nikad ne diraju.

Diagnose and repair corrupted photos (JPEG). Tells you *what* is broken, repairs what can be
repaired, and reports honestly how much of the image came back. Originals are never touched.

## The app (Windows)

Download `FotoMedik-windows` from the latest successful run under **Actions** on
GitHub, unzip, and start `FotoMedik.exe` — no Python or other installs needed.
Drop a folder or photos onto the window (or use the buttons), check where the
repaired copies go, press **Repair**. Click any photo in the list to compare
before and after and read what was found and repaired. Originals are never
changed.

From source: `pip install -e . pyside6`, then `python -m fotomedik.gui`.

### Claude (MCP)

**Connect to Claude** (top right) asks nothing and opens no window: it brings FotoMedik's
Claude extension up to date when it is installed (or sees a FotoMedik entry in Claude's
settings), and says so in the status bar. When FotoMedik is not in Claude yet, it writes
`FotoMedik.mcpb` and shows it in Explorer: drag it once into Claude's Settings > Extensions.
The extension starts FotoMedik's MCP server (`FotoMedik.exe --mcp`) with Claude's own
Node.js. Claude can then queue folders for repair on this computer
(`start_job`), follow them (`job_status`, `list_jobs`), stop them (`stop_job`) and
see what the program has (`app_info`). The jobs run one after another in processes
of their own, on this computer's graphics card, and go on when Claude is closed;
`skip_done` continues a stopped job. Originals are never changed: the output folder
is never the source, inside it or around it, never one of the folders and drives
listed in the dialog, and nothing is deleted. Without the window:
`FotoMedik.exe --cli batch <folder> -o <out>` (the command line below).

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

```
fotomedik fix   D:\slike -o out --reference D:\ai_verzije   # referentne slike vode popravku
fotomedik compare out\IMG_0031.JPG original.jpg --picture mapa.jpg --runs
```

**Referentna slika** (`--reference`, u prozoru dugme „Reference photo…" ili polje „References"):
približna kopija iste fotografije — AI rekonstrukcija, skenirana fotografija, kopija iz albuma —
zamenjuje ugrađenu sličicu 160×120. Uparuje se po imenu ili sadržaju, poravnava se lokalno sa
dekodiranim pikselima i kalibriše po boji; po njoj se slažu komadi i popunjavaju rupe (njenim
pikselima, u punoj rezoluciji), a dekodirani delovi ostaju bit-po-bit originalni. Rezultat:
`<ime>_ref.jpg` pored obične popravke.

**Slični snimci** (`--twins`, u prozoru „Colours from similar shots", podrazumevano isključeno):
slika bez upotrebljive sličice uzima boje rupa sa snimka iste scene napravljenog do 30 s ranije
ili kasnije (vreme iz EXIF-a, imena fajla ili datuma koji je ostao u oštećenom headeru). Koristi
se sličica tog snimka, a ako je nema, samo njegovi pravi delovi; poravnava se sa dekodiranim
delovima, a svaki deo gde se scena promenila (ljudi se pomere) izostavlja se. Daje samo boju i
svetlinu, bez detalja, i ne utiče na slaganje komada.

**Poređenje** (`compare`, dugme „Compare with the original…"): mera popravke prema netaknutoj
kopiji — greška po bloku, koliko je dekodiranog tačno, koliko su rupe daleko od istine, mapa greške.
Kopija se prvo poravna glatkim poljem pomeraja (AI-restaurirane kopije „plivaju“ i po desetak piksela),
pa se tek onda meri.

## Šta ume (recepti)

| | oštećenje | popravka |
|---|---|---|
| R1 | CardRecovery `.~CR` temp fajl | XOR dekodiranje prvih 512 B |
| R2/R3 | izvrnuti bitovi u markerima, dužinama, tabelama | heuristička popravka headera |
| R7 | uništen header | presađivanje iz profila kamere (zdrava slika) |
| R6 | tuđ sadržaj na kraju fajla | odsecanje + EOI |
| R5 | izvrnuti bitovi u podacima slike | pogađanje bita dekodovanjem do restart markera |
| R9 | nepopravljivi intervali | popunjavanje bojom suseda, fajl se otvara ceo |
| R11 | netaknuti komadi usred oštećenja, bez ikakve reference | komad sam kaže kolonu: pogrešno složen, kraj svakog reda slike pada uz početak sledećeg i pikseli se tu ne slažu (šav) |
| R14 | dekodirano smeće (šare, kockice, crno-bele mrlje) | spektar luma i hroma bloka + zasićenje, mereno prema statistici same slike; smeće postaje rupa i popunjava se nanovo |
| R8 | ništa od slike | ugrađeni Exif thumbnail |

Dizajn: `docs/superpowers/specs/`, plan i rezultati: `docs/superpowers/plans/`.
Testovi: `pytest` (sintetička oštećenja sa poznatim odgovorom; `samples/` je privatno, van Git-a).
