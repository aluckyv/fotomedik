# FotoMedik — dizajn (v0.1, jezgro)

Datum: 2026-10-29 · Status: čeka odobrenje · Radni naziv: **FotoMedik** (menja se po želji)

## 1. Šta pravimo i zašto

Opšta aplikacija za **dijagnozu i popravku oštećenih fotografija** (prvo JPEG), namenjena
običnim korisnicima koji su izvukli slike sa pokvarene kartice/diska i dobijaju „cannot open“
ili sivu sliku. Program:

1. **skenira** fajl i kaže šta je tačno pokvareno, koliko i gde,
2. **popravlja** ono što se popraviti može, primenom biblioteke „recepata“,
3. **izveštava pošteno**: šta je vraćeno (procenat slike), šta nije i zašto.

Vrednost nije u čudima nego u dijagnozi i maksimalnom izvlačenju. Postojeći alati ili ćute
ili obećavaju svašta.

Polazni materijal: test folder — fotografije 2005–2017 sa kartice koja umire
(bit-rot po 32 KB klasterima, uništeni headeri, zalepljen tuđ sadržaj, CardRecovery `.~CR`
temp fajlovi sa XOR `0x76` na prvih 512 bajtova).

Redosled: **prvo mehanizam (biblioteka + CLI), pa GUI** na isto jezgro.

## 2. Ciljevi / ne-ciljevi / anti-ciljevi

**Ciljevi v0.1**
- Otvoriti svaki JPEG koji ima makar deo ispravnog skena — nijedan „cannot open“.
- Za svaki fajl izveštaj: tip(ovi) oštećenja, lokacija, procenat vraćene slike, šta je urađeno.
- Recepti za sve tipove oštećenja nađene u test setu (odeljak 6).
- „Učenje primerom“: zdrava slika sa iste kamere kao izvor headera (DQT/DHT/SOF/DRI).
- Batch nad folderom, sa `.~CR` dekodovanjem.
- Izlaz je uvek **standardan JPEG** koji otvara bilo koji viewer, plus `report.json`.

**Ne-ciljevi v0.1** (dolaze kasnije ili nikad)
- GUI (v0.2+). RAW/HEIC/PNG/video (kasnije). Inpainting sivih rupa AI modelom (opcioni
  završni korak, ne u jezgru). Oporavak sa same kartice (to rade ddrescue/PhotoRec).

**Anti-ciljevi**
- Nikad ne prepisivati original — izlaz uvek u zaseban folder.
- Nikad ne prikazati „popravljeno“ bez broja: svaki rezultat nosi procenat i listu rupa.
- Ne „popravljati“ izmišljanjem sadržaja a da to nije jasno označeno kao rekonstrukcija.

## 3. Ograničenja

- Python 3.11+, zavisnosti: `numpy`, `Pillow` (samo za render/verifikaciju), `click` (CLI),
  `pytest`. Bez C ekstenzija u v0.1; ako brute-force bude prespor, `numba` kao opcija.
- Windows 11 primarna platforma; kod mora da radi i na Linuxu (razvoj je u cloud
  okruženju, testovi tamo).
- Privatne slike iz test seta se **ne komituju** (`samples/` je u `.gitignore`).
  U repo idu samo sintetički test fajlovi koje generišemo iz jedne slobodne slike.
- Kod i komentari na engleskom (opšta aplikacija), dokumentacija i poruke korisniku
  dvojezično (sr/en) preko jednostavnog rečnika stringova.

## 4. Razmatrani pristupi

| Pristup | Za | Protiv | Odluka |
|---|---|---|---|
| Osloniti se na libjpeg (Pillow) i samo krpiti headere | brzo | libjpeg ne daje gde je greška, ne može bit-brute-force, staje na prvoj grešci | **ne** |
| Sopstveni JPEG entropijski dekoder (Huffman, baseline + progressive detekcija) | tačna lokacija greške do bita, kontrola po MCU/restart-intervalu, brute-force | treba napisati; čist Python je sporiji | **da** (jezgro) |
| ML model za popravku | zvuči moderno | bit-rot u Huffman kodu je kombinatorni problem, ne statistički | ne (inpainting kasnije, opciono) |

Korisnik je izabrao: biblioteka recepata (mi učimo program) + učenje primerom (referentni
header). Python, GitHub repo + Desktop folder, mešoviti test set.

## 5. Arhitektura (moje odluke, označene ★)

```
fotomedik/
  jpeg/        ★ parser markera, Huffman dekoder sa lokalizacijom greške, pisač fajla
  scan/        detektori → Diagnosis (lista nalaza: tip, ozbiljnost, offset, pouzdanost)
  recipes/     ★ registar recepata; recept = detector() + fix(); redosled po prioritetu
  learn/       ekstrakcija „profila kamere“ iz zdrave slike (header tabele) i presađivanje
  report/      JSON + tekst izveštaj, procenat vraćenih MCU-ova
  cli.py       fotomedik scan | fix | batch | learn
tests/         pytest; sintetička oštećenja iz jedne zdrave slike (ground truth)
samples/       (gitignore) realni primeri (privatni)
docs/superpowers/specs|plans/
```

★ **Dekoder**: baseline Huffman (progressive samo detektujemo i prijavljujemo u v0.1).
Dekodira sken po MCU-ovima, pamti bit-offset svakog MCU-a i svakog restart-intervala,
vraća prvu nevalidnu poziciju (loš Huffman kod, prekoračenje koeficijenata, pogrešan RST).
Bez IDCT u Pythonu — render radi Pillow nad popravljenim fajlom. Ovo drži Python dovoljno
brzim: brute-force dekodira samo lokalni interval, ne ceo fajl.

★ **Mera uspeha popravke**: udeo MCU-ova dekodovanih bez greške + „glatkoća“ (razlika DC
koeficijenata susednih blokova) za izbor između kandidata u brute-force-u.

★ **Nepopravljivi intervali** se zamenjuju minimalno validnim entropijskim podacima
(DC-only blokovi sa bojom suseda), pa fajl ostaje standardan JPEG koji se ceo otvara; rupe se
upisuju u izveštaj i opciono markiraju bojom.

★ **Recept** je mala Python klasa: `name`, `detect(ctx) -> Finding|None`,
`fix(ctx) -> ctx'`, `priority`. Dodavanje novog tipa oštećenja = novi fajl u `recipes/`.

## 6. Recepti v0.1 (iz test seta)

| # | Oštećenje | Detekcija | Popravka |
|---|---|---|---|
| R1 | CardRecovery `.~CR` | ekstenzija + prvih 512 B XOR `0x76` daju `FF D8` | XOR dekodiranje |
| R2 | Izvrnuti bitovi u markerima headera | lanac markera puca; bajt je na Hamming ≤2 od validnog markera | ispravka markera |
| R3 | Pokvarena dužina segmenta | sledeći marker nije na očekivanom mestu, nađen u blizini | ispravka dužine |
| R4 | Bit-flip u skenu, sa RST | dekoder puca unutar intervala, sledeći RST ok | izolacija intervala, brute-force (R5) ili zamena |
| R5 | Retki bit-flip-ovi | dekoder puca; prostor pretrage: ±N bajtova, 1–2 bita, prvo 0→1 (vraćanje izgubljenih jedinica) | kandidat koji produži dekodovanje i prođe meru glatkoće |
| R6 | Tuđ sadržaj / smeće na kraju | statistika `FF xx` po klasteru (≈96 % loših = ne-JPEG) | odsecanje + EOI, prijava procenta |
| R7 | Uništen header, sken čitav | SOI/DQT/DHT/SOF nema ili nevalidni | presađivanje iz referentne slike (`learn`) |
| R8 | Ništa od skena | sken < 1 MCU reda | izvlačenje Exif thumbnail-a kao jedini rezultat |
| R9 | Fajl bez RST, greška rano | nema DRI | brute-force (R5); ako ne uspe, isečak do greške + thumbnail |

## 7. CLI (v0.1)

```
fotomedik scan  <file|folder>                 → dijagnoza, bez pisanja
fotomedik fix   <file|folder> -o <out> [--ref good.jpg] [--mark-holes]
fotomedik learn <good.jpg> -o camera.json     → profil kamere za --ref
fotomedik batch <folder> -o <out>             → fix + zbirni izveštaj (CSV/JSON)
```

## 8. Testiranje

TDD. Iz jedne slobodne zdrave slike generišemo oštećenja sa poznatim odgovorom
(flip bita na offsetu X, odsecanje, XOR 512 B, pokvaren marker) → test zna šta je tačno.
Realni primeri iz `samples/` služe za ručnu proveru i merenje procenta vraćenog, ne za
unit testove.

## 9. Šta ostaje graditelju

Konkretna implementacija Huffman dekodera, pragovi za detektore, format `report.json`,
tačan prostor pretrage u R5 (N bajtova, max bitova) — podešava se na test setu.

## 10. Otvorena pitanja

1. Ime aplikacije (radno: FotoMedik).
2. Da li popravljeni fajl čuva originalni Exif u celosti ili ga čistimo od oštećenih tagova.
