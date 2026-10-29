# FotoMedik v0.1 — rezultati (29. 10. 2026)

Test skup: 12 fotografija iz `the test folder` (kartica koja je umirala), plus 25 iz
`2008. BERLIN` za proveru dijagnoze. Mere su iz Linux kontejnera sa 2 jezgra.

## Brzina

| | ranije | sada |
|---|---|---|
| batch 12 slika | 15,5 s | **7,1 s** (2 paralelna posla) |
| najsporija slika (krit_2069) | ~11 s | 4,4 s |
| 25 slika Berlin | — | 1,8 s |

Na 16-nitnom računaru batch 12 slika traje očekivano 2–3 s, jer ga ograničava
najsporija pojedinačna slika.

## Vraćeno (udeo stvarnih podataka slike)

| fajl | ranije | sada | napomena |
|---|---|---|---|
| meteori_3115 | 95,7 % | 96,3 % | trake sa junk blokovima uklonjene |
| meteori_3116 | 51,0 % | 51,7 % | rupe prate sličicu |
| meteori_3118 | 32,3 % | 40,5 % | sličica oštećena, rupe interpolirane |
| krit_2069 | 32,6 % | 36,6 % | bez obojenih pruga |
| krit_2076 | 18,9 % | 21,1 % | donje 2/3 iz sličice umesto pruga |
| cr_0870 (CardRecovery) | 4,6 % | 22,8 % | ceo kadar vidljiv (vodi ga sličica) |
| ceska_0810 / 0811 | 1,0 / 0,3 % | 1,6 / 0,3 % | praktično sve iz sličice |
| berlin_0022/0023 | — | — | **u fajlu nema fotografije** |
| berlin_0021, cr_0006 | — | — | header previše oštećen, bez referentne slike |

Procenat broji samo stvarne podatke. Deo koji je rekonstruisan iz sličice ili
interpolacijom nije uračunat i izveštaj ga posebno navodi.

## Šta je urađeno u ovom krugu

1. **Rep posle oštećenja (C, `tail_flow`).** Sve moguće početne pozicije
   dekodiranja čine graf granica MCU-a. Program prati put u koji se sliva najviše
   startova, unazad od restart markera. Pronalazi duže i čistije repove (Meteori,
   interval 44: 167 od 180 MCU-a, ranije 128 sa junk početkom).
2. **Boja vraćenih repova.** Pomeraj DC vrednosti računa se prema redovima
   iznad i ispod, interpoliranim iz najbližih čistih redova. Repovi se postavljaju u
   krugovima, pa se i repovi bez čistog suseda oslanjaju na već postavljene.
3. **Tihi gubitak sinhronizacije.** Posle neotkrivene desinhronizacije DC
   prediktor ostaje pokvaren. Binarna segmentacija (change point) niza anomalija
   nalazi trenutak skoka, a zadržani deo intervala se seče pre njega.
4. **Sličica (Exif thumbnail) kao vodič.** Sličica se preslikava na mrežu
   blokova i kalibriše prema čistim blokovima (r = 0,93–0,99; pojačanje 0,98–1,1).
   Referenca je DC sličice plus njena greška na najbližim čistim redovima, koja
   bledi sa udaljenošću. Rupe dobijaju tačne boje i grube oblike umesto
   vertikalnih pruga.
5. **Dijagnoza.**
   - „Prepisan" fajl (ponavljajući zapis sa sopstvenom putanjom u UTF-16, koji se
     sabija na 1 %) dobija izveštaj, ali se ne pravi beskorisna kopija. Tako je
     19 od 25 fajlova u folderu Berlin.
   - Oštećen SOI (≤ 3 bita) prijavljuje se kao greška headera.

## Ograničenja

- Rupe su DC-only blokovi 8×8. Na 100 % uvećanja deluju blokovski, a sa sličicom
  daju mutnu sliku.
- Slike bez restart markera (ceska, berlin) sa gustim oštećenjem nemaju šta da
  se vrati osim početka. Slika je tada skoro cela iz sličice.
- Header uništen bez zdrave slike istog aparata u skupu (berlin_0021, cr_0006)
  ostaje nepopravljen.

## Moguće sledeće stavke

- Glatke rupe: niske AC frekvencije iz sličice (DCT po bloku + enkoder za AC
  koeficijente u C-u), umesto ravnih blokova.
- Pun batch nad `the test folder` u `a separate output folder`. Zdrave slike istog aparata
  iz drugih foldera mogu da posluže kao referentni header.
- GUI (PySide6).
