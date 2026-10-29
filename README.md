# FotoMedik

Dijagnoza i popravka oštećenih fotografija (JPEG). Program kaže *šta* je pokvareno, popravi
ono što se popraviti može i pošteno prijavi koliko je slike vraćeno.

Diagnose and repair corrupted photos (JPEG). Tells you *what* is broken, repairs what can be
repaired, and reports honestly how much of the image came back.

```
pip install -e ".[dev]"
fotomedik scan  photo.jpg
fotomedik fix   photo.jpg -o out/
fotomedik batch folder/  -o out/
```

Dizajn: `docs/superpowers/specs/`, plan: `docs/superpowers/plans/`.
