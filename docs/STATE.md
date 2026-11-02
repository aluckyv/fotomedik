# FotoMedik: state of the project

Read this first in a new session. It says how the work is done, what the program does today,
what was tried and did not work, and what is open. Keep it up to date when something lands.
(The repository is public: no personal photos, folder names or paths in here.)

Last update: 2026-11-02, build 122.

## How the work is done

- **Ask before every change** (code, files, settings) and wait for the OK. Before choosing a
  way to solve something, offer the options and let the owner pick; recommend one.
- Replies in Serbian, short. Commit messages, code, comments and docs in English.
- **Originals are never written.** Output always goes to another folder. Test photos are
  copied, never used in place; drives the owner marks protected are never written to.
- Photos never go into the repository (`samples/`, `samples_rest/` stay out, see .gitignore).
- Destructive git operations (force push, history rewrites, deleting branches) are left to the owner.
- Work in progress goes to a branch; `main` only after the owner's OK. Every push to `main`
  makes a numbered build on GitHub Actions.
- Before pushing to `main`, run the tests the way CI does, **also without OpenCV** (the test
  jobs do not install it):

      python -m pytest -q
      python -c "import sys; sys.modules['cv2'] = None; import pytest; sys.exit(pytest.main(['-q']))"

- Check a new method on photos that actually have the problem, and look at the result,
  before running the bench on it.
- Commit trailer lines are given by the session (Co-Authored-By / Claude-Session).

## Layout

| path | what |
|---|---|
| `fotomedik/jpeg/` | tolerant JPEG parsing: markers, tables, bit reader, entropy decoder (C kernel `_fastdec.c` via ctypes, identical Python fallback), encoder/writer |
| `fotomedik/scan/` | diagnosis: findings per file (garbage, overwritten, foreign sectors, damaged header) |
| `fotomedik/recipes/` | the repair recipes (table below), `registry.py` runs them in order |
| `fotomedik/learn/` | camera profiles from healthy photos, header vote (`consensus.py`) |
| `fotomedik/reference.py`, `twins.py` | a reference picture or a similar shot guides the repair |
| `fotomedik/ai/` | the enhanced copy (`run.py` Enhancer, `enhance.py` steps, `puzzle.py` jigsaw, `burst.py`, `faces.py`, `diffusion.py`, `models.py`, `cuda.py`, `runtime.py`) |
| `fotomedik/train/`, `scripts/trening/` | fine-tuning FBCNN on the program's own repairs (runs on the owner's GPU) |
| `fotomedik/gui/` | PySide6 desktop app |
| `fotomedik/cli.py` | `scan`, `fix`, `batch`, `learn`, `compare` |
| `fotomedik/mcp_server.py`, `jobs.py`, `claude_connect.py`, `entry.py` | FotoMedik for Claude (MCP) |
| `fotomedik/install.py` | folder app, self-update from GitHub |
| `scripts/bench.py` | measuring ground (below) |
| `scripts/mcp_smoke.py`, `ext_smoke.py` | CI checks of the packaged MCP server and the extension |
| `docs/superpowers/` | the first design, plan and results v0.1 (2026-10-29) |

## The repair (repaired copy, `<name>.jpg`)

| | damage | repair |
|---|---|---|
| R1 | CardRecovery `.~CR` temp file | XOR decode of the first 512 B |
| R2/R3 | flipped bits in markers, lengths, tables | header repair |
| R4 | foreign (blank) card sectors inside the scan | cut out |
| R5 | flipped bits in the image data | bit guessing between restart markers, kept only when the thumbnail agrees |
| R6 | foreign data after the image | cut + EOI |
| R7 | destroyed header | transplant from a camera profile; header voted bit by bit from every photo of the set |
| R8 | nothing of the image | the Exif thumbnail |
| R9 | lost intervals | holes filled (thumbnail-guided DC field, smooth surface, not flat squares) |
| R10 | intact tail after a rotted stretch | tail scan in C, cut where colours step away from the thumbnail |
| R11 | intact stretches in the middle of damage | placed by the thumbnail, or by their own seam without one |
| R12 | holes | smooth surface (`surface()`); tails after a hole set to the level of their seams (`tail_shifts`, see below) |
| R13 | orientation | portrait photos upright |
| R14 | decoded garbage (patterns, saturated patches, another photo's data) | becomes a hole and is filled again |

Also: damaged thumbnails are rescued (header borrowed from a healthy thumbnail of the same
camera, `dcfield._rescue_thumbnail`); the jigsaw puts the pieces of photos with restart
markers back by the thumbnail (`ai/puzzle.py`, used as the `puzzle` step).

Tail levels (R12, build 122): a run decoded after a hole in the same interval has a DC level
that is only R10's estimate. `tail_shifts` sets every such run to the median of the pixel
differences across its seams with real blocks (L1, the principle of Li, Karrenbauer, Saupe,
Kuo 2011; one unknown per run, all runs solved together by coordinate descent), only when the
seams above and below agree and the blocks stay possible colours; the shift goes into the
hole's last DC, the run's bits are copied as they are. Small gain: on the bench DC-type
errors 3358 -> 3213 blocks; most of the rest lie in runs with holes above and below.

Real damage in the owner's photos is mostly *dense rot*: about 30 % of the card sectors are
someone else's data. Bit guessing cannot bring that back; it is skipped on dense damage.

## The enhanced copy (`<name>_enhanced.jpg`)

Steps (`ai/run.py STEPS`): `puzzle`, `burst`, `deblock`, `ai`, `thumbsr`, `diffusion`, `patch`,
`faces`. The default for jobs is `puzzle,burst,deblock,thumbsr` (`patch` is not on by default yet).

- `puzzle`: jigsaw by the thumbnail (or a rescued thumbnail as a map); junk blocks of a piece
  (far off the thumbnail's colour, `puzzle.BLOCK_OFF = 40`) are left out. Build 122: whole
  regions of rows the repair put 8, 16, 24 rows off are moved first (`row_shifts`,
  `shift_regions`): rows compared with the thumbnail at shifts of 8, clear rows decided, rows
  between two that moved alike move alike, a lone decision not followed. Before, moved pieces
  and unmoved repair rows (a sky fits anywhere) interleaved into stripes (100_0223 lost 24
  rows at the top). Bench: 16 of 17 moved rows right; DSC_4537 psnr 28.86 -> 29.10.
- `patch` (build 122, off by default): holes given the photo's own texture like Content-Aware
  Fill (`ai/patchfill.py`): PatchMatch on a ~1000 px copy matched by the coarse look, voting
  for coherence, only clipped fine detail carried onto the fill's coarse picture, none on
  edges; faces, date stamp and skin neither textured nor taken from. Bench detail 0.66 -> 0.87
  and 0.56 -> 0.84 of the truth's; psnr -0.3/-0.5, ssim -0.02/-0.035 (they prefer smooth).
- `burst`: colours from similar shots of the set.
- `deblock`: FBCNN JPEG clean-up; a fine-tuned and a guided variant (guided sees the thumbnail
  and the holes). The fine-tuned one is used only on photos of the size it learnt on.
- `ai`: LaMa heals holes at full size. `thumbsr`: holes from the thumbnail enlarged
  (SwinIR default; Real-ESRNet, Real-ESRGAN), seams healed by LaMa. `diffusion`: Stable
  Diffusion inpainting with an optional prompt. `faces`: GFPGAN, skipped on bad landmarks.
- After the steps (in `enhance.py`, landed 2026-11-02):
  - `protect_real`: the clean-up result stays only in the holes and fades over 8 px into the
    untouched pixels: real detail is no longer blurred.
  - `blend_seams`: the colour step at every hole border is spread through the fill as a smooth
    field on the 8x8 grid: fewer sky bands where real and filled rows alternate.
  - both use numpy only (no OpenCV needed).
  - `destripe` (bands on flat surfaces evened against the thumbnail) is in the code but
    **switched off** (`DESTRIPE = False`), see below.
- Photos with no whole thumbnail: the GUI skips their enhanced copy, the CLI/MCP path makes it.

## FotoMedik for Claude (MCP)

- `FotoMedik.exe --mcp`: stdio JSON-RPC server, standard library only. Tools: `start_job`,
  `job_status`, `list_jobs`, `stop_job`, `app_info`.
- Jobs: JSON files in `%LOCALAPPDATA%\FotoMedik\jobs`, run one at a time by a detached runner
  (`--jobs`) that calls `--cli batch`. `skip_done` continues a stopped job. Output folders are
  checked: never the source, inside it or around it, never a protected place (`mcp.json`
  "protected" in the app's home).
- **Connect to Claude** (GUI) asks nothing and opens no window: it updates an installed
  extension in place, or writes `FotoMedik.mcpb` and shows it in Explorer (drag into Claude >
  Settings > Extensions once).
- The extension is of the **binary** kind: Claude starts the installed `FotoMedik.exe --mcp`.
- `--mcp-call <tool>` (one call, JSON in/out) serves the small JavaScript server used only when
  running from source.

What did not work (do not try again):
- Writing Claude's `claude_desktop_config.json` while Claude runs: Claude rewrites it from memory.
- A Node-type extension (Claude's built-in Node) on Windows: the server never got Claude's
  messages, with inherited streams, piping, or a JS server calling FotoMedik per tool.
- The windowed PyInstaller program ignores PYTHONIOENCODING: streams are reconfigured to UTF-8
  in `entry.py`; any exception must be caught there, or an error box hangs the process.
- On CI, Node crashes without SYSTEMROOT in the environment.

## Build and CI (`.github/workflows/build.yml`)

Jobs: `test-linux` and `dll-windows` (tests without OpenCV), `ai-models` (AI model tests,
check pictures to the `ci-preview` release), `app-windows` (PyInstaller folder app, smoke test
through the launcher, `mcp_smoke.py`, `ext_smoke.py`, screenshots). The app is published in the
`app` release (`FotoMedik.exe` installs it; it then updates itself part by part). The separate
CUDA build is gone: the AI runtime for the graphics card is downloaded by the app.
`fotomedik/_build.py` is stamped by CI (0 = from source); the build number is the run number.

## Bench (`scripts/bench.py`)

Good photos are damaged like real cards and recoveries (`make --kind mixed|dense|dense-head|
dense-nothumb|dense-rst|dense-rst-lost`), repaired, and scored against the truth (`score` for
pictures made by the app). Metrics: `psnr`, `good`, `stripes`, `ssim`, `keep` (SSIM on the
photo's own undamaged parts: did a step blur real detail), `detail` (fine detail in the damaged
area against the truth's: 1 = as sharp, lower = smeared), `faces`. The bench folder lives
outside the repository and has to be made again in a new environment.

Last result, 12 photos of a dense-rst-lost bench (1600 px), steps `puzzle,burst,deblock,thumbsr`:

| | psnr | good | stripes | ssim | keep |
|---|---|---|---|---|---|
| repair only | 21.68 | 0.633 | 0.271 | 0.735 | 0.890 |
| enhanced, build 119 | 29.28 | 0.942 | 0.015 | 0.872 | 0.928 |
| enhanced, build 120+ | 29.98 | 0.943 | 0.013 | 0.873 | 0.931 |
| + destripe | 29.98 | 0.943 | 0.013 | 0.873 | 0.931 |

## Tried, and what was learnt

- **Destripe without a thumbnail** (judging a block by its neighbours): made photos worse,
  broke clouds into new rectangles and cut lines. Without a reference a band and a real shade
  look alike. With a thumbnail it is safe but changed nothing on the bench, because photos
  with a whole thumbnail already lose their bands earlier. The photos that keep bands have no
  whole thumbnail. Hence: off until it can use a rescued thumbnail.
- **Rescued thumbnails** (`cli._rescued_thumbnail`, RGB; `dcfield.thumbnail_image` gives YCbCr)
  have the right shapes and colours where their mask says real; their lost part is a grey fill.
  (An earlier note said their colours were wrong: that was YCbCr shown as RGB.) Today only the
  jigsaw uses them; the thumbnail step, destripe and the hole colours do not.
- Guided FBCNN after the thumbnail step: went back to the old thumbnail handling; the guided
  network only on photos like those it learnt on.
- SwinIR in a batch mixed the tiles (scan-line noise): it enlarges one tile at a time.

## Open

Waiting for the owner's choice (proposed, not started):
1. Rescued thumbnail: let the thumbnail step, `destripe` and the hole colours use its real part.
2. No enhanced copy for photos with no real data at all (today the AI invents texture there).
3. A byte-level second repair round after the first (kept for last, only if the rest is not enough).

Known problems:
- The Enhancer fails on repaired files that do not open (header wholly destroyed).
- GUI and CLI treat rescued thumbnails differently (enhanced copy or not).
- onnxruntime's UTF-16 warning garbles job logs.
