# FotoMedik: state of the project

Read this first in a new session. It says how the work is done, what the program does today,
what was tried and did not work, and what is open. Keep it up to date when something lands.
(The repository is public: no personal photos, folder names or paths in here.)

Last update: 2026-11-03, build 161 (159 was the first build since 154 that passed CI: the program ends with TerminateProcess; 160: R7 margin, no enhanced copy for a rotted preview alone; 161: R7 judges an own header that decodes nothing the old way).

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
| R7 | destroyed header | transplant from a camera profile; header voted bit by bit from every photo of the set. A header that parses is kept unless a reference decodes clearly further (build 157: photos saved with their own optimised Huffman tables look "near" each other's; with dense rot every header stopped at the first rotted sector and 5 of 15 got another photo's tables; the margin is 64 MCUs and double, since wrong tables decode garbage for up to 80 MCUs before an impossible code shows) |
| R8 | nothing of the image | the Exif thumbnail |
| R9 | lost intervals | holes filled (thumbnail-guided DC field, smooth surface, not flat squares) |
| R10 | intact tail after a rotted stretch | tail scan in C, cut where colours step away from the thumbnail |
| R11 | intact stretches in the middle of damage | placed by the thumbnail, or by their own seam without one |
| R12 | holes | smooth surface (`surface()`); tails after a hole set to the level of their seams (`tail_shifts`, see below) |
| R13 | orientation | portrait photos upright |
| R14 | decoded garbage (patterns, saturated patches, another photo's data) | becomes a hole and is filled again |
| R15 | rotted Exif thumbnail (the preview galleries show) in a whole or repaired picture | a fresh one from the picture, written into the Exif block in place (`jpeg/exifthumb.py`); the scanner reports `thumbnail_damaged`, the overview counts them |

Also: damaged thumbnails are rescued (header borrowed from a healthy thumbnail of the same
camera, `dcfield._rescue_thumbnail`); the jigsaw puts the pieces of photos with restart
markers back by the thumbnail (`ai/puzzle.py`, used as the `puzzle` step).

Tail levels (R12, build 124): a run decoded after a hole in the same interval has a DC level
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
  (far off the thumbnail's colour, `puzzle.BLOCK_OFF = 40`) are left out. Build 124: whole
  regions of rows the repair put 8, 16, 24 rows off are moved first (`row_shifts`,
  `shift_regions`): rows compared with the thumbnail at shifts of 8, clear rows decided, rows
  between two that moved alike move alike, a lone decision not followed. Before, moved pieces
  and unmoved repair rows (a sky fits anywhere) interleaved into stripes (100_0223 lost 24
  rows at the top). Bench: 16 of 17 moved rows right; DSC_4537 psnr 28.86 -> 29.10.
- `patch` (build 124, off by default): holes given the photo's own texture like Content-Aware
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
  - deblock without a whole thumbnail (build 134): the guided network gets no hole map, so it
    only cleans what the repair and the jigsaw put in the holes. Before, it guessed grey and
    brown bands into skies (100_0223) and LaMa took its block colours from that guess. Photos
    with a thumbnail (the whole bench) are not affected.
  - Experiments as steps (build 135, off unless named in `steps`): `rows` evens faint runs of
    blocks on flat surfaces against a rescued thumbnail's real part (`even_rows`, 6 levels, runs
    of 4 blocks); `rowsnear` against the rows above and below when those agree; `holemap` gives
    the clean-up network the hole map without a thumbnail, as before build 134 (for A/B tests on
    the owner's graphics card). On 100_0223, `rowsnear` changed almost nothing: its stripes are
    stacked, so the rows around are off too. `rows` calmed its sky a lot; faces and skin are
    left out (build 136: without that, a face passed for a flat surface and went blotchy).
  - Graphics card results, build 136 (2026-11-02): `rowsnear` changed nothing anywhere (0223:
    0.04 % of pixels; dense-head and dense-nothumb benches identical to 134) and was removed
    (build 137). `rows` on dense-head: psnr 15.23 -> 15.25, good 0.422 -> 0.428 (better on 13
    of the 14 photos it touched), ssim 0.6016 -> 0.6010; on 100_0226 it put white and grey blots
    into clouds and darkened a tree, because the rescued thumbnail's blurred colour was taken as
    the target where the thumbnail itself changes. Build 137: the correction fades out where
    the reference spreads 6..14 levels over 5x5 blocks (soft, 12 px), so only plain sky moves.
    `holemap` (the old behaviour) on dense-nothumb: 13.06 vs 13.07 dB, equal; build 134 stays
    for 0223's sake.
  - Build 137 on the graphics card: 100_0226's clouds clean again, 0223's sky still calm;
    dense-head psnr 15.23 -> 15.24, good 0.4221 -> 0.4232, ssim 0.6016 -> 0.6018 (better on 9 of
    the 10 photos it touches, DSC_4524 ssim -0.002). The owner made it default: build 139,
    `EVEN_ROWS = True` (photos without a whole thumbnail that have a rescued one).
  - SDXL (build 140, step `sdxl`, an experiment): the GitHub export machine died converting
    the UNet to fp16 (out of memory, no log kept); `scripts/export_sdxl.bat` did it on the
    owner's computer (64 GB) in about an hour, all five parts load. `DiffusionXL` in
    `diffusion.py`: the two text encoders' second-to-last layers joined (2048), the second's
    pooled `text_embeds`, `time_ids` (size, crop 0, size), latents x 0.13025, 1024 windows,
    the same DDIM loop. The fp16 folder goes into the models folder as `sdxl-inpaint`
    (`enhance.SDXL_DIR`); without it the step falls back to SD 1.5 with a note.
    Result (build 141, RTX 4070 12 GB, 100_0223): 2106 s for 12 windows (SD 1.5: 36 s), and
    not better: more dark streaks in the sky, horizontal lines across the faces' area. Build
    142 lets the text encoders go and keeps CUDA pools as small as asked (untested). The owner
    left it as an option and stopped there (2026-11-02).
  - `level_real` (build 133, `LEVEL_FROM_NET`): the guided network moves decoded rows of a
    rotted level (too bright, tinted; the repair kept them) to the thumbnail's level;
    protect_real used to put them back between its fill as bands (100_0239: dark and light
    lines over the whole photo). Now real pixels take the network's low-frequency level
    (window 7 x 49 px over real pixels only) where it differs by 6..16+ gray levels, and keep
    their own detail. Bench (12 photos): psnr 29.75 -> 29.80, stripes 0.0131 -> 0.0129, better
    on 11 of 12, none worse; it touches ~1 % of the real picture there (decoded rows on the
    bench are mostly at the right level). Mistake made on the way (do not repeat): forcing `keep` pixels back to
    the raw input after protect_real dropped its feathered edge (keep = all real pixels after
    the jigsaw): -0.41 dB and more stripes on every photo.
  - `blend_seams`: the colour step at every hole border is spread through the fill as a smooth
    field on the 8x8 grid: fewer sky bands where real and filled rows alternate.
  - both use numpy only (no OpenCV needed).
  - `bridge_thin` (build 128, **off since build 132**, `enhance.BRIDGE = False`): holes up to
    32 px tall between real rows blended column by column from the real pixel above and below
    (meant for flat bands across legs and shop fronts, 100_0239). It smeared feet, floors and
    beams into vertical streaks where LaMa or the guided network had drawn them well: bench (6
    photos) psnr 28.69 -> 29.64 and stripes 0.0252 -> 0.0203 with it off, better on every
    photo; 100_2311 lost its "ghost legs". Do not switch it on again without a bench.
  - `destripe` (build 126, on again, numpy only): bands on flat surfaces evened against the
    thumbnail, or only where a rescued thumbnail is real (`rescued_picture`). Small effect.
  - `TEXTURE_WITH_FILL` (build 126): the patch step's texture runs after thumbsr/ai/diffusion
    without being asked for. The owner confirmed it stays on (2026-11-02).
    Build 130: texture is looked for near the hole first (`patchfill.NEAR`), and only close
    matches are taken (`MATCH_MAX = 10`): build 128 copied wrong texture from far away.
  - guided-network path (build 129): rotted stripes along the holes are taken out there too
    (before, only the thumbnail and LaMa paths did it). Bench: no measurable effect.
  - `seam_guard` (build 131): per run of hole blocks in a row, the new fill is compared with the
    fill the photo came with (repair or jigsaw) at the real rows above and below; where the new
    one meets them worse (`SEAM_PREF = 0.75`, `SEAM_MIN = 4.0`), the earlier fill stays. Meant
    for the dark bars and flat bands of 100_0239; not yet tried on the graphics card. On the
    bench and on 100_2311 it kept nothing (no harm, no effect so far).
  - Bench of the night's steps (6 photos, one off at a time, against build 131): texture costs
    0.08 dB psnr and 0.012 ssim, gives detail 0.65 -> 0.74; destripe and the guided-path stripes
    change nothing measurable; bridging was the loss (above).
  - diffusion (SD 1.5, build 126): faces, skin (with a margin) and the date stamp are never
    repainted; they keep the earlier fill.
  - `flat_ground` (build 155, photos without a whole thumbnail): blocks flat in the repaired
    copy, flat in the enhanced copy too, of the same tone (per channel: a pink junk row
    healed to gray differs) and on flat ground around keep the repaired copy's pixels,
    feathered 6 px; runs after the texture step, before destripe and rows. Why: on 100_0270
    (no thumbnail, 53 % holes) the sky, smooth in the repaired copy, came out banded: LaMa
    paints each wide hole band up to 8 levels lighter (the holes are unknown, so
    `keep_block_means` left their tones), real sky rows were healed as "wrongly coloured
    stripes", and the texture step copied lines in. 45 % of 0270 went back; people, poles and
    benches kept the AI's work. The LaMa path also pins unknown holes' block means to the
    fill's where the fill and the ground around are flat (`flat_holes`, not for junk rows).
- Photos with no whole thumbnail: the GUI skips their enhanced copy, the CLI/MCP path makes it.

## FotoMedik for Claude (MCP)

- `FotoMedik.exe --mcp`: stdio JSON-RPC server, standard library only. Tools: `start_job`,
  `job_status`, `list_jobs`, `stop_job`, `update_app`, `app_info`.
- `update_app` (build 125, `jobs.update_program`): between jobs, brings the installed program up
  to the newest GitHub build; a code-only build is swapped in at once (new jobs use it), a new
  base is staged for the launcher. The running server keeps its old code until Claude restarts
  it: `app_info` shows `build` (the server's) and `installed_build` (on disk, what jobs run).
  A staged base is NOT put in by restarting Claude: Claude starts `app\base\FotoMedik.exe`
  itself, not the launcher, and holds its files. The owner closes Claude fully, opens
  FotoMedik once from its shortcut (the launcher swaps the parts in), then opens Claude.
  The base id hashes `python --version`, `pip freeze`, the base's module list and the spec. Since
  build 138 the Windows build pins Python 3.12.10 and every base package (`base-constraints.txt`,
  including click and packaging, which the build machine's Python already carries and its image
  updates), and keeps the id's text as `base_id.txt` with the build's files, to diff when a
  build brings a new base again. Builds 135, 136 and 137 each brought one without a change of ours;
  139 -> 140 -> 141 were code-only, but 142 brought one again (cause unknown: the artifact with
  `base_id.txt` cannot be fetched from here, so the build log now prints it too). Build 143's log
  showed the cause: `pip freeze` also listed the build machine's own pipx, argcomplete,
  filelock, platformdirs, userpath and colorama, which its image updates. Since build 144 the id
  takes only the packages pinned in `base-constraints.txt` (`scripts/base_freeze.py`).
- 100_2311's light line (2026-11-02): across a hole the whole photo wide, the repair kept a
  6-block decoded stretch (black, checkerboard, coloured dots) as real; the junk check caught
  one block of it. On the benches (dh, dn, dense, rstl) decoded stretches of up to 16 blocks
  with holes on both sides in their row were right (within 8 levels) only 5 % of the time,
  junk (over 30) about half. Build 144 experiments: `flank` takes all of them as holes
  (`flanked_runs`), `flankjunk` only those that jump against the block above or below or are
  much rougher (`flank_junk`; 2311: 36 of 188 blocks, about half of the spot).
  Graphics card (build 145): 2311's checkerboard and coloured dots gone with `flank`
  (`flankjunk` left the dots); dense-head psnr 15.24 -> 15.25, stripes 0.4647 -> 0.4625 (better
  on 10, worse on 3, worst -0.11 dB); dense-nothumb 13.07 -> 13.06, even. The owner made
  `flank` default (build 146, `FLANK = True`); `flankjunk` removed.
  A server started before build 125 has no `update_app` at all: Claude has to be restarted once.
- Claude's screen control of the owner's computer lapses after 30 minutes without use and needs
  the owner's approval again: do not plan unattended installs on it.
- The window (build 147): a menu bar (Photos: choose, save to, references, open results;
  Program: Connect to Claude, Check for updates, About) and a **Quality** preset instead of the
  row of step checkboxes: Faithful (repair only), Standard (`puzzle,burst,deblock,thumbsr`, the
  jobs' default), Maximum (`puzzle,burst,deblock,diffusion,faces`), Custom (opens the Advanced
  panel, where the step checkboxes, the enlarger, the diffusion preset and scene, and the
  repair's options - magenta marks, reports, colours from similar shots - live). A step ticked
  by hand turns the preset into whatever matches, else Custom (`gui/app.py PRESETS`).
- The quick look (`overview.py`, build 148): before the repair, the chosen photos are looked at
  in the background (the set's headers learnt as in the repair, then the decoder's first pass
  and the thumbnail of each photo, whole / partly rescued / none, faces looked for on the
  thumbnail enlarged, the time taken): one line under the inputs says what the folder holds and
  which preset fits (`Overview.suggest`: Faithful when nothing is damaged, else Standard; faces
  and a graphics card are mentioned for Maximum). The preset follows the suggestion until the
  person picks one by hand. Also `fotomedik overview FOLDER [--photos]` and the MCP tool
  `look_at_folder` (Claude sees the same numbers before starting a job). Synthetic files: a
  colour photo without a thumbnail got a thumbnail "rescued" from its own scan (Pillow's tables
  match the set's thumbnails); real cameras' did not (Sarimsakli: 13 of 38 came out "none").
- **Enhance again…** (build 149, `gui/again_dialog.py`): another enhanced copy of the selected
  photo with other steps (the dialog starts from the steps it had; enlarger, diffusion preset,
  scene too). The photo is run again on its own with the set's learned headers kept; the copy
  is saved as `<name>_enhanced_2.jpg`, `_3`... (`options["ai_suffix"]`, `run.free_suffix`) and
  comes in as a new row "(again_2: steps)", so tries sit side by side; the Enhanced box's
  caption names the steps. The first copy is never overwritten.
- Builds 149-153 failed their self-test at the very end ("Fatal Python error: Aborted", no
  Python frame): the self-test quit the program as soon as the results were in, while the
  repair's QThread was still ending, and Qt aborts a program that destroys a running thread
  (148 passed by timing). `app._settle` (build 154) stops the worker, waits for the thread and
  closes the window before Qt is torn down, in the self-test and in the normal exit. Found
  through `fotomedik_app.py`'s new crash file: an error that ends the program, and a hard crash
  (faulthandler), go to `FotoMedik-crash.txt` in the program's home; CI prints it. Build 155
  (docs only) then died with an access violation at teardown, no Python frame: Qt's and
  Python's teardown order is not ours to fix, so since build 157 `fotomedik_app.py` ends the
  process with `os._exit` once the window is closed and its threads are done (stdio and the
  crash file flushed first); `_settle` also drops the worker before the window.
- At start the packaged window brings an installed extension up to date on its own
  (`claude_connect.connect(quiet=True)`, build 147): nothing is asked, written or shown when
  FotoMedik is not in Claude yet. Program > Check for updates asks GitHub on demand.
- Build numbers are GitHub run numbers (failed runs count too): main went 121 -> 124.
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
| build 131 (texture, bridging) | 28.71 | 0.932 | 0.018 | 0.848 | 0.916 |
| build 132 (bridging off) | 29.75 | 0.940 | 0.013 | 0.855 | 0.920 |
| build 133 (+ level_real) | 29.80 | 0.941 | 0.013 | 0.855 | 0.920 |

### Full-size test: the wedding set (2026-11-02, build 154)

15 photos of 16 MP (4948x3280, saved with optimised Huffman tables, a 160x120 thumbnail),
once with the thumbnail's scan rotted and the picture whole, once with the picture's scan
rotted (dense rot, 18-38 % of the sectors) and the thumbnail whole; repaired by the app on the
owner's card, scored against the originals (the set lives outside the repository).

- Thumbnail rotted: the picture comes out byte-identical (good), but the app called the photos
  healthy and left the rotted preview in the copy. Build 157: R15 and the `thumbnail_damaged`
  finding.
- Picture rotted: psnr 23.6 / 26.4 (repair / enhanced), good 0.77 / 0.84, detail 0.38 / 0.55.
  Colours and shapes are right, detail is half the truth's. Two causes, measured per MCU
  against the truth (scripts in the session's scratch, `mcu_audit`, `chain_audit`, `placer_audit`):
  1. R7 gave 5 of 15 photos another photo's Huffman tables (fixed, build 157).
  2. The decoder finds and syncs 95 % of the MCUs that survived in intact stretches, but at
     16 MP only 4 % of the stretches with detail land at their place, 14 % one to three rows
     off, 82 % are left out. The placer's reference is the 160x120 thumbnail: one of its
     pixels covers 2.6 MCU rows, so the best fit is the true place in only 30-50 % of the
     searches (a whole number of rows off otherwise), and the DISTINCT rule (a winner must
     beat every place more than 3 MCUs away) accepts 1-5 %. On the 1600 px bench the same
     rule accepts 66 % and the best fit is the true place in 94 %. The cost at the true place
     passes ACCEPT in 58-92 % of the stretches: the reference recognises them, it cannot tell
     the row.

Build 160 on the wedding set: picture rotted, repair good 0.89 / detail 0.50, enhanced
good 0.92 / detail 0.82 (154: 0.77/0.38 and 0.84/0.55); thumbnail rotted: all 15 copies
byte-identical pictures with a fresh preview, no enhanced copy. On the 1600 px benches
(enhanced): dh good 0.42 -> 0.44, ssim 0.60 -> 0.62, detail 0.61 -> 0.48; dn unchanged
(detail 0.61 -> 0.59). The detail drop is flat_ground doing what it is for (the smooth
repair kept on flat ground, no invented texture there) on photos that are mostly lost
(good 0.02-0.18); psnr/good/ssim did not move. One dh photo (DSC_4403) came out broken in
160: R2 had mended its header so it parsed, its tables decoded nothing, and the strict
rule kept them; build 161: an own header that decodes fewer than STRICT_MIN_OWN MCUs is
judged the old way.

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

Next (owner's choice, 2026-11-02): the placer at full size (R11). The row from the byte
count (R11 has it, within a row or two) and from the seams with real rows above and below,
the thumbnail for the column only, DISTINCT compared inside the chosen row, ACCEPT scaled by
the thumbnail-to-picture ratio. Measure on the wedding set (stretches at their place: 4 %
today) and on the dh/dn benches (no regression).
First try (2026-11-03, not kept, patch in the session's scratch): placed pieces taught to
the reference field (clean mask + rebuild, 0.08 s per rebuild at 16 MP), the decoded prefix
before the first error counted as clean, a row prior on the misfit, the chain-by-chain count
as the search centre (+-2.5 rows). No gain on DSC_4230 (12 of 241 stretches at their place):
the count itself drifts 2.8 rows (median) because nothing gets placed to re-anchor it, and
when the best place is the true one its runner-up is the row above or below (92 %), at 0.76
of its cost. What is still missing is a row test the thumbnail cannot give: the seam between
a piece's blocks and the real blocks above and below (DC or pixels), and placing the first
pieces after the real rows at the top by the count alone (it is within 24 MCUs there).
Second try (the same night, not kept, patch in the scratch): pixel seams between a chain's
edge pixels and known neighbours (the real top rows, placed pieces; `chain_pixels` with a
frame of only the rows needed, 1 ms a chain), soft placements for flat chains (no anchor,
the count goes on), anchors only from textured chains, a joint score (misfit + seam +
count prior). Still no good: with the truth's own pixels as neighbours (an oracle) the seam
tells the true row from the next one in only 66 % of textured blocks on these heavily
compressed photos (seam 0.98 at the true row, 1.99 one row off), the count's error per gap
is 10-30 % with a heavy tail (3x where a gap spans a change of content), and every
decision rule loose enough to place something placed wrong rows and the anchors cascaded.
Conclusion: on a 16 MP photo with a 160x120 thumbnail and no restart markers, the row of
a piece is not locally decidable; a joint solution over the whole sequence of chains
(the count's gap estimates with heavy-tailed tolerance, the thumbnail misfit and the seams
scored together, dynamic programming like `place_sequence`) is the remaining route, a
multi-day piece of work. Measuring tools in the session's scratch: `eval_place.py`
(stretches at their place), `seam_debug.py` (every decision against the truth),
`seam_oracle.py`.

Waiting for the owner's choice (proposed, not started):
1. Rescued thumbnail: let the thumbnail step, `destripe` and the hole colours use its real part.
2. No enhanced copy for photos with no real data at all (today the AI invents texture there).
3. A byte-level second repair round after the first (kept for last, only if the rest is not enough).

SDXL: see the enhanced copy's notes (step `sdxl`, an experiment; slow and not better on the
owner's card). The GitHub export workflow dies on memory; `scripts/export_sdxl.bat` does it
on the owner's computer.

Tidying the program (owner's plan, 2026-11-02, in this order): 4. connect to Claude at start
(build 147), 1. menu and presets (147), 2. the quick look before the repair (148), 3. Enhance again (149); after a result: redo one photo's enhanced copy with other steps and
compare. Then faces: GFPGAN alone helps little (sharper eyes on fair faces, the worst faces
left alone by `agrees`, three non-faces touched); next is recognition of the owner's people
across photos (ArcFace) plus a face network fine-tuned on them, as a night job on his card.

Known problems:
- Rotted decoded rows the stripe detection misses stay in the enhanced copy as thin light or
  dark lines (100_2311's was a short stretch between two holes: `flank`, build 146).
- The Enhancer fails on repaired files that do not open (header wholly destroyed).
- GUI and CLI treat rescued thumbnails differently (enhanced copy or not).
- (fixed in build 127: onnxruntime warnings off, job log lines cleaned)
