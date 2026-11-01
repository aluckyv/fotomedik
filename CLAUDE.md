# FotoMedik

Diagnosis and repair of damaged JPEG photos (Windows app, CLI, MCP server for Claude).

**Read `docs/STATE.md` first**: how the work is done, what the program does, what was tried
and did not work, what is open. Update it when something lands.

The short rules:
- Ask the owner before every change and wait for the OK; offer options before choosing an approach.
- Replies in Serbian, short. Code, comments, commits and docs in English.
- Originals are never written; output goes to another folder. Photos never go into this
  public repository, nor personal folder names or paths.
- Work in progress on a branch; `main` after the OK. Run the tests also without OpenCV
  before pushing (see `docs/STATE.md`).
