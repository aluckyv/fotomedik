"""The installed versions of the packages pinned in base-constraints.txt, one per line: the
part of the app base's id that comes from Python packages (.github/workflows/build.yml). The
build machine's own tools (pipx and what it needs) stay out: their updates do not change the
base, but they used to change its id, and every such build asked for a new base download."""
import re
import subprocess
import sys
from pathlib import Path


def name(line: str) -> str:
    return re.split(r"[=<>!~ ]", line.strip(), maxsplit=1)[0].lower().replace("_", "-")


pins = {name(x) for x in Path(sys.argv[1] if len(sys.argv) > 1 else "base-constraints.txt").read_text().splitlines()
        if x.strip() and not x.lstrip().startswith("#")}
frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout
print("\n".join(sorted(x for x in frozen.splitlines() if name(x) in pins)))
