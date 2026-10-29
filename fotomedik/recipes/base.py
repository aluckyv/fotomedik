"""Recipe = one kind of damage: how to detect it and how to fix it.

Adding support for a new corruption type means adding one small module with
a Recipe subclass and registering it. The pipeline (registry.run_pipeline)
applies recipes in priority order over a shared Context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from ..learn.profile import CameraProfile


@dataclass
class LogEntry:
    recipe: str
    message: str
    size_before: int
    size_after: int
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"recipe": self.recipe, "message": self.message, "size_before": self.size_before, "size_after": self.size_after}
        if self.extra:
            d["extra"] = dict(self.extra)
        return d


@dataclass
class Hole:
    """A region of the image that could not be recovered and was filled in."""

    mcu_first: int
    mcu_count: int
    reason: str

    def to_dict(self) -> dict:
        return {"mcu_first": self.mcu_first, "mcu_count": self.mcu_count, "reason": self.reason}


@dataclass
class Context:
    data: bytes
    name: str
    original: bytes = b""
    log: list[LogEntry] = field(default_factory=list)
    holes: list[Hole] = field(default_factory=list)
    ref: "CameraProfile | None" = None
    options: dict[str, Any] = field(default_factory=dict)
    suggested_name: str | None = None
    fallback: str | None = None  # e.g. "thumbnail" when only the Exif preview survived
    recovered_fraction: float | None = None  # share of MCUs kept from the original (set by R9)
    notes: list[str] = field(default_factory=list)
    cache: dict[str, Any] = field(default_factory=dict)  # per-stage scratch (parse results etc.)

    def __post_init__(self):
        if not self.original:
            self.original = self.data

    def replace_data(self, new: bytes, recipe: "Recipe | str", message: str, **extra) -> None:
        name = recipe if isinstance(recipe, str) else recipe.name
        self.log.append(LogEntry(name, message, len(self.data), len(new), dict(extra)))
        self.data = new
        self.cache.clear()

    def note(self, recipe: "Recipe | str", message: str, **extra) -> None:
        name = recipe if isinstance(recipe, str) else recipe.name
        self.log.append(LogEntry(name, message, len(self.data), len(self.data), dict(extra)))

    @property
    def applied(self) -> list[str]:
        seen: list[str] = []
        for e in self.log:
            if e.recipe not in seen:
                seen.append(e.recipe)
        return seen


class Recipe:
    """Base class. Subclasses set `name`, `priority` (lower runs first) and implement detect/fix."""

    name: str = ""
    priority: int = 100
    description: str = ""

    def __init__(self):
        if not self.name:
            raise TypeError(f"{type(self).__name__} must define a non-empty `name`")

    def detect(self, ctx: Context) -> Any | None:
        """Return a truthy finding object when this recipe applies, else None."""
        raise NotImplementedError

    def fix(self, ctx: Context, finding: Any) -> None:
        """Mutate ctx (via replace_data / note / holes) to repair the finding."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<Recipe {self.name} p={self.priority}>"
