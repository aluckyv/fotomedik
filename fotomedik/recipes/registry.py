"""Recipe registry and the pipeline that applies recipes to a Context."""

from __future__ import annotations

from collections.abc import Iterable

from .base import Context, Recipe

_RECIPES: list[Recipe] = []


def register(cls: type[Recipe]) -> type[Recipe]:
    """Class decorator: instantiate and add to the default recipe list."""
    inst = cls()
    if any(r.name == inst.name for r in _RECIPES):
        raise ValueError(f"recipe {inst.name!r} registered twice")
    _RECIPES.append(inst)
    return cls


def unregister(name: str) -> None:
    _RECIPES[:] = [r for r in _RECIPES if r.name != name]


def all_recipes() -> list[Recipe]:
    return list(_RECIPES)


def run_pipeline(ctx: Context, recipes: Iterable[Recipe] | None = None, max_passes: int = 3) -> Context:
    """Apply recipes in priority order.

    A fix can expose new damage (decoding a .~CR file reveals a broken header),
    so the list is walked again until a pass applies nothing or max_passes is hit.
    """
    ordered = sorted(_RECIPES if recipes is None else recipes, key=lambda r: r.priority)
    for _ in range(max_passes):
        applied_this_pass = False
        for r in ordered:
            finding = r.detect(ctx)
            if finding is None or finding is False:
                continue
            before = len(ctx.log)
            r.fix(ctx, finding)
            if len(ctx.log) != before:
                applied_this_pass = True
        if not applied_this_pass:
            break
    return ctx
