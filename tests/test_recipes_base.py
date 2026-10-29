import pytest

from fotomedik.recipes import base, registry
from fotomedik.recipes.base import Context, Recipe
from synth import xor_prefix


class _Upper(Recipe):
    name = "upper"
    priority = 20

    def detect(self, ctx):
        return "lower" if ctx.data != ctx.data.upper() else None

    def fix(self, ctx, finding):
        ctx.replace_data(ctx.data.upper(), self, "uppercased")


class _Strip(Recipe):
    name = "strip"
    priority = 10

    def detect(self, ctx):
        return "spaces" if ctx.data.startswith(b" ") else None

    def fix(self, ctx, finding):
        ctx.replace_data(ctx.data.lstrip(b" "), self, "stripped")


class _Never(Recipe):
    name = "never"
    priority = 5

    def detect(self, ctx):
        return None

    def fix(self, ctx, finding):  # pragma: no cover
        raise AssertionError("must not run")


def test_pipeline_runs_in_priority_order_and_logs():
    ctx = Context(data=b"  abc", name="x")
    out = registry.run_pipeline(ctx, recipes=[_Upper(), _Never(), _Strip()])
    assert out.data == b"ABC"
    assert [e.recipe for e in out.log] == ["strip", "upper"]
    assert out.log[0].message == "stripped"
    assert out.log[0].size_before == 5 and out.log[0].size_after == 3
    assert out.applied == ["strip", "upper"]


def test_pipeline_skips_when_detect_returns_none():
    ctx = Context(data=b"ABC", name="x")
    out = registry.run_pipeline(ctx, recipes=[_Upper(), _Strip()])
    assert out.data == b"ABC" and out.log == []


def test_context_keeps_original():
    ctx = Context(data=b" a", name="x")
    registry.run_pipeline(ctx, recipes=[_Strip()])
    assert ctx.original == b" a" and ctx.data == b"a"


def test_register_decorator_adds_to_default_list():
    before = len(registry.all_recipes())

    @registry.register
    class _Tmp(Recipe):
        name = "tmp-test-recipe"
        priority = 999

        def detect(self, ctx):
            return None

        def fix(self, ctx, finding):  # pragma: no cover
            pass

    try:
        assert len(registry.all_recipes()) == before + 1
        assert registry.all_recipes()[-1].name == "tmp-test-recipe"
    finally:
        registry.unregister("tmp-test-recipe")
    assert len(registry.all_recipes()) == before


def test_r1_cardrecovery_decodes(healthy_jpeg):
    from fotomedik.recipes.r1_cardrecovery import CardRecoveryXor

    ctx = Context(data=xor_prefix(healthy_jpeg), name="Temporary_v5_File0001_JPG.~CR")
    out = registry.run_pipeline(ctx, recipes=[CardRecoveryXor()])
    assert out.data == healthy_jpeg
    assert out.applied == ["cardrecovery_xor"]
    assert out.suggested_name == "Temporary_v5_File0001_JPG.jpg"


def test_r1_leaves_normal_jpeg_alone(healthy_jpeg):
    from fotomedik.recipes.r1_cardrecovery import CardRecoveryXor

    ctx = Context(data=healthy_jpeg, name="a.jpg")
    out = registry.run_pipeline(ctx, recipes=[CardRecoveryXor()])
    assert out.data == healthy_jpeg and out.log == []


def test_default_registry_contains_r1():
    import fotomedik.recipes  # noqa: F401  (importing the package registers built-ins)

    assert "cardrecovery_xor" in [r.name for r in registry.all_recipes()]


def test_recipe_requires_name():
    class Bad(Recipe):
        def detect(self, ctx):
            return None

        def fix(self, ctx, finding):
            pass

    with pytest.raises(TypeError):
        Bad()
