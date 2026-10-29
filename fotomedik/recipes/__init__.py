"""Repair recipes. Importing this package registers the built-in recipes."""

from . import r1_cardrecovery, r2_header, r4_foreign, r5_bitflip, r6_garbage, r7_reference, r8_thumbnail, r9_holes, r10_tails, r11_segments, r12_smooth, r13_orientation  # noqa: F401
from .base import Context, Hole, LogEntry, Recipe  # noqa: F401
from .registry import all_recipes, register, run_pipeline, unregister  # noqa: F401
