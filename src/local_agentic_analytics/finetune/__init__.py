"""Synthetic dataset generation for fine-tuning the energy insight narrator.

This package builds an instruction-tuning corpus that teaches a small language
model to enrich its statistical/electrical vocabulary *while* keeping every
number grounded in the input statistics block. It is intentionally dependency
light and fully testable offline through an injected fake teacher client.
"""

from __future__ import annotations
