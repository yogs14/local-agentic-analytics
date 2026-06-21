"""Immutable toggles for ablating the text-to-SQL pipeline scaffolding."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineToggles:
    """Switch deterministic scaffolding on or off for faithful ablation.

    All toggles default to ``True`` so the default configuration reproduces the
    current production pipeline behavior exactly. Set a toggle to ``False`` to
    isolate the contribution of that component.
    """

    use_rule_based_resolver: bool = True
    apply_domain_normalization: bool = True
    use_semantic_guard: bool = True
    use_repair: bool = True
    # When True the planner may consult the LLM to pick a retrieval route after
    # the deterministic resolver is unsure. When False only the deterministic
    # ``RuleBasedRouteResolver`` runs (no LLM planning call). Energy is always
    # short-circuited to STRUCTURED_SQL regardless of this flag.
    use_planner: bool = True
