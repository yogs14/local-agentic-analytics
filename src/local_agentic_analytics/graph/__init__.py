"""Workflow graph modules."""

__all__ = ["LangGraphAnalyticsWorkflow", "SequentialAnalyticsWorkflow"]


def __getattr__(name: str):
    if name == "LangGraphAnalyticsWorkflow":
        from local_agentic_analytics.graph.langgraph_workflow import (
            LangGraphAnalyticsWorkflow,
        )

        return LangGraphAnalyticsWorkflow
    if name == "SequentialAnalyticsWorkflow":
        from local_agentic_analytics.graph.workflow import SequentialAnalyticsWorkflow

        return SequentialAnalyticsWorkflow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
