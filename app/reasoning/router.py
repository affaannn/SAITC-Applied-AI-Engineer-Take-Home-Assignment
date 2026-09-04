from dataclasses import dataclass


@dataclass
class RouteDecision:
    route: str
    reason: str


class QueryRouter:
    """
    Lightweight deterministic router for assignment-specific
    query handling before retrieval/generation.
    """

    SYSTEM_PROMPT_PATTERNS = [
        "system prompt",
        "system instructions",
        "repeat your instructions",
        "show your instructions",
        "reveal your prompt",
    ]

    BYPASS_PATTERNS = [
        "bypass approval",
        "skip approval",
        "avoid approval",
        "circumvent approval",
        "ignore approval process",
    ]

    CALCULATION_PATTERNS = [
        "joins",
        "leaves",
        "leave entitlement",
        "calculate",
        "accrual",
    ]

    def route(
        self,
        query: str,
        ambiguous: bool = False,
    ) -> RouteDecision:

        q = query.lower().strip()

        if any(
            pattern in q
            for pattern in self.SYSTEM_PROMPT_PATTERNS
        ):
            return RouteDecision(
                route="refuse_prompt_extraction",
                reason="System prompt extraction attempt detected.",
            )

        if any(
            pattern in q
            for pattern in self.BYPASS_PATTERNS
        ):
            return RouteDecision(
                route="decline_policy_bypass",
                reason="Request attempts to bypass an approval process.",
            )

        if ambiguous:
            return RouteDecision(
                route="clarify",
                reason="Query is too ambiguous to answer reliably.",
            )

        if any(
            pattern in q
            for pattern in self.CALCULATION_PATTERNS
        ):
            return RouteDecision(
                route="calculator",
                reason="Deterministic calculation route selected.",
            )

        return RouteDecision(
            route="retrieve",
            reason="Normal grounded retrieval route.",
        )