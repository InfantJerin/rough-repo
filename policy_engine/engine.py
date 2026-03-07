from __future__ import annotations

import copy
from typing import Any

from policy_engine.models import UserContext, PolicyDecision
from policy_engine.opa_client import OPAClient


class PolicyEngine:
    """
    Application-layer policy engine that sits between MCP tools and OpenSearch.

    Responsibilities:
    1. Call OPA to get a policy decision for a user.
    2. Translate that decision into OpenSearch query modifications
       (document filters + field masks).

    This module is deliberately decoupled from the MCP server so it can be
    reused in other services.
    """

    def __init__(self, opa_client: OPAClient):
        self.opa = opa_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, user_ctx: UserContext, memo_ids: list[str] | None = None) -> PolicyDecision:
        """Ask OPA for a policy decision and return a structured object."""
        raw = self.opa.evaluate_memo_authz(user_ctx.user_id, memo_ids)
        return self._parse_decision(raw)

    def apply_to_query(self, query_body: dict[str, Any], user_ctx: UserContext) -> dict[str, Any]:
        """
        Take an OpenSearch query body and inject DLS filters + FLS source masks
        based on the user's policy decision.

        Returns a *new* query body (does not mutate the original).
        """
        decision = self.evaluate(user_ctx)
        return self._inject(copy.deepcopy(query_body), decision)

    def apply_decision_to_query(self, query_body: dict[str, Any], decision: PolicyDecision) -> dict[str, Any]:
        """
        Same as apply_to_query but accepts a pre-evaluated decision.
        Useful when you need the decision for other checks too.
        """
        return self._inject(copy.deepcopy(query_body), decision)

    def check_document_access(self, decision: PolicyDecision, memo_id: str, document: dict[str, Any]) -> bool:
        """
        Check whether a single document is accessible under the decision.
        Used for get_memo (single doc fetch).
        """
        if decision.unrestricted:
            return True

        # Deal team check
        if decision.use_deal_teams and memo_id in decision.deal_team_memos:
            return True

        # Coverage check
        if decision.use_coverage and decision.coverage_filter:
            match = all(
                document.get(field) in values
                for field, values in decision.coverage_filter.items()
            )
            if match:
                return True

        return False

    def filter_fields(self, document: dict[str, Any], decision: PolicyDecision) -> dict[str, Any]:
        """Strip fields from a document that the user is not allowed to see."""
        if not decision.allowed_fields:
            return document
        return {k: v for k, v in document.items() if k in decision.allowed_fields}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_decision(raw: dict[str, Any]) -> PolicyDecision:
        doc_filter = raw.get("document_filter", {})

        # OPA returns deal_team_memos as a list or set
        deal_team_memos = list(doc_filter.get("deal_team_memos", []))
        coverage_filter = doc_filter.get("coverage_filter", {})
        use_deal_teams = doc_filter.get("use_deal_teams", False)
        use_coverage = doc_filter.get("use_coverage", False)
        role_filters = doc_filter.get("role_filters", [])

        unrestricted = not bool(doc_filter)  # empty dict means unrestricted

        return PolicyDecision(
            allow=raw.get("allow", False),
            user_role=raw.get("user_role", ""),
            deal_team_memos=deal_team_memos,
            coverage_filter=coverage_filter,
            use_deal_teams=use_deal_teams,
            use_coverage=use_coverage,
            role_filters=role_filters,
            unrestricted=unrestricted,
            allowed_fields=list(raw.get("allowed_fields", [])),
        )

    @staticmethod
    def _inject(body: dict[str, Any], decision: PolicyDecision) -> dict[str, Any]:
        """Inject DLS + FLS into an OpenSearch query body."""

        # ---- DLS: inject document filter ----
        if not decision.unrestricted:
            security_filter = PolicyEngine._build_security_filter(decision)
            if security_filter:
                body.setdefault("query", {}).setdefault("bool", {}).setdefault("filter", [])
                # filter may be a single dict, normalise to list
                existing = body["query"]["bool"]["filter"]
                if isinstance(existing, dict):
                    body["query"]["bool"]["filter"] = [existing, security_filter]
                else:
                    existing.append(security_filter)

        # ---- FLS: restrict returned fields ----
        if decision.allowed_fields:
            body["_source"] = {"includes": decision.allowed_fields}

        return body

    @staticmethod
    def _build_security_filter(decision: PolicyDecision) -> dict[str, Any] | None:
        """
        Build a bool filter that combines:
        - deal team memo IDs (OR)
        - coverage filters (AND within coverage, OR with deal teams)
        - role-level arbitrary filters (AND on top)
        """
        # Build the access filter (deal team OR coverage)
        access_should: list[dict[str, Any]] = []

        if decision.use_deal_teams and decision.deal_team_memos:
            access_should.append({"terms": {"memoId": decision.deal_team_memos}})

        if decision.use_coverage and decision.coverage_filter:
            coverage_must = [
                {"terms": {field: values}}
                for field, values in decision.coverage_filter.items()
                if values
            ]
            if coverage_must:
                access_should.append({"bool": {"must": coverage_must}})

        must_clauses: list[dict[str, Any]] = []

        if access_should:
            must_clauses.append({
                "bool": {
                    "should": access_should,
                    "minimum_should_match": 1,
                }
            })

        # Role-level filters are ANDed on top
        for f in decision.role_filters:
            must_clauses.append(f)

        if not must_clauses:
            return None

        if len(must_clauses) == 1:
            return must_clauses[0]

        return {"bool": {"must": must_clauses}}
