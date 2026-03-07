"""
Unit tests for the policy engine.

These tests mock the OPA client so they run without OPA or OpenSearch.
They verify that the policy engine correctly translates OPA decisions
into OpenSearch query modifications.
"""
from __future__ import annotations

import copy
import json
import pytest
from unittest.mock import MagicMock

from policy_engine.engine import PolicyEngine
from policy_engine.models import UserContext, PolicyDecision
from policy_engine.opa_client import OPAClient


# ---------------------------------------------------------------------------
# Fixtures: simulated OPA responses for each persona
# ---------------------------------------------------------------------------

ALL_FIELDS = [
    "memoId", "clientName", "clientID", "industry", "sector", "region",
    "currency", "businessDescription", "executiveSummary", "proposedCommitments",
    "riskFactors", "lendingThesis", "environmentalRisks",
    "keyCommitteeDiscussionPoints",
]

DERIV_ALLOWED_FIELDS = [
    "memoId", "clientName", "clientID", "industry", "sector", "region",
    "currency", "businessDescription", "executiveSummary",
    "riskFactors", "keyCommitteeDiscussionPoints",
]

DERIV_DENIED_FIELDS = ["proposedCommitments", "lendingThesis", "environmentalRisks"]


def _opa_response_compliance():
    """Compliance: unrestricted document access, all fields."""
    return {
        "allow": True,
        "user_role": "compliance",
        "document_filter": {},
        "allowed_fields": ALL_FIELDS,
    }


def _opa_response_deal_team_banker():
    """Deal team banker: deal team memos + coverage, all fields."""
    return {
        "allow": True,
        "user_role": "deal_team_banker",
        "document_filter": {
            "deal_team_memos": ["MEMO-001", "MEMO-003", "MEMO-005"],
            "coverage_filter": {
                "region": ["India"],
                "industry": ["Industrials", "Energy"],
            },
            "use_deal_teams": True,
            "use_coverage": True,
            "role_filters": [],
        },
        "allowed_fields": ALL_FIELDS,
    }


def _opa_response_credit_officer():
    """Credit officer: coverage-based access, all fields."""
    return {
        "allow": True,
        "user_role": "credit_officer",
        "document_filter": {
            "deal_team_memos": [],
            "coverage_filter": {
                "region": ["India"],
                "industry": ["Industrials", "Energy", "Materials"],
            },
            "use_deal_teams": False,
            "use_coverage": True,
            "role_filters": [],
        },
        "allowed_fields": ALL_FIELDS,
    }


def _opa_response_derivatives_desk():
    """Derivatives desk: role-level filters + denied fields."""
    return {
        "allow": True,
        "user_role": "derivatives_desk",
        "document_filter": {
            "deal_team_memos": [],
            "coverage_filter": {},
            "use_deal_teams": False,
            "use_coverage": False,
            "role_filters": [
                {"range": {"commitmentDate": {"lte": "now-6M/d"}}},
                {"term": {"clientType": "private"}},
                {"exists": {"field": "sponsors"}},
            ],
        },
        "allowed_fields": DERIV_ALLOWED_FIELDS,
    }


def _make_engine(opa_response: dict) -> PolicyEngine:
    """Create a PolicyEngine with a mocked OPA client."""
    mock_opa = MagicMock(spec=OPAClient)
    mock_opa.evaluate_memo_authz.return_value = opa_response
    return PolicyEngine(mock_opa)


def _base_query() -> dict:
    """A simple search query body."""
    return {
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": "supply chain risks",
                            "fields": ["riskFactors", "keyCommitteeDiscussionPoints"],
                        }
                    }
                ],
                "filter": [],
                "minimum_should_match": 1,
            }
        },
        "size": 10,
    }


# ===================================================================
# TEST: PolicyDecision parsing
# ===================================================================

class TestDecisionParsing:
    def test_compliance_is_unrestricted(self):
        engine = _make_engine(_opa_response_compliance())
        decision = engine.evaluate(UserContext(user_id="compliance_01"))
        assert decision.allow is True
        assert decision.unrestricted is True
        assert decision.user_role == "compliance"
        assert set(decision.allowed_fields) == set(ALL_FIELDS)

    def test_deal_team_banker_parsed(self):
        engine = _make_engine(_opa_response_deal_team_banker())
        decision = engine.evaluate(UserContext(user_id="banker_01"))
        assert decision.allow is True
        assert decision.unrestricted is False
        assert decision.use_deal_teams is True
        assert decision.use_coverage is True
        assert "MEMO-001" in decision.deal_team_memos
        assert decision.coverage_filter["region"] == ["India"]

    def test_derivatives_has_role_filters(self):
        engine = _make_engine(_opa_response_derivatives_desk())
        decision = engine.evaluate(UserContext(user_id="deriv_user_01"))
        assert decision.unrestricted is False
        assert len(decision.role_filters) == 3
        assert decision.role_filters[0] == {"range": {"commitmentDate": {"lte": "now-6M/d"}}}

    def test_derivatives_denied_fields(self):
        engine = _make_engine(_opa_response_derivatives_desk())
        decision = engine.evaluate(UserContext(user_id="deriv_user_01"))
        for field in DERIV_DENIED_FIELDS:
            assert field not in decision.allowed_fields
        for field in DERIV_ALLOWED_FIELDS:
            assert field in decision.allowed_fields


# ===================================================================
# TEST: DLS — query injection
# ===================================================================

class TestDocumentLevelSecurity:
    def test_compliance_no_filter_injected(self):
        """Unrestricted role should not add any security filter."""
        engine = _make_engine(_opa_response_compliance())
        result = engine.apply_to_query(_base_query(), UserContext(user_id="compliance_01"))
        filters = result["query"]["bool"]["filter"]
        assert filters == []  # original empty list, nothing added

    def test_deal_team_banker_filter_has_deal_memos_and_coverage(self):
        """Deal team banker gets deal_team OR coverage filter."""
        engine = _make_engine(_opa_response_deal_team_banker())
        result = engine.apply_to_query(_base_query(), UserContext(user_id="banker_01"))
        filters = result["query"]["bool"]["filter"]
        assert len(filters) == 1

        sec_filter = filters[0]
        # Should be a bool with should clauses
        should = sec_filter["bool"]["should"]
        assert len(should) == 2  # deal team terms + coverage bool

        # First should: deal team memos
        deal_terms = should[0]
        assert "terms" in deal_terms
        assert "memoId" in deal_terms["terms"]
        assert set(deal_terms["terms"]["memoId"]) == {"MEMO-001", "MEMO-003", "MEMO-005"}

        # Second should: coverage
        coverage = should[1]
        assert "bool" in coverage
        coverage_must = coverage["bool"]["must"]
        # Should have region and industry terms
        field_names = {list(clause["terms"].keys())[0] for clause in coverage_must}
        assert field_names == {"region", "industry"}

    def test_credit_officer_coverage_only(self):
        """Credit officer has coverage filter but no deal team memos."""
        engine = _make_engine(_opa_response_credit_officer())
        result = engine.apply_to_query(_base_query(), UserContext(user_id="credit_officer_01"))
        filters = result["query"]["bool"]["filter"]
        assert len(filters) == 1

        sec_filter = filters[0]
        should = sec_filter["bool"]["should"]
        # Only coverage, no deal team memos (empty list)
        assert len(should) == 1
        coverage = should[0]
        assert "bool" in coverage

    def test_derivatives_role_filters_injected(self):
        """Derivatives desk gets role-level filters (range, term, exists)."""
        engine = _make_engine(_opa_response_derivatives_desk())
        result = engine.apply_to_query(_base_query(), UserContext(user_id="deriv_user_01"))
        filters = result["query"]["bool"]["filter"]

        # Should have 3 role filters (no deal team, no coverage)
        assert len(filters) == 3
        assert filters[0] == {"range": {"commitmentDate": {"lte": "now-6M/d"}}}
        assert filters[1] == {"term": {"clientType": "private"}}
        assert filters[2] == {"exists": {"field": "sponsors"}}

    def test_original_query_not_mutated(self):
        """Applying policy should not mutate the original query body."""
        engine = _make_engine(_opa_response_deal_team_banker())
        original = _base_query()
        original_copy = copy.deepcopy(original)
        engine.apply_to_query(original, UserContext(user_id="banker_01"))
        assert original == original_copy


# ===================================================================
# TEST: FLS — field masking
# ===================================================================

class TestFieldLevelSecurity:
    def test_compliance_sees_all_fields(self):
        """Compliance gets _source.includes with all fields."""
        engine = _make_engine(_opa_response_compliance())
        result = engine.apply_to_query(_base_query(), UserContext(user_id="compliance_01"))
        includes = result["_source"]["includes"]
        assert set(includes) == set(ALL_FIELDS)

    def test_derivatives_excluded_fields(self):
        """Derivatives desk should not see denied fields."""
        engine = _make_engine(_opa_response_derivatives_desk())
        result = engine.apply_to_query(_base_query(), UserContext(user_id="deriv_user_01"))
        includes = result["_source"]["includes"]
        for field in DERIV_DENIED_FIELDS:
            assert field not in includes
        for field in DERIV_ALLOWED_FIELDS:
            assert field in includes

    def test_filter_fields_strips_denied(self):
        """filter_fields should strip fields from a document dict."""
        engine = _make_engine(_opa_response_derivatives_desk())
        decision = engine.evaluate(UserContext(user_id="deriv_user_01"))

        doc = {
            "memoId": "MEMO-001",
            "clientName": "Test Corp",
            "proposedCommitments": "SECRET",
            "lendingThesis": "SECRET",
            "environmentalRisks": "SECRET",
            "riskFactors": "visible",
        }
        filtered = engine.filter_fields(doc, decision)
        assert "proposedCommitments" not in filtered
        assert "lendingThesis" not in filtered
        assert "environmentalRisks" not in filtered
        assert filtered["riskFactors"] == "visible"
        assert filtered["memoId"] == "MEMO-001"


# ===================================================================
# TEST: Document access checks (for get_memo)
# ===================================================================

class TestDocumentAccessCheck:
    def test_compliance_can_access_any_memo(self):
        engine = _make_engine(_opa_response_compliance())
        decision = engine.evaluate(UserContext(user_id="compliance_01"))
        doc = {"region": "US", "industry": "Financials"}
        assert engine.check_document_access(decision, "MEMO-999", doc) is True

    def test_banker_can_access_deal_team_memo(self):
        engine = _make_engine(_opa_response_deal_team_banker())
        decision = engine.evaluate(UserContext(user_id="banker_01"))
        doc = {"region": "US", "industry": "Financials"}  # outside coverage
        # But MEMO-001 is in deal team, so access is granted
        assert engine.check_document_access(decision, "MEMO-001", doc) is True

    def test_banker_can_access_coverage_memo(self):
        engine = _make_engine(_opa_response_deal_team_banker())
        decision = engine.evaluate(UserContext(user_id="banker_01"))
        doc = {"region": "India", "industry": "Industrials"}
        assert engine.check_document_access(decision, "MEMO-999", doc) is True

    def test_banker_denied_outside_coverage_and_deal_team(self):
        engine = _make_engine(_opa_response_deal_team_banker())
        decision = engine.evaluate(UserContext(user_id="banker_01"))
        doc = {"region": "US", "industry": "Financials"}
        assert engine.check_document_access(decision, "MEMO-999", doc) is False

    def test_credit_officer_coverage_match(self):
        engine = _make_engine(_opa_response_credit_officer())
        decision = engine.evaluate(UserContext(user_id="credit_officer_01"))
        doc = {"region": "India", "industry": "Materials"}
        assert engine.check_document_access(decision, "MEMO-013", doc) is True

    def test_credit_officer_coverage_mismatch(self):
        engine = _make_engine(_opa_response_credit_officer())
        decision = engine.evaluate(UserContext(user_id="credit_officer_01"))
        doc = {"region": "US", "industry": "Technology"}
        assert engine.check_document_access(decision, "MEMO-012", doc) is False


# ===================================================================
# TEST: Extensibility — combining DLS + FLS
# ===================================================================

class TestCombinedDLSFLS:
    def test_derivatives_full_pipeline(self):
        """
        Derivatives desk: role filters injected (DLS) + fields masked (FLS).
        Simulates the full query transformation pipeline.
        """
        engine = _make_engine(_opa_response_derivatives_desk())
        query = _base_query()
        result = engine.apply_to_query(query, UserContext(user_id="deriv_user_01"))

        # DLS: 3 role filters
        filters = result["query"]["bool"]["filter"]
        assert len(filters) == 3

        # FLS: denied fields excluded
        includes = result["_source"]["includes"]
        assert "proposedCommitments" not in includes
        assert "lendingThesis" not in includes

        # But allowed fields are present
        assert "riskFactors" in includes
        assert "keyCommitteeDiscussionPoints" in includes

    def test_banker_full_pipeline(self):
        """
        Deal team banker: deal team + coverage filters (DLS) + all fields (FLS).
        """
        engine = _make_engine(_opa_response_deal_team_banker())
        query = _base_query()
        result = engine.apply_to_query(query, UserContext(user_id="banker_01"))

        # DLS: 1 filter with should (deal team OR coverage)
        filters = result["query"]["bool"]["filter"]
        assert len(filters) == 1

        # FLS: all fields
        includes = result["_source"]["includes"]
        assert set(includes) == set(ALL_FIELDS)


# ===================================================================
# TEST: Edge cases
# ===================================================================

class TestEdgeCases:
    def test_query_with_existing_filter_dict(self):
        """If the query already has a filter as a dict, it should be normalized."""
        engine = _make_engine(_opa_response_deal_team_banker())
        query = {
            "query": {
                "bool": {
                    "must": [{"match_all": {}}],
                    "filter": {"term": {"status": "active"}},  # dict, not list
                }
            },
            "size": 10,
        }
        result = engine.apply_to_query(query, UserContext(user_id="banker_01"))
        filters = result["query"]["bool"]["filter"]
        assert isinstance(filters, list)
        assert len(filters) == 2  # original dict + security filter

    def test_empty_query_body(self):
        """Policy engine should handle a minimal query body."""
        engine = _make_engine(_opa_response_deal_team_banker())
        query = {"query": {"bool": {}}, "size": 5}
        result = engine.apply_to_query(query, UserContext(user_id="banker_01"))
        assert "filter" in result["query"]["bool"]
        assert "_source" in result

    def test_access_denied_decision(self):
        """If OPA says allow=false, the decision should reflect that."""
        denied_response = {
            "allow": False,
            "user_role": "",
            "document_filter": {},
            "allowed_fields": [],
        }
        engine = _make_engine(denied_response)
        decision = engine.evaluate(UserContext(user_id="unknown_user"))
        assert decision.allow is False
