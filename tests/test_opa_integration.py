"""
Integration tests that run against a live OPA server.

These tests require OPA to be running (docker-compose up opa) with the
policies loaded. Skip these in CI unless OPA is available.

Run with:
    pytest tests/test_opa_integration.py -v
"""
from __future__ import annotations

import pytest

from policy_engine.opa_client import OPAClient
from policy_engine.engine import PolicyEngine
from policy_engine.models import UserContext


@pytest.fixture(scope="module")
def opa_client():
    client = OPAClient(base_url="http://localhost:8181")
    if not client.health():
        pytest.skip("OPA server is not running")
    yield client
    client.close()


@pytest.fixture(scope="module")
def engine(opa_client):
    return PolicyEngine(opa_client)


# -------------------------------------------------------------------
# Compliance user — unrestricted
# -------------------------------------------------------------------
class TestComplianceOPA:
    def test_compliance_allow(self, engine):
        decision = engine.evaluate(UserContext(user_id="compliance_01"))
        assert decision.allow is True
        assert decision.unrestricted is True
        assert decision.user_role == "compliance"

    def test_compliance_all_fields(self, engine):
        decision = engine.evaluate(UserContext(user_id="compliance_01"))
        assert "proposedCommitments" in decision.allowed_fields
        assert "lendingThesis" in decision.allowed_fields
        assert "riskFactors" in decision.allowed_fields

    def test_compliance_no_filter(self, engine):
        query = {"query": {"bool": {"must": [{"match_all": {}}], "filter": []}}, "size": 10}
        result = engine.apply_to_query(query, UserContext(user_id="compliance_01"))
        # No security filter should be added
        assert result["query"]["bool"]["filter"] == []


# -------------------------------------------------------------------
# Deal team banker — deal teams + coverage
# -------------------------------------------------------------------
class TestDealTeamBankerOPA:
    def test_banker_01_allow(self, engine):
        decision = engine.evaluate(UserContext(user_id="banker_01"))
        assert decision.allow is True
        assert decision.user_role == "deal_team_banker"

    def test_banker_01_deal_team_memos(self, engine):
        decision = engine.evaluate(UserContext(user_id="banker_01"))
        assert decision.use_deal_teams is True
        # banker_01 is in deal teams for these memos (from data.json)
        expected_memos = {
            "MEMO-001", "MEMO-003", "MEMO-005", "MEMO-006", "MEMO-009",
            "MEMO-010", "MEMO-011", "MEMO-013", "MEMO-015", "MEMO-017",
            "MEMO-019", "MEMO-020",
        }
        assert set(decision.deal_team_memos) == expected_memos

    def test_banker_01_coverage(self, engine):
        decision = engine.evaluate(UserContext(user_id="banker_01"))
        assert decision.use_coverage is True
        assert "India" in decision.coverage_filter.get("region", [])
        assert "Industrials" in decision.coverage_filter.get("industry", [])

    def test_banker_01_query_injection(self, engine):
        query = {"query": {"bool": {"should": [{"match": {"riskFactors": "supply chain"}}], "filter": []}}, "size": 10}
        result = engine.apply_to_query(query, UserContext(user_id="banker_01"))
        filters = result["query"]["bool"]["filter"]
        assert len(filters) == 1
        sec_filter = filters[0]
        assert "bool" in sec_filter
        should = sec_filter["bool"]["should"]
        assert len(should) == 2  # deal team + coverage

    def test_banker_02_different_deal_teams(self, engine):
        decision = engine.evaluate(UserContext(user_id="banker_02"))
        # banker_02 has different deal team memos
        assert "MEMO-001" in decision.deal_team_memos  # shared with banker_01
        assert "MEMO-002" in decision.deal_team_memos
        assert "MEMO-003" not in decision.deal_team_memos  # only banker_01

    def test_banker_all_fields(self, engine):
        decision = engine.evaluate(UserContext(user_id="banker_01"))
        assert "proposedCommitments" in decision.allowed_fields
        assert "lendingThesis" in decision.allowed_fields


# -------------------------------------------------------------------
# Credit officer — coverage only
# -------------------------------------------------------------------
class TestCreditOfficerOPA:
    def test_credit_officer_allow(self, engine):
        decision = engine.evaluate(UserContext(user_id="credit_officer_01"))
        assert decision.allow is True
        assert decision.user_role == "credit_officer"

    def test_credit_officer_no_deal_teams(self, engine):
        decision = engine.evaluate(UserContext(user_id="credit_officer_01"))
        assert decision.use_deal_teams is False
        assert decision.deal_team_memos == []

    def test_credit_officer_coverage_filter(self, engine):
        decision = engine.evaluate(UserContext(user_id="credit_officer_01"))
        assert decision.use_coverage is True
        assert "Materials" in decision.coverage_filter.get("industry", [])

    def test_credit_officer_query_has_coverage_only(self, engine):
        query = {"query": {"bool": {"must": [{"match_all": {}}], "filter": []}}, "size": 10}
        result = engine.apply_to_query(query, UserContext(user_id="credit_officer_01"))
        filters = result["query"]["bool"]["filter"]
        assert len(filters) == 1
        should = filters[0]["bool"]["should"]
        # Only coverage, no deal teams
        assert len(should) == 1


# -------------------------------------------------------------------
# Derivatives desk — role filters + denied fields
# -------------------------------------------------------------------
class TestDerivativesDeskOPA:
    def test_deriv_allow(self, engine):
        decision = engine.evaluate(UserContext(user_id="deriv_user_01"))
        assert decision.allow is True
        assert decision.user_role == "derivatives_desk"

    def test_deriv_role_filters(self, engine):
        decision = engine.evaluate(UserContext(user_id="deriv_user_01"))
        assert len(decision.role_filters) == 3
        # Check the filters match what's in data.json
        filter_types = [list(f.keys())[0] for f in decision.role_filters]
        assert "range" in filter_types
        assert "term" in filter_types
        assert "exists" in filter_types

    def test_deriv_denied_fields(self, engine):
        decision = engine.evaluate(UserContext(user_id="deriv_user_01"))
        assert "proposedCommitments" not in decision.allowed_fields
        assert "lendingThesis" not in decision.allowed_fields
        assert "environmentalRisks" not in decision.allowed_fields

    def test_deriv_allowed_fields(self, engine):
        decision = engine.evaluate(UserContext(user_id="deriv_user_01"))
        assert "riskFactors" in decision.allowed_fields
        assert "keyCommitteeDiscussionPoints" in decision.allowed_fields
        assert "businessDescription" in decision.allowed_fields

    def test_deriv_query_injection(self, engine):
        query = {"query": {"bool": {"should": [{"match": {"riskFactors": "risk"}}], "filter": []}}, "size": 10}
        result = engine.apply_to_query(query, UserContext(user_id="deriv_user_01"))

        # DLS: 3 role filters injected
        filters = result["query"]["bool"]["filter"]
        assert len(filters) == 3

        # FLS: source includes set
        includes = result["_source"]["includes"]
        assert "proposedCommitments" not in includes
        assert "riskFactors" in includes


# -------------------------------------------------------------------
# Unknown user — should be denied
# -------------------------------------------------------------------
class TestUnknownUserOPA:
    def test_unknown_user_denied(self, engine):
        """An unknown user_id should not get access."""
        raw = engine.opa.evaluate_memo_authz("nonexistent_user_999")
        # OPA should return allow=false or empty result
        assert raw.get("allow", False) is False
