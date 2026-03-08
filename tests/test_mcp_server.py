from __future__ import annotations

import json
from unittest.mock import MagicMock

from mcp_server import server as server_module
from policy_engine.models import PolicyDecision


def test_get_memo_uses_policy_enforced_search(monkeypatch):
    decision = PolicyDecision(
        user_id="deriv_user_01",
        allow=True,
        user_role="derivatives_desk",
        role_filters=[{"term": {"clientType": "private"}}],
        unrestricted=False,
        allowed_fields=["memoId", "riskFactors"],
    )
    mock_policy_engine = MagicMock()
    mock_policy_engine.evaluate.return_value = decision
    mock_policy_engine.apply_decision_to_query.side_effect = (
        server_module.policy_engine.apply_decision_to_query
    )

    mock_os_client = MagicMock()
    mock_os_client.raw_search.return_value = {
        "hits": {
            "hits": [
                {
                    "_source": {
                        "memoId": "MEMO-001",
                        "riskFactors": "visible",
                    }
                }
            ]
        }
    }

    monkeypatch.setattr(server_module, "policy_engine", mock_policy_engine)
    monkeypatch.setattr(server_module, "os_client", mock_os_client)

    raw = server_module.get_memo("deriv_user_01", "MEMO-001")
    payload = json.loads(raw)

    assert payload == {
        "memo_id": "MEMO-001",
        "data": {
            "memoId": "MEMO-001",
            "riskFactors": "visible",
        },
    }

    query_body = mock_os_client.raw_search.call_args.args[0]
    assert query_body["query"]["bool"]["must"] == [{"term": {"memoId": "MEMO-001"}}]
    assert query_body["query"]["bool"]["filter"] == [{"term": {"clientType": "private"}}]
    assert query_body["_source"]["includes"] == ["memoId", "riskFactors"]


def test_get_memo_returns_generic_not_found_or_denied(monkeypatch):
    decision = PolicyDecision(
        user_id="banker_01",
        allow=True,
        user_role="deal_team_banker",
        unrestricted=False,
        allowed_fields=["memoId"],
    )
    mock_policy_engine = MagicMock()
    mock_policy_engine.evaluate.return_value = decision
    mock_policy_engine.apply_decision_to_query.side_effect = (
        server_module.policy_engine.apply_decision_to_query
    )

    mock_os_client = MagicMock()
    mock_os_client.raw_search.return_value = {"hits": {"hits": []}}

    monkeypatch.setattr(server_module, "policy_engine", mock_policy_engine)
    monkeypatch.setattr(server_module, "os_client", mock_os_client)

    raw = server_module.get_memo("banker_01", "MEMO-404")
    payload = json.loads(raw)

    assert payload == {"error": "Memo not found or access denied"}


def test_get_schema_honors_empty_allowed_fields(monkeypatch):
    decision = PolicyDecision(
        user_id="restricted_user",
        allow=True,
        user_role="restricted_role",
        unrestricted=False,
        allowed_fields=[],
    )
    mock_policy_engine = MagicMock()
    mock_policy_engine.evaluate.return_value = decision

    mock_os_client = MagicMock()
    mock_os_client.get_mapping.return_value = {
        server_module.OPENSEARCH_INDEX: {
            "mappings": {
                "properties": {
                    "memoId": {"type": "keyword"},
                    "riskFactors": {"type": "text"},
                }
            }
        }
    }

    monkeypatch.setattr(server_module, "policy_engine", mock_policy_engine)
    monkeypatch.setattr(server_module, "os_client", mock_os_client)

    raw = server_module.get_schema("restricted_user")
    payload = json.loads(raw)

    assert payload["fields"] == {}
