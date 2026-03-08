"""
MCP Server that exposes OpenSearch memo tools with policy-based access control.

Usage:
    python -m mcp_server.server
"""
from __future__ import annotations

import json
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from opensearch_client import MemoOpenSearchClient, SearchRequest, Criteria, TextQuery
from policy_engine.engine import PolicyEngine
from policy_engine.opa_client import OPAClient
from policy_engine.models import UserContext

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "memo")
OPA_URL = os.getenv("OPA_URL", "http://localhost:8181")
MAX_SEARCH_SIZE = int(os.getenv("MAX_SEARCH_SIZE", "50"))

# ---------------------------------------------------------------------------
# Bootstrap clients
# ---------------------------------------------------------------------------
os_client = MemoOpenSearchClient(
    hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
    use_ssl=False,
    verify_certs=False,
    index_name=OPENSEARCH_INDEX,
)

opa_client = OPAClient(base_url=OPA_URL)
policy_engine = PolicyEngine(opa_client)

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "memo-search",
    instructions=(
        "This server provides tools for searching and retrieving banking credit memos. "
        "Use get_schema to understand what fields are available, list_facets to discover "
        "filterable values, and search_memos to find memos matching criteria. "
        "If the user's query is vague, ask clarifying questions about industry, region, "
        "or what type of information they're looking for before searching."
    ),
)

def _clamp_size(size: int) -> int:
    return max(1, min(size, MAX_SEARCH_SIZE))


# ---------------------------------------------------------------------------
# Tool: get_schema
# ---------------------------------------------------------------------------
@mcp.tool()
def get_schema(user_id: str) -> str:
    """
    Return the OpenSearch index mapping so the LLM knows which fields are
    searchable and their types. Respects field-level security — only fields
    the user is allowed to see are returned.
    """
    decision = policy_engine.evaluate(UserContext(user_id=user_id))
    if not decision.allow:
        return json.dumps({"error": "Access denied"})

    mapping = os_client.get_mapping()
    # Extract properties from the mapping
    index_mapping = mapping.get(OPENSEARCH_INDEX, mapping.get(list(mapping.keys())[0], {}))
    properties = index_mapping.get("mappings", {}).get("properties", {})

    # Filter to only allowed fields
    if decision.allowed_fields is not None:
        properties = {k: v for k, v in properties.items() if k in decision.allowed_fields}

    return json.dumps({
        "index": OPENSEARCH_INDEX,
        "fields": properties,
        "user_role": decision.user_role,
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool: list_facets
# ---------------------------------------------------------------------------
@mcp.tool()
def list_facets(
    user_id: str,
    facet_fields: list[str] | None = None,
) -> str:
    """
    Return the distinct values (facets) for filterable fields like industry,
    sector, region, currency. Helps the LLM understand what filter values
    are available. Results are scoped to the user's access.
    """
    decision = policy_engine.evaluate(UserContext(user_id=user_id))
    if not decision.allow:
        return json.dumps({"error": "Access denied"})

    facet_fields = facet_fields or ["industry", "sector", "region", "currency"]

    # Filter facet_fields to only allowed fields
    if decision.allowed_fields is not None:
        facet_fields = [f for f in facet_fields if f in decision.allowed_fields]

    # Build a match_all query and apply policy filters
    query_body: dict[str, Any] = {
        "query": {"bool": {"must": [{"match_all": {}}], "filter": []}},
        "size": 0,
    }
    query_body = policy_engine.apply_decision_to_query(query_body, decision)

    # Add aggregations
    query_body["aggs"] = {
        f"{f}_facet": {"terms": {"field": f, "size": 50}}
        for f in facet_fields
    }

    resp = os_client.raw_search(query_body)
    aggs = resp.get("aggregations", {})

    facets = {}
    for f in facet_fields:
        buckets = aggs.get(f"{f}_facet", {}).get("buckets", [])
        facets[f] = [{"value": b["key"], "count": b["doc_count"]} for b in buckets]

    return json.dumps(facets, indent=2)


# ---------------------------------------------------------------------------
# Tool: search_memos
# ---------------------------------------------------------------------------
@mcp.tool()
def search_memos(
    user_id: str,
    query: str | None = None,
    industry: list[str] | None = None,
    region: list[str] | None = None,
    sector: list[str] | None = None,
    currency: list[str] | None = None,
    size: int = 10,
) -> str:
    """
    Search memos by text query and/or structured filters.

    - query: free-text search across memo content (risk factors, executive
      summary, business description, etc.)
    - industry, region, sector, currency: optional filters to narrow results
    - size: number of results to return (default 10)

    Results are automatically scoped to the user's access permissions.
    Fields the user is not allowed to see are excluded from results.
    """
    decision = policy_engine.evaluate(UserContext(user_id=user_id))
    if not decision.allow:
        return json.dumps({"error": "Access denied"})

    size = _clamp_size(size)

    # Build the search request
    should_text = []
    if query:
        # Search across all text fields the user can see
        searchable_fields = [
            f for f in [
                "businessDescription", "executiveSummary", "riskFactors",
                "keyCommitteeDiscussionPoints", "lendingThesis",
                "environmentalRisks", "proposedCommitments",
            ] if decision.allowed_fields is None or f in decision.allowed_fields
        ]
        if searchable_fields:
            should_text.append(TextQuery(query=query, fields=searchable_fields))

    req = SearchRequest(
        criteria=Criteria(
            industry=industry or [],
            sector=sector or [],
            region=region or [],
            currency=currency or [],
        ),
        should_text=should_text,
        minimum_should_match=1 if should_text else 0,
        size=size,
    )

    query_body = os_client.build_bool_query(req.model_dump())
    full_body: dict[str, Any] = {
        "query": query_body,
        "size": size,
        "track_total_hits": True,
    }

    # Apply policy: inject DLS filters + FLS source mask
    full_body = policy_engine.apply_decision_to_query(full_body, decision)

    resp = os_client.raw_search(full_body)
    hits = resp.get("hits", {}).get("hits", [])
    total = resp.get("hits", {}).get("total", {})
    total_value = total.get("value") if isinstance(total, dict) else total

    results = []
    for h in hits:
        results.append({
            "memo_id": h["_id"],
            "score": h["_score"],
            "data": h["_source"],
        })

    return json.dumps({
        "total": total_value,
        "results": results,
        "query_used": full_body.get("query"),
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool: get_memo
# ---------------------------------------------------------------------------
@mcp.tool()
def get_memo(user_id: str, memo_id: str) -> str:
    """
    Retrieve a single memo by its ID. Access is checked against the user's
    permissions — if the user is not entitled to see this memo, access
    is denied. Fields the user cannot see are stripped from the response.
    """
    decision = policy_engine.evaluate(UserContext(user_id=user_id), memo_ids=[memo_id])

    if not decision.allow:
        return json.dumps({"error": "Access denied"})

    query_body: dict[str, Any] = {
        "query": {
            "term": {
                "memoId": memo_id,
            }
        },
        "size": 1,
    }
    query_body = policy_engine.apply_decision_to_query(query_body, decision)

    resp = os_client.raw_search(query_body)
    hits = resp.get("hits", {}).get("hits", [])
    if not hits:
        return json.dumps({"error": "Memo not found or access denied"})

    filtered = hits[0].get("_source", {})

    return json.dumps({
        "memo_id": memo_id,
        "data": filtered,
    }, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()
