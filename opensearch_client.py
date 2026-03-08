from __future__ import annotations

from typing import Any

from opensearchpy import OpenSearch, RequestsHttpConnection
from pydantic import BaseModel, Field


class Criteria(BaseModel):
    industry: list[str] = Field(default_factory=list)
    sector: list[str] = Field(default_factory=list)
    region: list[str] = Field(default_factory=list)
    currency: list[str] = Field(default_factory=list)


class TextQuery(BaseModel):
    query: str
    fields: list[str]
    operator: str = "or"


class SearchRequest(BaseModel):
    criteria: Criteria = Field(default_factory=Criteria)
    should_text: list[TextQuery] = Field(default_factory=list)
    minimum_should_match: int = 0
    size: int = 10


class MemoOpenSearchClient:
    """Minimal OpenSearch wrapper for the memo MCP server."""

    def __init__(
        self,
        hosts: list[dict[str, Any]],
        index_name: str = "memos",
        use_ssl: bool = False,
        verify_certs: bool = False,
        timeout: int = 30,
    ):
        self.index = index_name
        self.client = OpenSearch(
            hosts=hosts,
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            connection_class=RequestsHttpConnection,
            timeout=timeout,
        )

    def get_mapping(self) -> dict[str, Any]:
        return self.client.indices.get_mapping(index=self.index)

    def build_bool_query(self, request: dict[str, Any]) -> dict[str, Any]:
        filters: list[dict[str, Any]] = []
        criteria = request.get("criteria") or {}
        for field in ("industry", "sector", "region", "currency"):
            values = criteria.get(field) or []
            if values:
                filters.append({"terms": {field: values}})

        should = []
        for text_query in request.get("should_text", []) or []:
            should.append({
                "multi_match": {
                    "query": text_query["query"],
                    "fields": text_query["fields"],
                    "operator": text_query.get("operator", "or"),
                }
            })

        minimum_should_match = request.get("minimum_should_match", 0) if should else 0

        return {
            "bool": {
                "filter": filters,
                "should": should,
                "minimum_should_match": minimum_should_match,
            }
        }

    def raw_search(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.client.search(index=self.index, body=body)
