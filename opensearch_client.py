from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal, Tuple, Iterable
from pydantic import BaseModel, Field

import os
import json

from opensearchpy import OpenSearch, RequestsHttpConnection, helpers
from opensearchpy.exceptions import NotFoundError


# --- Your memo document schema (matches mapping) ---

class MemoDoc(BaseModel):
    memoId: str
    clientName: Optional[str] = None
    clientID: Optional[str] = None

    industry: Optional[str] = None
    sector: Optional[str] = None
    region: Optional[str] = None
    currency: Optional[str] = None

    businessDescription: Optional[str] = None
    executiveSummary: Optional[str] = None
    proposedCommitments: Optional[str] = None
    riskFactors: Optional[str] = None
    lendingThesis: Optional[str] = None
    environmentalRisks: Optional[str] = None
    keyCommitteeDiscussionPoints: Optional[str] = None


# --- Filter criteria the agent can populate ---

class Criteria(BaseModel):
    industry: List[str] = Field(default_factory=list)
    sector: List[str] = Field(default_factory=list)
    region: List[str] = Field(default_factory=list)
    currency: List[str] = Field(default_factory=list)


# --- Text queries the agent can generate ---

class TextQuery(BaseModel):
    query: str
    # agent may specify where to search; default is a sensible set
    fields: List[str] = Field(default_factory=lambda: [
        "businessDescription",
        "executiveSummary",
        "riskFactors",
        "keyCommitteeDiscussionPoints",
        "lendingThesis",
        "environmentalRisks",
        "proposedCommitments",
    ])
    type: Literal["best_fields", "most_fields", "phrase", "phrase_prefix", "bool_prefix"] = "best_fields"
    operator: Literal["and", "or"] = "or"
    boost: float = 1.0


class SearchRequest(BaseModel):
    criteria: Criteria = Field(default_factory=Criteria)
    should_text: List[TextQuery] = Field(default_factory=list)

    # exact phrase queries can be added separately if you like
    must_text: List[TextQuery] = Field(default_factory=list)

    minimum_should_match: int = 1
    size: int = 10
    from_: int = Field(default=0, alias="from")
    track_total_hits: bool = True
    explain: bool = False

    # optional post-filtering/sorting knobs
    sort: Optional[List[Dict[str, Any]]] = None
    source_includes: Optional[List[str]] = None
    source_excludes: Optional[List[str]] = None


DEFAULT_FACET_FIELDS = ["industry", "sector", "region", "currency"]


class MemoOpenSearchClient:
    """
    Agent-friendly OpenSearch client for memos index.

    Core principles:
    - Keep low-level OpenSearch calls available (raw_* methods)
    - Provide high-level methods that accept structured request objects
    - Return both hits + useful metadata (total, facets, scroll ids, etc.)
    """

    def __init__(
        self,
        hosts: List[Dict[str, Any]],
        index_name: str = "memos",
        http_auth: Optional[Tuple[str, str]] = None,
        use_ssl: bool = False,
        verify_certs: bool = False,
        ssl_assert_hostname: bool = False,
        ssl_show_warn: bool = False,
        timeout: int = 30,
    ):
        self.index = index_name
        self.client = OpenSearch(
            hosts=hosts,
            http_auth=http_auth,
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            ssl_assert_hostname=ssl_assert_hostname,
            ssl_show_warn=ssl_show_warn,
            connection_class=RequestsHttpConnection,
            timeout=timeout,
        )

    # ---------------------------
    # Index management
    # ---------------------------

    def index_exists(self) -> bool:
        return self.client.indices.exists(self.index)

    def create_index(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self.client.indices.create(index=self.index, body=body)

    def get_mapping(self) -> Dict[str, Any]:
        return self.client.indices.get_mapping(index=self.index)

    def put_mapping(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self.client.indices.put_mapping(index=self.index, body=body)

    def refresh(self) -> Dict[str, Any]:
        return self.client.indices.refresh(index=self.index)

    # ---------------------------
    # CRUD
    # ---------------------------

    def upsert_memo(self, doc_id: str, doc: Dict[str, Any]) -> Dict[str, Any]:
        # doc_id could be memoId or another stable id
        return self.client.index(index=self.index, id=doc_id, body=doc, refresh=False)

    def bulk_upsert(self, docs: Iterable[Tuple[str, Dict[str, Any]]], chunk_size: int = 500) -> Dict[str, Any]:
        """
        docs: iterable of (doc_id, doc_dict)
        """
        def gen_actions():
            for doc_id, doc in docs:
                yield {
                    "_op_type": "index",
                    "_index": self.index,
                    "_id": doc_id,
                    "_source": doc,
                }

        success, errors = helpers.bulk(self.client, gen_actions(), chunk_size=chunk_size, raise_on_error=False)
        return {"success": success, "errors": errors}

    def get_memo(self, doc_id: str, source_includes: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        try:
            return self.client.get(index=self.index, id=doc_id, _source_includes=source_includes)
        except NotFoundError:
            return None

    def delete_memo(self, doc_id: str) -> Dict[str, Any]:
        return self.client.delete(index=self.index, id=doc_id, refresh=False)

    def partial_update(self, doc_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        return self.client.update(index=self.index, id=doc_id, body={"doc": fields}, refresh=False)

    # ---------------------------
    # Query builders (important for agent usage)
    # ---------------------------

    @staticmethod
    def _criteria_filters(criteria: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        filters: List[Dict[str, Any]] = []
        for field in ["industry", "sector", "region", "currency"]:
            vals = criteria.get(field) or []
            if vals:
                filters.append({"terms": {field: vals}})
        return filters

    @staticmethod
    def _multi_match_clause(tq: Dict[str, Any]) -> Dict[str, Any]:
        # tq has keys from TextQuery model
        clause = {
            "multi_match": {
                "query": tq["query"],
                "fields": tq["fields"],
                "type": tq.get("type", "best_fields"),
                "operator": tq.get("operator", "or"),
            }
        }
        boost = tq.get("boost")
        if boost is not None and boost != 1.0:
            clause["multi_match"]["boost"] = boost
        return clause

    def build_bool_query(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        request is SearchRequest.dict(by_alias=True)
        """
        criteria = (request.get("criteria") or {})
        filters = self._criteria_filters(criteria)

        must = []
        for tq in request.get("must_text", []) or []:
            must.append(self._multi_match_clause(tq))

        should = []
        for tq in request.get("should_text", []) or []:
            should.append(self._multi_match_clause(tq))

        minimum_should_match = request.get("minimum_should_match", 1)
        # If there are no should clauses, do not force MSM=1
        if not should:
            minimum_should_match = 0

        return {
            "bool": {
                "filter": filters,
                "must": must,
                "should": should,
                "minimum_should_match": minimum_should_match,
            }
        }

    # ---------------------------
    # Search (agent-friendly)
    # ---------------------------

    def search(
        self,
        request: Dict[str, Any],
        facets: bool = True,
        facet_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Returns: hits, total, (optional) aggregations, raw response
        """
        facet_fields = facet_fields or DEFAULT_FACET_FIELDS

        query = self.build_bool_query(request)

        body: Dict[str, Any] = {
            "query": query,
            "size": request.get("size", 10),
            "from": request.get("from_", request.get("from", 0)),
            "track_total_hits": request.get("track_total_hits", True),
        }

        if request.get("explain"):
            body["explain"] = True

        if request.get("sort"):
            body["sort"] = request["sort"]

        # Source filtering
        src_in = request.get("source_includes")
        src_ex = request.get("source_excludes")
        if src_in is not None or src_ex is not None:
            body["_source"] = {}
            if src_in is not None:
                body["_source"]["includes"] = src_in
            if src_ex is not None:
                body["_source"]["excludes"] = src_ex

        if facets:
            body["aggs"] = {
                f"{f}_facet": {"terms": {"field": f, "size": 50}}
                for f in facet_fields
            }

        #print(json.dumps(body, indent=2))
        resp = self.client.search(index=self.index, body=body)

        hits = resp.get("hits", {}).get("hits", [])
        total = resp.get("hits", {}).get("total", {})
        # total may be dict or int depending on settings/version
        total_value = total.get("value") if isinstance(total, dict) else total

        return {
            "total": total_value,
            "hits": hits,
            "aggregations": resp.get("aggregations", {}),
            "raw": resp,
            "query": body,  # helpful for debugging the agent
        }

    def agent_search(self, industry: list[str], region: list[str], currency: list[str], query: list[str], size: int = 10) -> Dict[str, Any]:
        """
        For use by agents; returns hits, total, (optional) aggregations, raw response
        """
        req = self.build_agent_request(industry, region, currency, query, size)
        return self.search(req.model_dump(by_alias=True), facets=True)

    def build_agent_request(self, industry: list[str] = [], region: list[str] = [], currency: list[str] = [], query: list[str] = [], size: int = 10) -> Dict[str, Any]:
        """
        Build a request for agent search
        """
        should_text = [ TextQuery(query=q, fields=["riskFactors", "keyCommitteeDiscussionPoints"]) for q in query ]
        req = SearchRequest(
        criteria=Criteria(
            industry=industry,
            region=region,
            currency=currency,
        ),
        should_text=should_text,
        minimum_should_match=1,
        size=size,
        explain=False,
    )
        return req
        
    # ---------------------------
    # Useful extras for agentic retrieval
    # ---------------------------

    def search_with_scroll(
        self,
        request: Dict[str, Any],
        scroll: str = "2m",
    ) -> Dict[str, Any]:
        """
        For large result sets; returns scroll_id + first page.
        """
        query = self.build_bool_query(request)
        body = {
            "query": query,
            "size": request.get("size", 100),
            "track_total_hits": request.get("track_total_hits", True),
        }
        resp = self.client.search(index=self.index, body=body, scroll=scroll)
        return {
            "scroll_id": resp.get("_scroll_id"),
            "hits": resp.get("hits", {}).get("hits", []),
            "raw": resp,
            "query": body,
        }

    def scroll_next(self, scroll_id: str, scroll: str = "2m") -> Dict[str, Any]:
        resp = self.client.scroll(scroll_id=scroll_id, scroll=scroll)
        return {
            "scroll_id": resp.get("_scroll_id"),
            "hits": resp.get("hits", {}).get("hits", []),
            "raw": resp,
        }

    def clear_scroll(self, scroll_id: str) -> Dict[str, Any]:
        return self.client.clear_scroll(body={"scroll_id": [scroll_id]})

    def msearch(self, bodies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Multi-search. `bodies` should already be in msearch format:
        [header, body, header, body, ...]
        """
        return self.client.msearch(body=bodies)

    def count(self, request: Dict[str, Any]) -> int:
        query = self.build_bool_query(request)
        resp = self.client.count(index=self.index, body={"query": query})
        return int(resp.get("count", 0))

    def raw_search(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Escape hatch for agent experiments.
        """
        return self.client.search(index=self.index, body=body)


if __name__ == "__main__":
    
    os_client = MemoOpenSearchClient(
    hosts=[{"host": "localhost", "port": 9200}],
    use_ssl=False,
    verify_certs=False,
    index_name="memo")

    # Build a request like your earlier payload
    req = SearchRequest(
        criteria=Criteria(
            industry=["Healthcare", "Industrials"],
            region=["India", "US"],
            currency=["INR", "USD"],
        ),
        should_text=[
            TextQuery(query="Fuel price movements",
                    fields=["riskFactors", "keyCommitteeDiscussionPoints"]),
            TextQuery(query="Customer concentration", fields=["riskFactors", "keyCommitteeDiscussionPoints"]),
            TextQuery(query="Supply chain disruptions", fields=["riskFactors", "keyCommitteeDiscussionPoints"]),
            TextQuery(query="Military escalation risks", fields=["riskFactors", "keyCommitteeDiscussionPoints"]),
            TextQuery(query="Energy supply chain concerns", fields=["riskFactors", "keyCommitteeDiscussionPoints"]),
            TextQuery(query="Cybersecurity vulnerabilities", fields=["riskFactors", "keyCommitteeDiscussionPoints"]),
        ],
        minimum_should_match=1,
        size=10,
        explain=False,
    )

    # req = SearchRequest(
    #     criteria = Criteria(
    #         industry=["Healthcare", "Industrials"],
    #         region=["India", "US"],
    #         currency=["INR", "USD"],
    #     ),
    #     should_text=[],
    #     minimum_should_match=1,
    #     size=10,
    #     explain=False,
    # )

    #resp = os_client.search(req.model_dump(by_alias=True), facets=True)
    resp = os_client.agent_search(industry=["Healthcare", "Industrials"], region=["India", "US"], currency=["INR", "USD"], query=["Fuel price movements", "Customer concentration", "Supply chain disruptions", "Military escalation risks", "Energy supply chain concerns", "Cybersecurity vulnerabilities"], size=10)
    
    print("Total:", resp["total"])
    for h in resp["hits"]:
        print(h["_id"], h["_score"], h["_source"].get("clientName"), h["_source"].get("industry"), h["_source"].get("region"), h["_source"].get("currency"))
