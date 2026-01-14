import json
import os
import requests
from typing import List, Any


# ----------------------------
# Config
# ----------------------------
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
MEMO_INDEX = os.getenv("MEMO_INDEX", "memo")


# ----------------------------
# OpenSearch helpers
# ----------------------------
def os_request(method: str, path: str, body: Any = None) -> requests.Response:
    url = f"{OPENSEARCH_URL.rstrip('/')}/{path.lstrip('/')}"
    headers = {"Content-Type": "application/json"}
    data = json.dumps(body) if body is not None else None
    resp = requests.request(method, url, headers=headers, data=data, timeout=60)
    return resp


def ensure_index() -> None:
    """Create the memo index if it doesn't exist"""
    memo_mapping = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "memoId": {"type": "keyword"},
                "clientName": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "clientID": {"type": "keyword"},
                "industry": {"type": "keyword"},
                "sector": {"type": "keyword"},
                "region": {"type": "keyword"},
                "currency": {"type": "keyword"},
                "businessDescription": {"type": "text"},
                "executiveSummary": {"type": "text"},
                "proposedCommitments": {"type": "text"},
                "riskFactors": {"type": "text"},
                "lendingThesis": {"type": "text"},
                "environmentalRisks": {"type": "text"},
                "keyCommitteeDiscussionPoints": {"type": "text"},
            }
        },
    }

    r = os_request("HEAD", MEMO_INDEX)
    if r.status_code == 404:
        cr = os_request("PUT", MEMO_INDEX, memo_mapping)
        cr.raise_for_status()
        print(f"Created index: {MEMO_INDEX}")
    elif r.ok:
        print(f"Index exists: {MEMO_INDEX}")
    else:
        r.raise_for_status()


def bulk_index(ndjson_lines: List[str]) -> None:
    """Bulk index documents using NDJSON format"""
    url = f"{OPENSEARCH_URL.rstrip('/')}/_bulk"
    headers = {"Content-Type": "application/x-ndjson"}
    payload = "\n".join(ndjson_lines) + "\n"
    resp = requests.post(url, headers=headers, data=payload.encode("utf-8"), timeout=120)
    resp.raise_for_status()
    result = resp.json()
    if result.get("errors"):
        # Print first few errors to help debug
        items = result.get("items", [])
        errors = []
        for it in items:
            action = next(iter(it.values()))
            if "error" in action:
                errors.append(action["error"])
            if len(errors) >= 3:
                break
        raise RuntimeError(f"Bulk indexing had errors. Sample: {json.dumps(errors, indent=2)}")


def memo_to_bulk_lines(memo: dict) -> List[str]:
    """Convert a memo document to bulk index NDJSON lines"""
    lines: List[str] = []
    memo_id = memo["memoId"]
    
    # Index action + document
    lines.append(json.dumps({"index": {"_index": MEMO_INDEX, "_id": memo_id}}))
    lines.append(json.dumps(memo))
    
    return lines


def main() -> None:
    # Ensure index exists
    ensure_index()
    
    # Read memo.json
    memo_file = "memo.json"
    if not os.path.exists(memo_file):
        raise FileNotFoundError(f"File not found: {memo_file}")
    
    with open(memo_file, "r", encoding="utf-8") as f:
        memos = json.load(f)
    
    print(f"Loaded {len(memos)} memos from {memo_file}")
    
    # Convert all memos to bulk index format
    all_bulk_lines: List[str] = []
    for memo in memos:
        all_bulk_lines.extend(memo_to_bulk_lines(memo))
        print(f"Prepared {memo['memoId']} - {memo['clientName']}")
    
    # Bulk index all documents
    if all_bulk_lines:
        print(f"\nIndexing {len(memos)} documents to {MEMO_INDEX}...")
        bulk_index(all_bulk_lines)
        print(f"✅ Successfully indexed {len(memos)} documents")
        
        # Refresh the index to make documents searchable immediately
        refresh_resp = os_request("POST", f"{MEMO_INDEX}/_refresh")
        if refresh_resp.ok:
            print("✅ Index refreshed")
    
    # Verify count
    r = os_request("GET", f"{MEMO_INDEX}/_count")
    if r.ok:
        count = r.json().get("count", 0)
        print(f"✅ Index now contains {count} documents")


if __name__ == "__main__":
    main()

