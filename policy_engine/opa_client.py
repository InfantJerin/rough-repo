from __future__ import annotations

from typing import Any

import httpx


class OPAClient:
    """Thin REST client for Open Policy Agent."""

    def __init__(self, base_url: str = "http://localhost:8181"):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=10.0)

    # ----- core query --------------------------------------------------

    def query(self, path: str, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        POST /v1/data/<path> with the given input.
        Returns the 'result' key from the OPA response.
        """
        url = f"/v1/data/{path}"
        resp = self._http.post(url, json={"input": input_data})
        resp.raise_for_status()
        body = resp.json()
        return body.get("result", {})

    # ----- convenience -------------------------------------------------

    def evaluate_memo_authz(self, user_id: str, memo_ids: list[str] | None = None) -> dict[str, Any]:
        """
        Evaluate memo.authz.decision for a user.
        Returns the full decision dict from Rego.
        """
        input_data: dict[str, Any] = {"user_id": user_id}
        if memo_ids is not None:
            input_data["memo_ids"] = memo_ids
        return self.query("memo/authz/decision", input_data)

    # ----- health / management -----------------------------------------

    def health(self) -> bool:
        try:
            resp = self._http.get("/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
