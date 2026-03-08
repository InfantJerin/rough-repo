from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UserContext:
    user_id: str


@dataclass
class PolicyDecision:
    user_id: str
    allow: bool
    user_role: str
    # DLS
    coverage_filter: dict[str, list[str]] = field(default_factory=dict)
    use_deal_teams: bool = False
    use_coverage: bool = False
    role_filters: list[dict[str, Any]] = field(default_factory=list)
    unrestricted: bool = False
    # FLS
    allowed_fields: list[str] | None = None
