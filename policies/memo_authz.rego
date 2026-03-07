package memo.authz

import future.keywords.in
import future.keywords.contains
import future.keywords.if

default allow := false

# -----------------------------------------------------------
# Helpers: resolve user profile and role definition
# -----------------------------------------------------------

user := data.memo.authz.users[input.user_id]
role_def := data.memo.authz.roles[user.role]

# -----------------------------------------------------------
# DOCUMENT-LEVEL ACCESS
# -----------------------------------------------------------

# 1. Unrestricted roles (e.g. compliance) can see everything
allow if {
    role_def.document_access.unrestricted == true
}

# 2. Deal-team membership: user is explicitly staffed on the memo
allow if {
    role_def.document_access.use_deal_teams == true
    some memo_id in input.memo_ids
    memo_id in data.memo.authz.deal_teams
    input.user_id in data.memo.authz.deal_teams[memo_id]
}

# 3. Coverage match: user's coverage overlaps with the memo attributes
allow if {
    role_def.document_access.use_coverage == true
}

# -----------------------------------------------------------
# Document filter — returns the OpenSearch filter to inject
# -----------------------------------------------------------

# For unrestricted roles: no filter needed
document_filter := {} if {
    role_def.document_access.unrestricted == true
}

# For roles with deal teams + coverage
document_filter := filter if {
    not role_def.document_access.unrestricted

    # Collect deal team memos for this user
    deal_memos := {memo_id |
        some memo_id, members in data.memo.authz.deal_teams
        input.user_id in members
    }

    # Build coverage filter if applicable
    coverage := _coverage_filter

    # Build role-level extra filters
    role_filters := role_def.document_access.filters

    filter := {
        "deal_team_memos": deal_memos,
        "coverage_filter": coverage,
        "use_deal_teams": role_def.document_access.use_deal_teams,
        "use_coverage": role_def.document_access.use_coverage,
        "role_filters": role_filters,
    }
}

# Build coverage filter from user attributes
_coverage_filter := coverage if {
    role_def.document_access.use_coverage == true
    coverage := {field: values |
        some field in ["region", "industry"]
        values := user.coverage[field]
        count(values) > 0
    }
}

_coverage_filter := {} if {
    not role_def.document_access.use_coverage
}

# -----------------------------------------------------------
# FIELD-LEVEL ACCESS
# -----------------------------------------------------------

# Return the list of fields the user is allowed to see
allowed_fields contains field if {
    role_def.field_access.mode == "allow_all"
    some field in data.memo.authz.all_fields
}

allowed_fields contains field if {
    role_def.field_access.mode == "allow_list"
    some field in role_def.field_access.allowed_fields
}

allowed_fields contains field if {
    role_def.field_access.mode == "deny_list"
    some field in data.memo.authz.all_fields
    not field in role_def.field_access.denied_fields
}

# -----------------------------------------------------------
# Combined decision — single query endpoint
# -----------------------------------------------------------

decision := {
    "allow": allow,
    "document_filter": document_filter,
    "allowed_fields": allowed_fields,
    "user_role": user.role,
}
