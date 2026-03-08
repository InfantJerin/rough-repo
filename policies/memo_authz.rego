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

# 2. Known users with role-based access are allowed to query; document filtering
# is enforced in OpenSearch using the returned role/coverage constraints.
allow if {
    user.role != ""
}

# -----------------------------------------------------------
# Document filter — returns the OpenSearch filter to inject
# -----------------------------------------------------------

# For unrestricted roles: no filter needed
document_filter := {} if {
    role_def.document_access.unrestricted == true
}

# For restricted roles, return the constraints needed for OpenSearch injection.
document_filter := filter if {
    not role_def.document_access.unrestricted

    coverage := _coverage_filter
    role_filters := role_def.document_access.filters

    filter := {
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
    "unrestricted": role_def.document_access.unrestricted,
    "user_role": user.role,
}
