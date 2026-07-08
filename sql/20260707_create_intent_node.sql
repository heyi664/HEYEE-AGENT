CREATE TABLE IF NOT EXISTS t_intent_node (
    id                    VARCHAR(20) PRIMARY KEY,
    kb_id                 VARCHAR(20),
    intent_code           VARCHAR(64) NOT NULL UNIQUE,
    name                  VARCHAR(64) NOT NULL,
    level                 SMALLINT NOT NULL,
    parent_code           VARCHAR(64),
    description           VARCHAR(512),
    examples              TEXT,
    collection_name       VARCHAR(128),
    top_k                 INTEGER,
    mcp_tool_id           VARCHAR(128),
    kind                  SMALLINT NOT NULL DEFAULT 0,
    prompt_snippet        TEXT,
    prompt_template       TEXT,
    param_prompt_template TEXT,
    sort_order            INTEGER NOT NULL DEFAULT 0,
    enabled               SMALLINT NOT NULL DEFAULT 1,
    created_by            VARCHAR(20),
    updated_by            VARCHAR(20),
    create_time           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted               SMALLINT NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_intent_node_parent_code ON t_intent_node(parent_code);
CREATE INDEX IF NOT EXISTS idx_intent_node_enabled_deleted ON t_intent_node(enabled, deleted);
CREATE INDEX IF NOT EXISTS idx_intent_node_kind ON t_intent_node(kind);
