BEGIN;

WITH seed (
    id,
    kb_id,
    intent_code,
    name,
    level,
    parent_code,
    description,
    examples,
    collection_name,
    top_k,
    mcp_tool_id,
    kind,
    sort_order,
    enabled,
    deleted
) AS (
    VALUES
        (
            'intent_domain_prod', NULL, 'product-service', '商品服务', 0, NULL,
            '商品、店铺和售后服务相关的知识库咨询。', '["商品服务有哪些？"]',
            NULL, NULL, NULL, 0, 100, 1, 0
        ),
        (
            'intent_cat_general', NULL, 'general-products', '通用商品', 1, 'product-service',
            '未明确具体商品品类时的通用商品咨询。', '["商品售后怎么处理？"]',
            NULL, NULL, NULL, 0, 110, 1, 0
        ),
        (
            'intent_topic_return', '54b39ac6fec64f34be19', 'general-return-policy', '退换政策',
            2, 'general-products', '商品退货、换货、退款、售后时效和办理条件的通用规则。',
            '["退换政策是什么？", "退货需要满足什么条件？", "商品可以提现吗？", "换货流程怎么走？", "退款多久能到账？"]',
            'test111', 3, NULL, 0, 120, 1, 0
        )
), updated AS (
    UPDATE t_intent_node AS target
    SET
        kb_id = seed.kb_id,
        name = seed.name,
        level = seed.level,
        parent_code = seed.parent_code,
        description = seed.description,
        examples = seed.examples,
        collection_name = seed.collection_name,
        top_k = seed.top_k,
        mcp_tool_id = seed.mcp_tool_id,
        kind = seed.kind,
        sort_order = seed.sort_order,
        enabled = seed.enabled,
        update_time = CURRENT_TIMESTAMP,
        deleted = seed.deleted
    FROM seed
    WHERE target.id = seed.id AND target.intent_code = seed.intent_code
    RETURNING target.id
)
INSERT INTO t_intent_node (
    id,
    kb_id,
    intent_code,
    name,
    level,
    parent_code,
    description,
    examples,
    collection_name,
    top_k,
    mcp_tool_id,
    kind,
    sort_order,
    enabled,
    deleted
)
SELECT
    seed.id,
    seed.kb_id,
    seed.intent_code,
    seed.name,
    seed.level,
    seed.parent_code,
    seed.description,
    seed.examples,
    seed.collection_name,
    seed.top_k,
    seed.mcp_tool_id,
    seed.kind,
    seed.sort_order,
    seed.enabled,
    seed.deleted
FROM seed
WHERE NOT EXISTS (
    SELECT 1
    FROM t_intent_node AS existing
    WHERE existing.id = seed.id OR existing.intent_code = seed.intent_code
);

COMMIT;
