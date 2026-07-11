BEGIN;

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
    kind,
    sort_order,
    enabled,
    created_by,
    updated_by,
    deleted
)
VALUES
    (
        'intent_domain_prod',
        NULL,
        'product-service',
        '商品服务',
        0,
        NULL,
        '商品、店铺和售后服务相关的知识库咨询。',
        '["商品服务有哪些？"]',
        NULL,
        NULL,
        0,
        100,
        1,
        'system',
        'system',
        0
    ),
    (
        'intent_cat_general',
        NULL,
        'general-products',
        '通用商品',
        1,
        'product-service',
        '未明确具体商品品类时的通用商品咨询。',
        '["商品售后怎么处理？"]',
        NULL,
        NULL,
        0,
        110,
        1,
        'system',
        'system',
        0
    ),
    (
        'intent_topic_return',
        '54b39ac6fec64f34be19',
        'general-return-policy',
        '退换政策',
        2,
        'general-products',
        '商品退货、换货、退款、售后时效和办理条件的通用规则。',
        '["退换政策是什么？", "退货需要满足什么条件？", "商品可以提现吗？", "换货流程怎么走？", "退款多久能到账？"]',
        'test111',
        3,
        0,
        120,
        1,
        'system',
        'system',
        0
    )
ON CONFLICT (intent_code) DO UPDATE
SET
    name = EXCLUDED.name,
    level = EXCLUDED.level,
    parent_code = EXCLUDED.parent_code,
    description = EXCLUDED.description,
    examples = EXCLUDED.examples,
    collection_name = EXCLUDED.collection_name,
    top_k = EXCLUDED.top_k,
    kind = EXCLUDED.kind,
    sort_order = EXCLUDED.sort_order,
    enabled = EXCLUDED.enabled,
    updated_by = EXCLUDED.updated_by,
    update_time = CURRENT_TIMESTAMP,
    deleted = EXCLUDED.deleted
WHERE t_intent_node.created_by = 'system';

COMMIT;
