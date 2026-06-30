-- pgvector-backed vector table for knowledge chunks.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS public.t_knowledge_vector (
    id varchar(20) NOT NULL,
    kb_id varchar(20) NOT NULL,
    doc_id varchar(20) NOT NULL,
    chunk_id varchar(20) NOT NULL,
    chunk_index int4 NOT NULL,
    embedding vector(1024) NOT NULL,
    metadata jsonb,
    enabled int2 NOT NULL DEFAULT 1,
    created_by varchar(20) NOT NULL,
    updated_by varchar(20),
    create_time timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted int2 NOT NULL DEFAULT 0,
    PRIMARY KEY (id)
);

COMMENT ON TABLE public.t_knowledge_vector IS '知识库文档分块向量表';
COMMENT ON COLUMN public.t_knowledge_vector.id IS 'ID';
COMMENT ON COLUMN public.t_knowledge_vector.kb_id IS '知识库ID';
COMMENT ON COLUMN public.t_knowledge_vector.doc_id IS '文档ID';
COMMENT ON COLUMN public.t_knowledge_vector.chunk_id IS '分块ID';
COMMENT ON COLUMN public.t_knowledge_vector.chunk_index IS '分块序号';
COMMENT ON COLUMN public.t_knowledge_vector.embedding IS '向量，BAAI/bge-m3 1024维';
COMMENT ON COLUMN public.t_knowledge_vector.metadata IS '元数据';
COMMENT ON COLUMN public.t_knowledge_vector.enabled IS '是否启用';
COMMENT ON COLUMN public.t_knowledge_vector.created_by IS '创建人';
COMMENT ON COLUMN public.t_knowledge_vector.updated_by IS '修改人';
COMMENT ON COLUMN public.t_knowledge_vector.create_time IS '创建时间';
COMMENT ON COLUMN public.t_knowledge_vector.update_time IS '更新时间';
COMMENT ON COLUMN public.t_knowledge_vector.deleted IS '是否删除 0：正常 1：删除';

CREATE INDEX IF NOT EXISTS idx_knowledge_vector_doc_id
    ON public.t_knowledge_vector USING btree (doc_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_vector_chunk_id
    ON public.t_knowledge_vector USING btree (chunk_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_vector_embedding
    ON public.t_knowledge_vector USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
