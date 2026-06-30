-- Add RocketMQ message id to knowledge document chunk logs.
ALTER TABLE public.t_knowledge_document_chunk_log
    ADD COLUMN IF NOT EXISTS message_id varchar(64);

COMMENT ON COLUMN public.t_knowledge_document_chunk_log.message_id IS 'RocketMQ 消息ID';

CREATE INDEX IF NOT EXISTS idx_chunk_log_message_id
    ON public.t_knowledge_document_chunk_log USING btree (message_id);
