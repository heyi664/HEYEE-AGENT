CREATE UNIQUE INDEX IF NOT EXISTS uk_summary_conv_user_active
ON public.t_conversation_summary (conversation_id, user_id)
WHERE deleted = 0;

CREATE INDEX IF NOT EXISTS idx_message_conv_user_role_time_active
ON public.t_message (conversation_id, user_id, role, create_time DESC)
WHERE deleted = 0;
