from __future__ import annotations

from agent_service.memory.models import MemoryContext
from agent_service.rag.schemas import RetrievedSource

SYSTEM_PROMPT = (
    "你是 HYEEE AI，一个面向本地生活点评、店铺推荐和用户问答场景的助手。"
    "回答要简洁、自然、可靠；如果信息不足，先说明需要哪些补充信息。"
)


def build_messages(
    memory_context: MemoryContext,
    message: str,
    *,
    retrieved_sources: list[RetrievedSource] | None = None,
    retrieval_attempted: bool = False,
    mcp_context: str = "",
    mcp_attempted: bool = False,
) -> list[dict[str, str]]:
    system_prompt = SYSTEM_PROMPT + _retrieval_instruction(
        retrieved_sources or [],
        retrieval_attempted=retrieval_attempted,
    ) + _mcp_instruction(mcp_context, mcp_attempted=mcp_attempted)
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(memory_context.to_prompt_messages())
    messages.append({"role": "user", "content": message})
    return messages


def _retrieval_instruction(
    sources: list[RetrievedSource],
    *,
    retrieval_attempted: bool,
) -> str:
    if sources:
        context = "\n\n".join(
            f"[{index}] {source.title}\n{source.content}"
            for index, source in enumerate(sources, start=1)
        )
        return (
            "\n\nKnowledge base context follows. Answer only from this context. "
            "If it does not contain enough information, say that the knowledge base "
            "does not provide enough information. Do not invent policies, dates, fees, "
            "or facts.\n\n"
            f"{context}"
        )
    if retrieval_attempted:
        return (
            "\n\nThe knowledge base search returned no usable context for this request. "
            "Tell the user that the current knowledge base does not provide enough "
            "information. Do not answer from general knowledge or invent facts."
        )
    return ""


def _mcp_instruction(mcp_context: str, *, mcp_attempted: bool) -> str:
    if mcp_context:
        return (
            "\n\nReal-time MCP context follows. It is authoritative for live business data. "
            "Answer only from successful real-time results. If the context says that "
            "information is missing, a tool is unavailable, or a call failed, follow that "
            "instruction and never invent a real-time status.\n\n"
            f"{mcp_context}"
        )
    if mcp_attempted:
        return (
            "\n\nA real-time MCP query was required but returned no usable context. "
            "Tell the user that real-time information cannot currently be obtained; "
            "do not invent it."
        )
    return ""
