from __future__ import annotations

import time

from agent_service.core.observability import record_stage
from agent_service.rag.intent_models import NodeScore
from agent_service.rag.llm_response_cleaner import parse_json_object
from agent_service.rag.prompt_template_loader import PromptTemplateLoader
from agent_service.services.llm_service import LLMService

GUIDANCE_AMBIGUITY_PROMPT = "guidance-ambiguity-check.st"


class AmbiguityLLMChecker:
    def __init__(
        self,
        *,
        llm_service: LLMService,
        prompt_loader: PromptTemplateLoader | None = None,
    ) -> None:
        self._llm_service = llm_service
        self._prompt_loader = prompt_loader or PromptTemplateLoader()

    async def check_ambiguity(self, question: str, ranked: list[NodeScore]) -> bool:
        started_at = time.perf_counter()
        try:
            prompt = self._prompt_loader.render(
                GUIDANCE_AMBIGUITY_PROMPT,
                {
                    "question": question,
                    "candidates": _build_candidates_text(ranked),
                },
            )
            result = await self._llm_service.complete(
                [{"role": "user", "content": prompt}],
                use_tools=False,
            )
            parsed = parse_json_object(result.reply)
            ambiguous = bool(parsed.get("ambiguous"))
            record_stage(
                "ambiguity_llm_check",
                elapsed_ms=_elapsed_ms(started_at),
                candidateCount=len(ranked),
                ambiguous=ambiguous,
                status="success",
            )
            return ambiguous
        except Exception:
            record_stage(
                "ambiguity_llm_check",
                elapsed_ms=_elapsed_ms(started_at),
                candidateCount=len(ranked),
                ambiguous=True,
                status="failed",
            )
            return True


def _build_candidates_text(ranked: list[NodeScore]) -> str:
    lines: list[str] = []
    for score in ranked:
        node = score.node
        path = node.full_path or node.name
        lines.append(
            f"- id={node.id}\n"
            f"  name={node.name}\n"
            f"  path={path}\n"
            f"  score={score.score}"
        )
    return "\n".join(lines)


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)
