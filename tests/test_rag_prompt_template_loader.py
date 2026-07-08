from __future__ import annotations

from agent_service.rag.prompt_template_loader import PromptTemplateLoader


def test_prompt_template_loader_renders_slots_and_cleans_blank_lines(tmp_path) -> None:
    template = tmp_path / "demo.st"
    template.write_text("Hello {name}\n\n\n\n{missing}\n", encoding="utf-8")
    loader = PromptTemplateLoader(tmp_path)

    rendered = loader.render("demo.st", {"name": "RAG"})

    assert rendered == "Hello RAG"


def test_prompt_template_loader_caches_loaded_templates(tmp_path) -> None:
    template = tmp_path / "demo.st"
    template.write_text("{value}", encoding="utf-8")
    loader = PromptTemplateLoader(tmp_path)

    assert loader.render("demo.st", {"value": "one"}) == "one"
    template.write_text("{value} changed", encoding="utf-8")
    assert loader.render("demo.st", {"value": "two"}) == "two"
