from __future__ import annotations

from fastapi.testclient import TestClient

from agent_service.main import create_app


def test_frontend_chat_page_is_served() -> None:
    client = TestClient(create_app())

    response = client.get("/ui/chat.html")

    assert response.status_code == 200
    assert "HYEEE AI" in response.text
    assert "/v1/agent/chat" in response.text


def test_root_redirects_to_chat_page() -> None:
    client = TestClient(create_app(), follow_redirects=False)

    response = client.get("/")

    assert response.status_code == 307
    assert response.headers["location"] == "/ui/chat.html"

def test_chat_layout_keeps_message_area_scrollable() -> None:
    client = TestClient(create_app())

    response = client.get("/ui/css/chat.css")

    assert response.status_code == 200
    css = response.text
    assert ".chat-main {" in css
    assert "min-height: 0;" in css
    assert "overflow: hidden;" in css
    assert ".chat-body {" in css
    assert "overflow-y: auto;" in css
    assert "grid-template-rows: minmax(0, 1fr);" in css
    assert "height: 0;" in css
    assert "flex: 1 1 0;" in css

def test_assistant_messages_use_sanitized_markdown_rendering() -> None:
    client = TestClient(create_app())

    page = client.get("/ui/chat.html")
    marked = client.get("/ui/js/marked.umd.js")
    purifier = client.get("/ui/js/purify.min.js")

    assert page.status_code == 200
    assert marked.status_code == 200
    assert purifier.status_code == 200
    assert 'v-html="renderMarkdown(m.content)"' in page.text
    assert "window.marked.parse" in page.text
    assert "window.DOMPurify.sanitize" in page.text
    assert "fallback.innerHTML.replace(/\\n/g" in page.text
    assert "m.role === 'user'" in page.text
