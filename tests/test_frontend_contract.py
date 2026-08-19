from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

STATIC_DIR = Path(__file__).parents[1] / "src" / "ragagent" / "static"


class ElementIndex(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.by_id: dict[str, tuple[str, dict[str, str | None]]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if element_id := attributes.get("id"):
            self.by_id[element_id] = (tag, attributes)


def frontend_elements() -> dict[str, tuple[str, dict[str, str | None]]]:
    parser = ElementIndex()
    parser.feed((STATIC_DIR / "index.html").read_text(encoding="utf-8"))
    return parser.by_id


def test_dialog_exit_buttons_never_submit_or_trigger_required_validation() -> None:
    elements = frontend_elements()
    assert elements["close-kb-dialog"][1]["type"] == "button"
    assert elements["cancel-kb-dialog"][1]["type"] == "button"
    assert elements["create-kb-submit"][1]["type"] == "submit"
    assert "method" not in elements["new-kb-form"][1]


def test_dialog_and_navigation_accessibility_contract() -> None:
    elements = frontend_elements()
    assert "required" in elements["kb-name"][1]
    assert elements["kb-name"][1]["aria-describedby"] == "kb-name-help"
    assert elements["tab-button-chat"][1]["role"] == "tab"
    assert elements["tab-button-chat"][1]["aria-selected"] == "true"
    assert elements["dropzone"][1]["role"] == "button"
    assert elements["dropzone"][1]["tabindex"] == "0"


def test_dialog_javascript_has_explicit_open_close_and_reset_paths() -> None:
    javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert 'dialog.close("cancel")' in javascript
    assert 'dialog.addEventListener("close", resetKnowledgeBaseDialog)' in javascript
    assert 'event.key !== "Escape"' in javascript
    assert 'event.target === dialog' in javascript
    assert 'setCustomValidity(name ? "" : "请输入知识库名称")' in javascript


def test_frontend_assets_are_cache_busted() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert 'styles.css?v=0.2.0' in html
    assert 'app.js?v=0.2.0' in html


def test_conversation_memory_control_is_bounded_and_sent_with_an_isolated_id() -> None:
    elements = frontend_elements()
    assert elements["conversation-id"][1]["maxlength"] == "120"
    assert elements["memory-turns"][1]["min"] == "0"
    assert elements["memory-turns"][1]["max"] == "10"
    assert elements["memory-turns"][1]["value"] == "5"
    assert elements["retrieval-concurrency"][1]["min"] == "1"
    assert elements["retrieval-concurrency"][1]["max"] == "4"
    assert elements["retrieval-concurrency"][1]["value"] == "2"
    assert elements["new-conversation-button"][1]["type"] == "button"
    javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert "conversation_id: configuredConversationId()" in javascript
    assert "memory_turns: memoryTurns" in javascript
    assert "retrieval_concurrency: retrievalConcurrency" in javascript
    assert "state.conversationIds[state.activeKnowledgeBase.id] = createConversationId()" in javascript


def test_index_strategy_controls_are_sent_to_the_api() -> None:
    elements = frontend_elements()
    assert elements["chunk-strategy"][0] == "select"
    assert elements["dense-backend"][0] == "select"
    javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert 'chunk_strategy: $("#chunk-strategy").value' in javascript
    assert 'dense_backend: $("#dense-backend").value' in javascript


def test_demo_questions_are_discoverable_and_fill_the_composer() -> None:
    javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert 'const marker = "示例问题："' in javascript
    assert 'data-example-question=' in javascript
    assert "input.value = button.dataset.exampleQuestion" in javascript
