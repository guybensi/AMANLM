import pytest
from backend.models.document import Chunk
from backend.services.rag_service import ScoredChunk
from backend.services.llm_service import build_prompt


def make_scored_chunk(filename="source.pdf", page_number=1, text="Relevant content here."):
    chunk = Chunk(
        chunk_id="c1",
        doc_id="doc1",
        filename=filename,
        page_number=page_number,
        chunk_index=0,
        text=text,
    )
    return ScoredChunk(chunk=chunk, score=0.8)


class TestBuildPrompt:
    def test_returns_two_messages(self):
        messages = build_prompt("What is X?", [make_scored_chunk()], mode="short")
        assert len(messages) == 2

    def test_first_message_is_system(self):
        messages = build_prompt("What is X?", [make_scored_chunk()], mode="short")
        assert messages[0]["role"] == "system"

    def test_second_message_is_user_with_query(self):
        query = "What is X?"
        messages = build_prompt(query, [make_scored_chunk()], mode="short")
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == query

    def test_short_mode_includes_concise_instruction(self):
        messages = build_prompt("Q?", [make_scored_chunk()], mode="short")
        system = messages[0]["content"]
        assert "2-3 concise sentences" in system

    def test_long_mode_includes_comprehensive_instruction(self):
        messages = build_prompt("Q?", [make_scored_chunk()], mode="long")
        system = messages[0]["content"]
        assert "comprehensive" in system

    def test_source_filename_appears_in_system_prompt(self):
        sc = make_scored_chunk(filename="report.pdf", page_number=3)
        messages = build_prompt("Q?", [sc], mode="short")
        system = messages[0]["content"]
        assert "report.pdf" in system
        assert "Page 3" in system

    def test_source_text_appears_in_system_prompt(self):
        sc = make_scored_chunk(text="The sky is blue.")
        messages = build_prompt("Q?", [sc], mode="short")
        assert "The sky is blue." in messages[0]["content"]

    def test_multiple_sources_all_appear(self):
        chunks = [
            make_scored_chunk(filename="a.pdf", text="Alpha text."),
            make_scored_chunk(filename="b.pdf", text="Beta text."),
        ]
        messages = build_prompt("Q?", chunks, mode="short")
        system = messages[0]["content"]
        assert "a.pdf" in system
        assert "b.pdf" in system
        assert "Alpha text." in system
        assert "Beta text." in system

    def test_sources_are_numbered_sequentially(self):
        chunks = [make_scored_chunk() for _ in range(3)]
        messages = build_prompt("Q?", chunks, mode="short")
        system = messages[0]["content"]
        assert "Source 1:" in system
        assert "Source 2:" in system
        assert "Source 3:" in system

    def test_sources_separated_by_delimiter(self):
        chunks = [make_scored_chunk(text=f"text {i}") for i in range(2)]
        messages = build_prompt("Q?", chunks, mode="short")
        system = messages[0]["content"]
        assert "---" in system


class TestBuildPromptNotebookLMBehaviors:
    """Tests for the NotebookLM-inspired prompting additions."""

    def test_citation_notation_instructed(self):
        """Prompt must instruct the model to use [i] citation notation."""
        system = build_prompt("Q?", [make_scored_chunk()], mode="short")[0]["content"]
        assert "[i]" in system

    def test_bold_instruction_present(self):
        """Prompt must ask the model to bold the most important parts."""
        system = build_prompt("Q?", [make_scored_chunk()], mode="short")[0]["content"]
        assert "Bold" in system or "bold" in system

    def test_outside_source_flagging_instructed(self):
        """Prompt must instruct the model to flag information not from sources."""
        system = build_prompt("Q?", [make_scored_chunk()], mode="short")[0]["content"]
        assert "outside" in system or "not from the sources" in system or "independently verify" in system

    def test_no_relevant_info_note_instructed(self):
        """Prompt must tell the model to note when sources have no relevant info."""
        system = build_prompt("Q?", [make_scored_chunk()], mode="short")[0]["content"]
        assert "not contain" in system or "no relevant" in system or "do not contain" in system

    def test_clarification_instruction_present(self):
        """Prompt must instruct the model to ask for clarification on ambiguous queries."""
        system = build_prompt("Q?", [make_scored_chunk()], mode="short")[0]["content"]
        assert "clarification" in system or "ambiguous" in system

    def test_no_delve_instruction_present(self):
        """Prompt must forbid the word 'delve'."""
        system = build_prompt("Q?", [make_scored_chunk()], mode="short")[0]["content"]
        assert "delve" in system  # the rule mentions the word to forbid it

    def test_english_default_instruction_present(self):
        """Prompt must instruct the model to default to English."""
        system = build_prompt("Q?", [make_scored_chunk()], mode="short")[0]["content"]
        assert "English" in system

    def test_multi_source_citation_format_instructed(self):
        """Prompt must show the multi-source citation format [i, j, k]."""
        system = build_prompt("Q?", [make_scored_chunk()], mode="short")[0]["content"]
        assert "[i, j, k]" in system or "multiple sources" in system

    def test_conversation_history_mentioned_in_system_prompt(self):
        """System prompt must reference conversation history."""
        system = build_prompt("Q?", [make_scored_chunk()], mode="short")[0]["content"]
        assert "conversation history" in system or "conversation" in system


class TestBuildPromptHistory:
    """Tests for conversation history injection."""

    def test_no_history_returns_two_messages(self):
        messages = build_prompt("Q?", [make_scored_chunk()], mode="short")
        assert len(messages) == 2

    def test_history_messages_inserted_between_system_and_user(self):
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        messages = build_prompt("Q?", [make_scored_chunk()], mode="short", history=history)
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1] == {"role": "user", "content": "Previous question"}
        assert messages[2] == {"role": "assistant", "content": "Previous answer"}
        assert messages[3] == {"role": "user", "content": "Q?"}

    def test_last_message_is_always_current_query(self):
        history = [{"role": "user", "content": "Earlier"}, {"role": "assistant", "content": "Reply"}]
        messages = build_prompt("Current?", [make_scored_chunk()], mode="short", history=history)
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Current?"

    def test_empty_history_same_as_no_history(self):
        with_empty = build_prompt("Q?", [make_scored_chunk()], mode="short", history=[])
        without = build_prompt("Q?", [make_scored_chunk()], mode="short")
        assert with_empty == without

    def test_single_turn_history(self):
        history = [{"role": "user", "content": "Hi"}]
        messages = build_prompt("Q?", [make_scored_chunk()], mode="short", history=history)
        assert len(messages) == 3
        assert messages[1] == {"role": "user", "content": "Hi"}
