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
