import pytest
from backend.utils.text_utils import clean_text, recursive_chunk


class TestCleanText:
    def test_strips_leading_and_trailing_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_collapses_multiple_spaces_and_tabs(self):
        result = clean_text("hello   world\t\there")
        assert "  " not in result
        assert "\t\t" not in result

    def test_collapses_excessive_newlines(self):
        result = clean_text("a\n\n\n\nb")
        assert "\n\n\n" not in result

    def test_removes_non_printable_characters(self):
        result = clean_text("hello\x00world")
        assert "\x00" not in result
        assert "hello" in result
        assert "world" in result

    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""

    def test_whitespace_only_returns_empty(self):
        assert clean_text("   \n\t  ") == ""

    def test_preserves_unicode_letters(self):
        result = clean_text("héllo wörld")
        assert "héllo" in result
        assert "wörld" in result

    def test_preserves_newlines_within_limit(self):
        result = clean_text("line1\n\nline2")
        assert "\n\n" in result


class TestRecursiveChunk:
    def test_empty_string_returns_empty_list(self):
        assert recursive_chunk("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert recursive_chunk("   ") == []

    def test_short_text_returns_single_chunk(self):
        text = "hello world"
        result = recursive_chunk(text, chunk_size=512, overlap=64)
        assert result == ["hello world"]

    def test_text_exactly_one_chunk_size(self):
        words = " ".join(["word"] * 512)
        result = recursive_chunk(words, chunk_size=512, overlap=64)
        assert len(result) == 1

    def test_text_splits_into_multiple_chunks(self):
        words = " ".join([f"w{i}" for i in range(600)])
        result = recursive_chunk(words, chunk_size=512, overlap=64)
        assert len(result) == 2

    def test_overlap_causes_shared_words(self):
        words = " ".join([f"w{i}" for i in range(600)])
        result = recursive_chunk(words, chunk_size=512, overlap=64)
        last_words_of_first = set(result[0].split()[-64:])
        first_words_of_second = set(result[1].split()[:64])
        assert last_words_of_first == first_words_of_second

    def test_no_overlap(self):
        words = " ".join([f"w{i}" for i in range(600)])
        result = recursive_chunk(words, chunk_size=512, overlap=0)
        # With no overlap the second chunk starts right after the first
        first_end_word = result[0].split()[-1]
        second_start_word = result[1].split()[0]
        assert first_end_word != second_start_word

    def test_chunk_size_one(self):
        result = recursive_chunk("a b c", chunk_size=1, overlap=0)
        assert result == ["a", "b", "c"]

    def test_all_chunks_are_non_empty(self):
        words = " ".join([f"w{i}" for i in range(1000)])
        result = recursive_chunk(words, chunk_size=100, overlap=10)
        assert all(chunk.strip() for chunk in result)
