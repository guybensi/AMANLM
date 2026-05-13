import pytest
from unittest.mock import patch, MagicMock
from backend.models.document import Chunk
from backend.models.chat import SourceProof
from backend.services.rag_service import (
    ScoredChunk,
    calculate_confidence,
    confidence_label,
    build_source_proofs,
    contains_inference_markers,
)


def make_chunk(doc_id="doc1", chunk_index=0, text="sample text", page_number=1, filename="file.txt"):
    return Chunk(
        chunk_id=f"{doc_id}-{chunk_index}",
        doc_id=doc_id,
        filename=filename,
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
    )


class TestCalculateConfidence:
    def test_empty_list_returns_zero(self):
        assert calculate_confidence([]) == 0.0

    def test_single_high_score_chunk(self):
        chunk = make_chunk()
        scored = [ScoredChunk(chunk=chunk, score=0.45)]
        result = calculate_confidence(scored)
        assert result == 100.0

    def test_single_low_score_returns_low_confidence(self):
        # score=0.05 is exactly at the retrieval floor so retrieval_score=0 and
        # coverage_score=0, but a single chunk has std=0 → consistency_score=100,
        # contributing 0.15*100=15.0 to confidence.
        chunk = make_chunk()
        scored = [ScoredChunk(chunk=chunk, score=0.05)]
        result = calculate_confidence(scored)
        assert result == 15.0

    def test_score_below_floor_clamps_retrieval_to_zero(self):
        # With score=0.0 only the consistency component (std=0 for 1 chunk) contributes.
        chunk = make_chunk()
        scored = [ScoredChunk(chunk=chunk, score=0.0)]
        result = calculate_confidence(scored)
        assert result == 15.0

    def test_confidence_is_between_0_and_100(self):
        chunks = [ScoredChunk(chunk=make_chunk(chunk_index=i), score=0.3 + i * 0.01) for i in range(5)]
        result = calculate_confidence(chunks)
        assert 0.0 <= result <= 100.0

    def test_higher_scores_yield_higher_confidence(self):
        low_chunks = [ScoredChunk(chunk=make_chunk(chunk_index=i), score=0.1) for i in range(3)]
        high_chunks = [ScoredChunk(chunk=make_chunk(chunk_index=i), score=0.4) for i in range(3)]
        assert calculate_confidence(high_chunks) > calculate_confidence(low_chunks)

    def test_result_is_rounded_to_one_decimal(self):
        chunk = make_chunk()
        scored = [ScoredChunk(chunk=chunk, score=0.25)]
        result = calculate_confidence(scored)
        assert result == round(result, 1)


class TestConfidenceLabel:
    def test_high_at_65(self):
        assert confidence_label(65.0) == "High"

    def test_high_above_65(self):
        assert confidence_label(80.0) == "High"

    def test_medium_at_40(self):
        assert confidence_label(40.0) == "Medium"

    def test_medium_at_64(self):
        assert confidence_label(64.9) == "Medium"

    def test_low_at_20(self):
        assert confidence_label(20.0) == "Low"

    def test_low_at_39(self):
        assert confidence_label(39.9) == "Low"

    def test_very_low_below_20(self):
        assert confidence_label(19.9) == "Very Low"

    def test_very_low_at_zero(self):
        assert confidence_label(0.0) == "Very Low"


class TestBuildSourceProofs:
    def test_empty_list_returns_empty(self):
        assert build_source_proofs([]) == []

    def test_basic_proof_structure(self):
        chunk = make_chunk(text="This is a quote.", filename="doc.pdf", page_number=2)
        scored = [ScoredChunk(chunk=chunk, score=0.75)]
        proofs = build_source_proofs(scored)
        assert len(proofs) == 1
        p = proofs[0]
        assert p.filename == "doc.pdf"
        assert p.page_number == 2
        assert p.quote == "This is a quote."
        assert p.relevance_score == 0.75

    def test_long_text_is_truncated_at_400_chars(self):
        long_text = "x" * 500
        chunk = make_chunk(text=long_text)
        scored = [ScoredChunk(chunk=chunk, score=0.5)]
        proofs = build_source_proofs(scored)
        assert proofs[0].quote.endswith("…")
        assert len(proofs[0].quote) == 401  # 400 chars + ellipsis

    def test_text_under_400_chars_not_truncated(self):
        short_text = "a" * 399
        chunk = make_chunk(text=short_text)
        scored = [ScoredChunk(chunk=chunk, score=0.5)]
        proofs = build_source_proofs(scored)
        assert not proofs[0].quote.endswith("…")

    def test_duplicate_doc_id_and_chunk_index_are_deduplicated(self):
        chunk = make_chunk(doc_id="doc1", chunk_index=0)
        scored = [
            ScoredChunk(chunk=chunk, score=0.8),
            ScoredChunk(chunk=chunk, score=0.6),
        ]
        proofs = build_source_proofs(scored)
        assert len(proofs) == 1

    def test_different_chunks_same_doc_are_both_included(self):
        chunk_a = make_chunk(doc_id="doc1", chunk_index=0)
        chunk_b = make_chunk(doc_id="doc1", chunk_index=1)
        scored = [
            ScoredChunk(chunk=chunk_a, score=0.8),
            ScoredChunk(chunk=chunk_b, score=0.6),
        ]
        proofs = build_source_proofs(scored)
        assert len(proofs) == 2

    def test_relevance_score_is_rounded_to_3_decimals(self):
        chunk = make_chunk()
        scored = [ScoredChunk(chunk=chunk, score=0.123456)]
        proofs = build_source_proofs(scored)
        assert proofs[0].relevance_score == 0.123


class TestContainsInferenceMarkers:
    """Tests for inference detection via phrase scanning and confidence threshold."""

    def test_clean_answer_high_confidence_returns_false(self):
        assert contains_inference_markers("The report states sales grew 12%.", 80.0) is False

    def test_independently_verify_phrase_triggers_true(self):
        assert contains_inference_markers("You may want to independently verify this claim.", 75.0) is True

    def test_not_from_the_sources_triggers_true(self):
        assert contains_inference_markers("This information is not from the sources.", 70.0) is True

    def test_outside_the_sources_triggers_true(self):
        assert contains_inference_markers("This is outside the given sources.", 60.0) is True

    def test_general_knowledge_phrase_triggers_true(self):
        assert contains_inference_markers("Based on general knowledge, this is expected.", 65.0) is True

    def test_phrase_detection_is_case_insensitive(self):
        assert contains_inference_markers("You should INDEPENDENTLY VERIFY this.", 75.0) is True

    def test_very_low_confidence_triggers_true_regardless_of_text(self):
        assert contains_inference_markers("The answer is perfectly clear.", 15.0) is True

    def test_confidence_exactly_at_threshold_is_not_flagged(self):
        assert contains_inference_markers("Plain answer with no inference phrases.", 20.0) is False

    def test_confidence_just_below_threshold_triggers_true(self):
        assert contains_inference_markers("Plain answer with no inference phrases.", 19.9) is True

    def test_both_signals_present_returns_true(self):
        assert contains_inference_markers("Independently verify this claim.", 10.0) is True

    def test_empty_answer_very_low_confidence_returns_true(self):
        assert contains_inference_markers("", 5.0) is True

    def test_empty_answer_high_confidence_returns_false(self):
        assert contains_inference_markers("", 80.0) is False

    def test_partial_phrase_does_not_trigger(self):
        # "verify" alone should not trigger — only the full phrase "independently verify"
        assert contains_inference_markers("Please verify the document.", 50.0) is False

    def test_my_training_phrase_triggers_true(self):
        assert contains_inference_markers("Based on my training data, this seems likely.", 60.0) is True
