from __future__ import annotations

import numpy as np
import pytest

from tribe_capabilities.inference import (
    apply_torch_compat_patches,
    build_text_events_fast,
    predict_from_text_resilient,
    validate_prediction_shape,
)


def _require_neuralset() -> None:
    pytest.importorskip("neuralset")


def test_build_text_events_has_words_and_context() -> None:
    _require_neuralset()
    events = build_text_events_fast("Program A will save 200 people.")
    assert (events["type"] == "Word").any()
    assert (events["type"] == "Text").any()
    word_rows = events[events["type"] == "Word"]
    assert "context" in word_rows.columns
    assert len(word_rows) >= 5


def test_build_text_events_rejects_empty_text() -> None:
    _require_neuralset()
    with pytest.raises(ValueError, match="empty"):
        build_text_events_fast("   ")


def test_torch_compat_patch_is_idempotent() -> None:
    apply_torch_compat_patches()
    apply_torch_compat_patches()


def test_predict_from_text_resilient_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def flaky_predict(_model, _text: str, *, timeline: str = "default") -> np.ndarray:
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("simulated OOM")
        return np.ones((4, 20484), dtype=np.float32)

    monkeypatch.setattr(
        "tribe_capabilities.inference.predict_from_text",
        flaky_predict,
    )
    preds = predict_from_text_resilient(object(), "hello", max_attempts=3, retry_delay_seconds=0)
    validate_prediction_shape(preds, expected_vertices=20484)
    assert calls["n"] == 2
