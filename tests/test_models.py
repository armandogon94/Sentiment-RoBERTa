"""Protocol conformance and the D8 attention guard.

The Protocol is what lets one results table rank a scikit-learn pipeline against a fine-tuned
transformer, so it is worth asserting rather than assuming.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.protocols import SentimentModel
from models.registry import create_model, register, registered_names
from utils.device import capability_report, resolve_device

TEXTS = [
    "Absolutely wonderful, exceeded every expectation and arrived early.",
    "Terrible quality, broke on the second day, requesting a refund.",
] * 20
LABELS = [1, 0] * 20


def test_both_models_are_registered():
    assert set(registered_names()) == {"tfidf_logreg", "roberta"}


def test_duplicate_registration_is_rejected():
    with pytest.raises(ValueError, match="already registered"):

        @register("tfidf_logreg")
        def _dupe(**kwargs):  # pragma: no cover - never called
            raise AssertionError


def test_unknown_model_name_is_rejected():
    with pytest.raises(KeyError, match="unknown model"):
        create_model("bert")


def test_baseline_satisfies_the_protocol():
    model = create_model("tfidf_logreg", seed=0)
    assert isinstance(model, SentimentModel)


def test_transformer_satisfies_the_protocol():
    model = create_model("roberta", random_weight_layers=2, max_len=32, epochs=1, batch_size=8)
    assert isinstance(model, SentimentModel)


def test_baseline_round_trip(tmp_path):
    model = create_model("tfidf_logreg", seed=0, remove_stopwords=False, stem=False)
    model.fit(TEXTS, LABELS)
    preds = model.predict(TEXTS)
    assert preds.shape == (len(TEXTS),)
    assert set(np.unique(preds)) <= {0, 1}
    saved = model.save(tmp_path / "m.pkl")
    assert saved.exists()


def test_baseline_refuses_to_predict_before_fitting():
    model = create_model("tfidf_logreg", seed=0)
    with pytest.raises(RuntimeError, match="has not been fit"):
        model.predict(TEXTS)


def test_baseline_feature_report_surfaces_negation_when_it_survives():
    """The control's own interpretability, and a direct D3 check."""
    model = create_model(
        "tfidf_logreg", seed=0, remove_stopwords=False, stem=False, alphanumeric_only=False
    )
    model.fit(
        [
            "this is good and lovely",
            "this is not good at all",
            "great and lovely",
            "not great at all",
        ]
        * 10,
        [1, 0, 1, 0] * 10,
    )
    report = model.feature_report(top_k=10)
    negatives = {f["feature"] for f in report["most_negative"]}
    assert "not" in negatives


def test_transformer_uses_eager_attention_D8():
    """On transformers 5.x, sdpa returns an EMPTY attentions tuple with only a warning."""
    model = create_model("roberta", random_weight_layers=2, max_len=32, epochs=1, batch_size=8)
    assert model.model.config._attn_implementation == "eager"
    ids = torch.randint(4, 100, (1, 12))
    out = model.model(input_ids=ids, attention_mask=torch.ones_like(ids), output_attentions=True)
    assert len(out.attentions) == 2
    assert out.attentions[0].shape[-1] == 12


def test_transformer_trains_and_records_a_bounded_report():
    model = create_model(
        "roberta",
        random_weight_layers=2,
        max_len=32,
        epochs=2,
        batch_size=8,
        wall_clock_cap_min=5.0,
        device=torch.device("cpu"),
    )
    model.fit_with_validation(TEXTS[:24], LABELS[:24], TEXTS[24:32], LABELS[24:32])
    report = model.train_report
    assert report["epochs_run"] <= report["epochs_configured"]
    assert report["selected_epoch"] >= 1
    assert report["wall_clock_capped"] is False
    assert report["attn_implementation"] == "eager"
    assert len(report["history"]) == report["epochs_run"]
    assert all(h["epoch_seconds"] > 0 for h in report["history"])


def test_wall_clock_cap_actually_stops_a_run():
    """A cap of ~0 must truncate after epoch 1 rather than being advisory."""
    model = create_model(
        "roberta",
        random_weight_layers=2,
        max_len=32,
        epochs=5,
        batch_size=8,
        wall_clock_cap_min=1e-6,
        device=torch.device("cpu"),
    )
    model.fit_with_validation(TEXTS[:24], LABELS[:24], TEXTS[24:32], LABELS[24:32])
    assert model.train_report["wall_clock_capped"] is True
    assert model.train_report["epochs_run"] == 1
    assert model.train_report["epochs_configured"] == 5


def test_truncation_rate_is_measured():
    model = create_model("roberta", random_weight_layers=2, max_len=8, epochs=1, batch_size=8)
    report = model.evaluate_truncation(["short", " ".join(["word"] * 200)])
    assert report["n_examples"] == 2
    assert report["n_truncated"] == 1
    assert report["frac_truncated"] == 0.5


# ── device ───────────────────────────────────────────────────────────────────────────


def test_resolve_device_degrades_to_cpu_where_mps_is_absent():
    """CI runs on ubuntu-latest with no MPS; `auto` must not raise there."""
    device = resolve_device("auto")
    assert device.type in {"mps", "cpu"}
    assert resolve_device("cpu").type == "cpu"


def test_explicit_mps_request_raises_when_unavailable():
    if torch.backends.mps.is_available():
        assert resolve_device("mps").type == "mps"
    else:  # pragma: no cover - only on non-Apple CI
        with pytest.raises(RuntimeError, match="mps"):
            resolve_device("mps")


def test_capability_report_records_what_could_change_a_number():
    report = capability_report(torch.device("cpu"))
    for key in ("device", "mps_available", "torch_num_threads", "platform", "low_power_mode"):
        assert key in report
