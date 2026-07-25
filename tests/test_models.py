"""Protocol conformance and the D8 attention guard.

The Protocol is what lets one results table rank a scikit-learn pipeline against a fine-tuned
transformer, so it is worth asserting rather than assuming.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.protocols import SentimentModel
from models.registry import create_model, register, registered_names
from models.roberta import RobertaSentiment
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


class _BatchLossModel:
    """Tiny stand-in whose mean loss differs across deliberately unequal batches."""

    def __init__(self) -> None:
        self.parameter = torch.nn.Parameter(torch.tensor(0.0))

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass

    def parameters(self):
        return [self.parameter]

    def __call__(self, *, input_ids, attention_mask, labels):
        del input_ids, attention_mask
        mean_loss = 1.0 if len(labels) == 2 else 9.0
        loss = self.parameter * 0.0 + mean_loss
        logits = torch.zeros((len(labels), 2), dtype=torch.float32)
        logits[:, 0] = 1.0
        return SimpleNamespace(loss=loss, logits=logits)


def _unequal_batches() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "input_ids": torch.ones((2, 3), dtype=torch.long),
            "attention_mask": torch.ones((2, 3), dtype=torch.long),
            "labels": torch.zeros(2, dtype=torch.long),
        },
        {
            "input_ids": torch.ones((1, 3), dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "labels": torch.zeros(1, dtype=torch.long),
        },
    ]


def _loss_harness() -> RobertaSentiment:
    model = RobertaSentiment.__new__(RobertaSentiment)
    model.model = _BatchLossModel()
    model.device = torch.device("cpu")
    model.log_every_steps = 100
    return model


def test_validation_loss_is_weighted_by_examples_not_batches():
    """Two rows at loss 1 and one at loss 9 have a per-example mean of 11/3, not 5."""
    model = _loss_harness()
    loss, accuracy = model._evaluate_loss(_unequal_batches())
    assert loss == pytest.approx(11 / 3)
    assert accuracy == 1.0


def test_training_loss_is_weighted_by_examples_not_batches():
    """The training history must use the same per-example weighting as validation."""
    model = _loss_harness()
    optimizer = torch.optim.SGD(model.model.parameters(), lr=0.1)
    loss = model._train_one_epoch(_unequal_batches(), optimizer, epoch=1)
    assert loss == pytest.approx(11 / 3)


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
    """A cap of ~0 must interrupt epoch 1 and retain an explicit partial-epoch record."""
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
    assert model.train_report["epochs_run"] == 0
    assert model.train_report["epochs_configured"] == 5
    assert model.train_report["partial_epoch"]["epoch"] == 1
    assert model.train_report["partial_epoch"]["steps_run"] == 0


def test_pretrained_revision_is_threaded_to_tokenizer_and_model(monkeypatch):
    """A requested immutable revision must reach both Hugging Face loads."""
    from transformers import AutoTokenizer, RobertaForSequenceClassification

    calls: list[tuple[str, str, str | None]] = []

    class _LoadedModel:
        config = SimpleNamespace(_attn_implementation="eager")

        def to(self, device):
            return self

    def fake_tokenizer(name, *, revision):
        calls.append(("tokenizer", name, revision))
        return object()

    def fake_model(name, *, revision, num_labels, attn_implementation):
        assert num_labels == 2
        assert attn_implementation == "eager"
        calls.append(("model", name, revision))
        return _LoadedModel()

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", fake_tokenizer)
    monkeypatch.setattr(RobertaForSequenceClassification, "from_pretrained", fake_model)

    model = RobertaSentiment.__new__(RobertaSentiment)
    model.pretrained = "roberta-base"
    model.revision = "immutable-revision"
    model.random_weight_layers = None
    model.num_labels = 2
    model.device = torch.device("cpu")

    model._build_tokenizer()
    model._build_model()
    assert calls == [
        ("tokenizer", "roberta-base", "immutable-revision"),
        ("model", "roberta-base", "immutable-revision"),
    ]


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
