from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest


def _caption_text(figure: Any) -> str:
    """``utils.plots.caption`` writes at figure level, so the caption lives here."""
    return " ".join(text.get_text() for text in figure.texts)


def test_ablation_value_labels_clear_upper_whiskers_and_axes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import scripts.export_figures as figures

    captured: dict[str, Any] = {}

    def capture_figure(figure: Any, name: str, out_dirs: list[Path]) -> list[Path]:
        captured["figure"] = figure
        return [out_dirs[0] / f"{name}.png"]

    monkeypatch.setattr(figures.plots, "save_figure", capture_figure)
    cells: list[dict[str, Any]] = [
        {
            "ablation_label": "notebook chain, unigram",
            "accuracy": 0.8480,
            "accuracy_ci": {"low": 0.8244, "high": 0.8689},
            "n_train": 8100,
            "n_test": 1000,
        },
        {
            "ablation_label": "notebook chain, uni+bigram",
            "accuracy": 0.8380,
            "accuracy_ci": {"low": 0.8139, "high": 0.8595},
            "n_train": 8100,
            "n_test": 1000,
        },
        {
            "ablation_label": "negation preserved, unigram",
            "accuracy": 0.8510,
            "accuracy_ci": {"low": 0.8276, "high": 0.8717},
            "n_train": 8100,
            "n_test": 1000,
        },
        {
            "ablation_label": "negation preserved, uni+bigram",
            "accuracy": 0.8700,
            "accuracy_ci": {"low": 0.8477, "high": 0.8894},
            "n_train": 8100,
            "n_test": 1000,
        },
    ]
    metrics = {"models": {"roberta": {"accuracy": 0.9600}}}
    ablation_metrics = {
        "ablation": cells,
        "config_name": "small",
        "seed": 1337,
    }

    figures.figure_baseline_ablation(metrics, [tmp_path], ablation_metrics)

    figure = captured["figure"]
    axis = figure.axes[0]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for text, cell in zip(axis.texts[: len(cells)], cells, strict=True):
        whisker_end_px = axis.transData.transform((cell["accuracy_ci"]["high"], 0.0))[0]
        text_box = text.get_window_extent(renderer=renderer)
        assert text_box.x0 > whisker_end_px
        assert text_box.x1 <= axis.bbox.x1


@pytest.mark.parametrize("label_name", ["positive", "negative"])
def test_saliency_uses_plain_language_y_axis_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, label_name: str
) -> None:
    import interpretability.saliency as saliency
    import scripts.export_figures as figures

    captured: dict[str, Any] = {}

    def fake_gradient_saliency(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            tokens=["<s>", "clear", "label", "</s>"],
            scores=np.array([0.0, 0.3, 0.2, 0.0]),
            predicted_label=1 if label_name == "positive" else 0,
        )

    def capture_figure(figure: Any, name: str, out_dirs: list[Path]) -> list[Path]:
        captured["figure"] = figure
        return [out_dirs[0] / f"{name}.png"]

    monkeypatch.setattr(saliency, "gradient_saliency", fake_gradient_saliency)
    monkeypatch.setattr(figures.plots, "save_figure", capture_figure)
    model = SimpleNamespace(model=object(), tokenizer=object())
    cfg = SimpleNamespace(MODEL=SimpleNamespace(MAX_LEN=16))
    metrics = {
        "_checkpoint_sha256": "fixture",
        "config_name": "small",
        "git_sha": "fixture",
        "seed": 1337,
    }

    figures.figure_saliency(
        model,
        cfg,
        "cpu",
        [("clear label", 1 if label_name == "positive" else 0)],
        label_name,
        metrics,
        [tmp_path],
    )

    assert {axis.get_ylabel() for axis in captured["figure"].axes} == {"Token importance"}
    assert captured["figure"]._suptitle.get_text() == (
        f"Gradient-norm saliency: {label_name} reviews"
    )
    plt.close(captured["figure"])


# ── the three representation figures ─────────────────────────────────────────────────


MODEL_METRICS: dict[str, Any] = {
    "_checkpoint_sha256": "fixture",
    "config_name": "small",
    "git_sha": "fixture",
    "models": {"roberta": {"accuracy": 0.96}},
    "seed": 1337,
}


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Intercept save_figure so a figure can be inspected without writing a PNG."""
    import scripts.export_figures as figures

    captured: dict[str, Any] = {}

    def capture_figure(figure: Any, name: str, out_dirs: list[Path]) -> list[Path]:
        captured["figure"] = figure
        captured["name"] = name
        return [out_dirs[0] / f"{name}.png"]

    monkeypatch.setattr(figures.plots, "save_figure", capture_figure)
    return captured


def _representations(logits: np.ndarray) -> Any:
    from interpretability.representations import ClsRepresentations

    rng = np.random.default_rng(1337)
    final = rng.normal(size=(logits.shape[0], 5))
    return ClsRepresentations(hidden=np.stack([final, final]), logits=logits)


def test_embedding_figure_marks_errors_and_names_the_t_sne_caveat(
    capture: dict[str, Any], tmp_path: Path
) -> None:
    """The caption must state that t-SNE distances carry no metric meaning."""
    import scripts.export_figures as figures
    from interpretability.representations import boundary_summary

    labels = np.array([0, 1] * 20)
    logits = np.stack([np.where(labels == 0, 3.0, -3.0), np.where(labels == 0, -3.0, 3.0)], axis=1)
    logits[[3, 7]] *= -1.0  # two deliberate errors
    representations = _representations(logits)
    summary = boundary_summary(representations, labels, n_neighbours=5)

    figures.figure_embedding_space(representations, labels, summary, MODEL_METRICS, [tmp_path])

    figure = capture["figure"]
    assert capture["name"] == "embedding_space_3d"
    axis = figure.axes[0]
    assert axis.get_zlabel() == "t-SNE 3"
    legend_labels = [text.get_text() for text in axis.get_legend().get_texts()]
    assert sum("MISCLASSIFIED" in label for label in legend_labels) == 2
    assert sum("predicted correctly" in label for label in legend_labels) == 2

    caption = _caption_text(figure)
    assert "NOT metrically meaningful" in caption
    assert f"{summary.probability_margin_incorrect:.4f}" in caption
    assert f"{100 * summary.opposite_neighbours_incorrect:.1f}%" in caption
    plt.close(figure)


def test_layer_probe_figure_labels_every_hidden_state_and_disclaims_causality(
    capture: dict[str, Any], tmp_path: Path
) -> None:
    import scripts.export_figures as figures
    from interpretability.representations import LayerProbe

    accuracies = [0.51, 0.82, 0.90, 0.95, 0.96]
    probes = [
        LayerProbe(layer=i, accuracy=a, n_train=8100, n_test=1000) for i, a in enumerate(accuracies)
    ]

    figures.figure_layer_probe(probes, 3, MODEL_METRICS, [tmp_path])

    figure = capture["figure"]
    assert capture["name"] == "layer_probe_accuracy"
    axis = figure.axes[0]
    assert [int(tick) for tick in axis.get_xticks()] == list(range(len(accuracies)))
    assert axis.get_ylabel() == "test accuracy of the linear probe"
    assert "saturated from block 3" in axis.get_title()

    caption = _caption_text(figure)
    assert "8100 TRAIN rows and scored on 1000 TEST rows, never the same rows" in caption
    assert "DECODABLE" in caption
    assert "not that the model uses that information downstream" in caption
    plt.close(figure)


def test_entropy_atlas_labels_both_extremes_and_keeps_the_attention_caveat(
    capture: dict[str, Any], tmp_path: Path
) -> None:
    import scripts.export_figures as figures
    from interpretability.representations import AttentionAtlas

    entropy = np.full((12, 12), 2.5)
    entropy[1, 2] = 0.004  # L2H3, the most focused
    entropy[0, 10] = 4.316  # L1H11, the most diffuse
    atlas = AttentionAtlas(
        entropy=entropy,
        sink_share=np.full((12, 12), 0.05),
        n_examples=1000,
        mean_inner_tokens=99.4,
        mean_max_entropy=4.4413,
    )

    figures.figure_attention_entropy_atlas(atlas, MODEL_METRICS, [tmp_path])

    figure = capture["figure"]
    assert capture["name"] == "attention_entropy_atlas"
    axis = figure.axes[0]
    assert axis.get_xlabel() == "attention head"
    assert axis.get_ylabel() == "encoder layer"
    assert "1 of 144 heads are sharply focused" in axis.get_title()
    annotations = " | ".join(text.get_text() for text in axis.texts)
    assert "most focused\nL2H3 · 0.004 nats" in annotations
    assert "most diffuse\nL1H11 · 4.316 nats" in annotations

    caption = _caption_text(figure)
    assert "attention is not causal explanation" in caption
    assert "<s>, </s> and padding excluded" in caption
    plt.close(figure)


def test_every_expected_figure_is_named_by_the_generator() -> None:
    """The three new figures must be in the published set, not only in the code."""
    import scripts.export_figures as figures

    assert len(figures.EXPECTED_FIGURES) == 11
    assert {
        "attention_entropy_atlas.png",
        "embedding_space_3d.png",
        "layer_probe_accuracy.png",
    } <= figures.EXPECTED_FIGURES
