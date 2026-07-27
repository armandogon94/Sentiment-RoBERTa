from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest


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
