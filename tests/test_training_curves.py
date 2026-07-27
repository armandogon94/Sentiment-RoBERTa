from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import scripts.export_figures as figures


def _published_metrics() -> dict[str, Any]:
    return {
        "config_name": "small",
        "seed": 1337,
        "device": "mps",
        "power_mode": "Low Power Mode OFF",
        "models": {
            "roberta": {
                "n_train": 8100,
                "n_val": 900,
                "training": {
                    "epochs_run": 3,
                    "epochs_configured": 3,
                    "history": [
                        {
                            "epoch": 1.0,
                            "train_loss": 0.22399002043750343,
                            "val_loss": 0.1238326669253152,
                            "val_accuracy": 0.9455555555555556,
                        },
                        {
                            "epoch": 2.0,
                            "train_loss": 0.1001160602107292,
                            "val_loss": 0.17931287751757893,
                            "val_accuracy": 0.9388888888888889,
                        },
                        {
                            "epoch": 3.0,
                            "train_loss": 0.06197184531612131,
                            "val_loss": 0.15632417052984238,
                            "val_accuracy": 0.9455555555555556,
                        },
                    ],
                    "selected_epoch": 1,
                    "selection_criterion": "min validation loss",
                    "wall_clock_capped": False,
                },
            }
        },
    }


def _render_training_curves(monkeypatch: Any, tmp_path: Path) -> Any:
    captured: dict[str, Any] = {}

    def capture_figure(
        figure: Any,
        name: str,
        _out_dirs: list[Path],
        *,
        show: bool = False,
    ) -> list[Path]:
        assert name == "training_curves"
        assert show is False
        captured["figure"] = figure
        return [tmp_path / f"{name}.png"]

    monkeypatch.setattr(figures.plots, "save_figure", capture_figure)
    figures.figure_training_curves(_published_metrics(), [tmp_path])
    return captured["figure"]


def test_training_curve_keeps_one_finding_title_and_existing_visual_contract(
    monkeypatch: Any, tmp_path: Path
) -> None:
    figure = _render_training_curves(monkeypatch, tmp_path)
    loss_axis, accuracy_axis = figure.axes

    assert figure._suptitle.get_text() == "Validation loss rose after epoch 1"
    assert loss_axis.get_title() == ""
    assert accuracy_axis.get_title() == ""
    assert loss_axis.get_ylabel() == "loss (cross entropy)"
    assert accuracy_axis.get_ylabel() == "validation accuracy"
    assert accuracy_axis.get_xlabel() == "training epoch"
    assert list(accuracy_axis.get_xticks()) == [1, 2, 3]

    lines_by_label = {line.get_label(): line for line in loss_axis.lines}
    training_line = lines_by_label["training loss"]
    validation_line = lines_by_label["validation loss"]
    assert training_line.get_marker() == "o"
    assert training_line.get_linestyle() == "-"
    assert validation_line.get_marker() == "s"
    assert validation_line.get_linestyle() == "--"
    assert loss_axis.get_legend_handles_labels()[1] == [
        "training loss",
        "validation loss",
    ]

    accuracy_line = next(line for line in accuracy_axis.lines if line.get_marker() == "^")
    assert accuracy_line.get_marker() == "^"
    assert accuracy_line.get_linestyle() == ":"
    assert loss_axis.get_position().y0 > accuracy_axis.get_position().y0
    plt.close(figure)


def test_selection_annotation_stays_attached_to_rule_and_clear_of_loss_lines(
    monkeypatch: Any, tmp_path: Path
) -> None:
    figure = _render_training_curves(monkeypatch, tmp_path)
    loss_axis = figure.axes[0]
    selection = next(
        text for text in loss_axis.texts if text.get_text() == "selected: lowest validation loss"
    )

    selected_epoch = 1
    text_x, text_y = selection.get_position()
    assert selected_epoch < text_x <= selected_epoch + 0.1
    assert text_y <= 0.15

    figure.canvas.draw()
    text_box = selection.get_window_extent(figure.canvas.get_renderer())
    plotted_lines = [
        line for line in loss_axis.lines if line.get_label() in {"training loss", "validation loss"}
    ]
    for line in plotted_lines:
        x_values = np.asarray(line.get_xdata(), dtype=float)
        y_values = np.asarray(line.get_ydata(), dtype=float)
        for start in range(len(x_values) - 1):
            weights = np.linspace(0.0, 1.0, 501)
            segment = np.column_stack(
                (
                    x_values[start] + weights * (x_values[start + 1] - x_values[start]),
                    y_values[start] + weights * (y_values[start + 1] - y_values[start]),
                )
            )
            display_points = loss_axis.transData.transform(segment)
            overlaps_text = (
                (display_points[:, 0] >= text_box.x0)
                & (display_points[:, 0] <= text_box.x1)
                & (display_points[:, 1] >= text_box.y0)
                & (display_points[:, 1] <= text_box.y1)
            )
            assert not overlaps_text.any()

    plt.close(figure)
