"""Every committed config validates, and every typo fails at load rather than at epoch 4."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cfg.schema import Config, apply_overrides, load_config

CFG_DIR = Path(__file__).resolve().parent.parent / "cfg"
CONFIG_FILES = sorted(CFG_DIR.glob("*.yaml"))

#: Configs whose numbers may be published, i.e. the ones that were actually run.
RUN_CONFIGS = {"smoke", "dev", "small"}
#: Configs committed as specifications and deliberately not run.
UNRUN_CONFIGS = {"default", "full"}


def test_every_config_file_is_discovered():
    assert {p.stem for p in CONFIG_FILES} == RUN_CONFIGS | UNRUN_CONFIGS


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: p.stem)
def test_config_validates(path):
    cfg = load_config(path)
    assert isinstance(cfg, Config)
    assert path.stem == cfg.NAME
    assert cfg.DESCRIPTION.strip()


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: p.stem)
def test_every_config_has_a_finite_wall_clock_cap(path):
    """No config in this repo may describe an unbounded job."""
    cfg = load_config(path)
    assert 0 < cfg.RUNTIME.WALL_CLOCK_CAP_MIN <= 45.0


def test_unknown_key_is_rejected(tmp_path):
    raw = yaml.safe_load((CFG_DIR / "dev.yaml").read_text())
    raw["MODEL"]["EPOCH"] = 3  # the plausible typo for EPOCHS
    path = tmp_path / "typo.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(Exception, match="EPOCH"):
        load_config(path)


def test_unknown_top_level_block_is_rejected(tmp_path):
    raw = yaml.safe_load((CFG_DIR / "dev.yaml").read_text())
    raw["TRAINING"] = {"EPOCHS": 3}
    path = tmp_path / "extra.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(Exception, match="TRAINING"):
        load_config(path)


def test_sample_larger_than_rows_read_is_rejected(tmp_path):
    raw = yaml.safe_load((CFG_DIR / "dev.yaml").read_text())
    raw["DATA"]["N_TRAIN"] = raw["DATA"]["ROWS_READ_TRAIN"] + 1
    path = tmp_path / "bad_sizes.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(Exception, match="N_TRAIN"):
        load_config(path)


def test_reversed_ngram_range_is_rejected(tmp_path):
    raw = yaml.safe_load((CFG_DIR / "dev.yaml").read_text())
    raw["PREPROCESSING"]["NGRAM_RANGE"] = [2, 1]
    path = tmp_path / "bad_ngram.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(Exception, match="NGRAM_RANGE"):
        load_config(path)


def test_smoke_config_never_fetches_model_weights_or_review_data():
    """NLTK assets are separate; this proves only the model/data paths are local."""
    cfg = load_config(CFG_DIR / "smoke.yaml")
    assert cfg.MODEL.RANDOM_WEIGHT_LAYERS is not None
    assert cfg.RUNTIME.DEVICE == "cpu"
    assert "data/sample" in str(cfg.DATA.TRAIN_PATH)
    assert "data/sample" in str(cfg.DATA.TEST_PATH)


def test_model_revision_is_explicitly_unset_until_an_online_run_resolves_it():
    """No revision hash may be guessed merely to make the schema look pinned."""
    for path in CONFIG_FILES:
        cfg = load_config(path)
        assert cfg.MODEL.REVISION is None


def test_smoke_train_and_test_sources_are_different_files():
    """A fixture with a train/test leak would still 'pass' — this repo does not ship one."""
    cfg = load_config(CFG_DIR / "smoke.yaml")
    assert cfg.DATA.TRAIN_PATH != cfg.DATA.TEST_PATH


def test_default_config_reproduces_the_notebook_exactly():
    """cfg/default.yaml is the notebook's configuration, preserved as executable data."""
    cfg = load_config(CFG_DIR / "default.yaml")
    assert (cfg.DATA.N_TRAIN, cfg.DATA.N_TEST) == (9000, 1000)
    assert (cfg.MODEL.MAX_LEN, cfg.MODEL.BATCH_SIZE, cfg.MODEL.EPOCHS) == (256, 32, 5)
    assert cfg.MODEL.LR == 2e-5
    assert cfg.PREPROCESSING.NGRAM_RANGE == (1, 1)


def test_small_config_matches_the_notebook_data_scale():
    """The published run differs from the notebook in epochs only — nothing else."""
    small = load_config(CFG_DIR / "small.yaml")
    default = load_config(CFG_DIR / "default.yaml")
    assert (small.DATA.N_TRAIN, small.DATA.N_TEST) == (default.DATA.N_TRAIN, default.DATA.N_TEST)
    assert small.MODEL.MAX_LEN == default.MODEL.MAX_LEN
    assert small.MODEL.BATCH_SIZE == default.MODEL.BATCH_SIZE
    assert small.MODEL.LR == default.MODEL.LR
    assert small.MODEL.EPOCHS < default.MODEL.EPOCHS


def test_every_config_has_a_validation_split_except_none():
    """D4: the notebook had no validation split. Every config here does."""
    for path in CONFIG_FILES:
        cfg = load_config(path)
        assert cfg.DATA.VAL_FRACTION > 0, f"{path.stem} has no validation split"


def test_apply_overrides_revalidates(tmp_path):
    cfg = load_config(CFG_DIR / "dev.yaml")
    changed = apply_overrides(cfg, {"PREPROCESSING": {"REMOVE_STOPWORDS": False}})
    assert changed.PREPROCESSING.REMOVE_STOPWORDS is False
    assert cfg.PREPROCESSING.REMOVE_STOPWORDS is True  # frozen: the original is untouched
    with pytest.raises(Exception, match="NOPE"):
        apply_overrides(cfg, {"PREPROCESSING": {"NOPE": 1}})
    with pytest.raises(ValueError, match="unknown config block"):
        apply_overrides(cfg, {"NOSUCHBLOCK": {"X": 1}})


def test_ablation_grid_cells_all_apply_cleanly():
    """cfg/baseline_ablation.json must not name a key the schema does not have."""
    import json

    cfg = load_config(CFG_DIR / "small.yaml")
    grid = json.loads((CFG_DIR / "baseline_ablation.json").read_text())
    assert len(grid["cells"]) == 4
    for cell in grid["cells"]:
        overrides = {k: v for k, v in cell.items() if k.isupper()}
        assert overrides, f"cell {cell['label']!r} overrides nothing"
        apply_overrides(cfg, overrides)
