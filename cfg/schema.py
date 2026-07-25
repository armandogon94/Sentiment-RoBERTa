"""Pydantic schema for ``cfg/*.yaml``. A config typo fails at load, not at epoch 4.

House style (``REFERENCE-STYLE-GUIDE.md`` §1.2, dialect A): SCREAMING_SNAKE keys nested
under fixed top-level blocks, with key names chosen to match constructor kwargs so a
block can be splatted straight into a callable.

``extra="forbid"`` everywhere is the point of the file. A misspelled ``EPOCH`` would
otherwise be silently ignored and the run would quietly train for the default number of
epochs while the config claimed otherwise — a reproducibility hole that looks like a
result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Block(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataCfg(_Block):
    """Which rows to read, and how many of them to actually use.

    ``ROWS_READ_*`` and ``N_*`` are separate knobs on purpose. The source notebook read
    200,000 rows and then sampled 9,000 from them, so "200K" and "9K" were a coincidence
    of two unrelated literals. Naming them separately makes the training scale explicit.
    """

    TRAIN_PATH: Path
    TEST_PATH: Path
    ROWS_READ_TRAIN: int = Field(gt=0, description="rows parsed from TRAIN_PATH before sampling")
    ROWS_READ_TEST: int = Field(gt=0, description="rows parsed from TEST_PATH before sampling")
    N_TRAIN: int = Field(gt=0, description="rows actually trained on (train + val)")
    N_TEST: int = Field(gt=0, description="rows in the held-out test set")
    VAL_FRACTION: Annotated[float, Field(ge=0.0, lt=0.5)] = 0.1

    @model_validator(mode="after")
    def _sample_fits(self) -> DataCfg:
        if self.N_TRAIN > self.ROWS_READ_TRAIN:
            raise ValueError(f"N_TRAIN={self.N_TRAIN} > ROWS_READ_TRAIN={self.ROWS_READ_TRAIN}")
        if self.N_TEST > self.ROWS_READ_TEST:
            raise ValueError(f"N_TEST={self.N_TEST} > ROWS_READ_TEST={self.ROWS_READ_TEST}")
        return self


class PreprocessingCfg(_Block):
    """The TF-IDF chain, with every destructive step behind an explicit flag.

    ``REMOVE_STOPWORDS`` and ``ALPHANUMERIC_ONLY`` both delete negation: NLTK's English
    stopword list contains ``not``/``no``/``nor``, and the ``^\\w+$`` filter destroys
    ``n't`` before the stopword filter runs. They are flags rather than hardcoded steps so
    the cost can be *measured* (``cfg/baseline_ablation.json``) instead of argued about.
    """

    LOWERCASE: bool = True
    ALPHANUMERIC_ONLY: bool = True
    REMOVE_STOPWORDS: bool = True
    STEM: bool = True
    NGRAM_RANGE: tuple[int, int] = (1, 1)
    MAX_FEATURES: int | None = None

    @model_validator(mode="after")
    def _ngram_ordered(self) -> PreprocessingCfg:
        lo, hi = self.NGRAM_RANGE
        if not 1 <= lo <= hi:
            raise ValueError(f"NGRAM_RANGE must satisfy 1 <= lo <= hi, got {self.NGRAM_RANGE}")
        return self


class ModelCfg(_Block):
    """Transformer hyperparameters, named as in the source notebook."""

    NAME: Literal["roberta"] = "roberta"
    PRETRAINED: str = "roberta-base"
    REVISION: str | None = None
    MAX_LEN: int = Field(default=256, gt=0, le=512)
    BATCH_SIZE: int = Field(default=32, gt=0)
    EPOCHS: int = Field(default=5, gt=0)
    LR: float = Field(default=2e-5, gt=0)
    WEIGHT_DECAY: float = Field(default=0.01, ge=0)
    NUM_LABELS: int = Field(default=2, ge=2)
    #: When set, build a tiny randomly-initialised model with this many layers instead of
    #: downloading pretrained weights. Used by cfg/smoke.yaml so CI does not contact the
    #: Hugging Face hub; the TF-IDF path still has documented NLTK prerequisites.
    RANDOM_WEIGHT_LAYERS: int | None = None


class BaselineCfg(_Block):
    """TF-IDF + logistic regression control."""

    C: float = Field(default=1.0, gt=0)
    MAX_ITER: int = Field(default=1000, gt=0)


class RuntimeCfg(_Block):
    """Device selection and the hard compute bound.

    ``WALL_CLOCK_CAP_MIN`` is checked inside the training loop. A run can exceed the cap by
    at most one in-flight optimizer step, then stops cleanly and records a partial epoch.
    """

    DEVICE: Literal["auto", "mps", "cpu"] = "auto"
    WALL_CLOCK_CAP_MIN: float = Field(default=45.0, gt=0)
    LOG_EVERY_STEPS: int = Field(default=25, gt=0)
    NUM_WORKERS: int = Field(default=0, ge=0)


class ResultsCfg(_Block):
    OUTPUT_DIR: Path = Path("runs")
    SAVE_FIGURES: bool = True


class Config(BaseModel):
    """A whole ``cfg/*.yaml`` file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    NAME: str
    DESCRIPTION: str
    SEED: int = 1337
    DATA: DataCfg
    PREPROCESSING: PreprocessingCfg
    MODEL: ModelCfg
    BASELINE: BaselineCfg
    RUNTIME: RuntimeCfg
    RESULTS: ResultsCfg


def load_config(path: str | Path) -> Config:
    """Parse and validate one YAML config. Raises on any unknown or malformed key."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return Config.model_validate(raw)


def apply_overrides(cfg: Config, overrides: dict[str, dict[str, object]]) -> Config:
    """Return a copy of ``cfg`` with block-level overrides applied.

    Used by the ablation driver, which varies only ``PREPROCESSING`` keys. Because every
    block is frozen, the override goes through ``model_validate`` again — so an ablation
    cell that names a nonexistent key fails at cell-construction time, not silently.
    """
    payload = cfg.model_dump(mode="json")
    for block, values in overrides.items():
        if block not in payload:
            raise ValueError(f"unknown config block {block!r}")
        if not isinstance(payload[block], dict):
            raise ValueError(f"{block!r} is a scalar, not a block")
        payload[block] = {**payload[block], **values}
    return Config.model_validate(payload)
