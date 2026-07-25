"""Fine-tuning ``roberta-base``, with the bounds and the fixes the notebook lacked.

Four things here are deliberate and not obvious:

1. **``attn_implementation="eager"`` is passed to the MODEL, not the config.** The notebook
   passed it to ``RobertaConfig.from_pretrained``, where ``transformers`` never reads it — it
   reads the private ``config._attn_implementation``. On ``transformers`` 5.x the default is
   ``sdpa``, and ``sdpa`` returns an **empty** attentions tuple with only a warning, so the
   attention figures would have silently had nothing to plot.

2. **One device per process.** Resolved once by ``utils.device.resolve_device`` and never
   changed. A verified failure on this torch build is a CPU transformer loop deadlocking at
   0% CPU after an earlier MPS matmul in the same process
   (``docs/adr/0003-mps-constraints.md``).

3. **A validation split and a hard wall-clock cap.** The epoch is selected on validation
   loss, the best checkpoint is kept, and the deadline is checked before every optimizer
   step. A truncated run records its partial epoch and ``wall_clock_capped: true``.

4. **Epoch-1 wall clock and the projected total are reported before epoch 2 begins**, so a
   multi-hour job is visible before it is committed to rather than after.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.torch_dataset import ReviewsDataset
from models.registry import register
from utils.logging import get_logger
from utils.seeding import seed_worker, torch_generator

log = get_logger(__name__)


class _WallClockCapReached(RuntimeError):
    """Internal control flow carrying honest partial-epoch progress."""

    def __init__(self, *, mean_loss: float | None, seen: int, steps_run: int) -> None:
        super().__init__("WALL_CLOCK_CAP_MIN reached inside training epoch")
        self.mean_loss = mean_loss
        self.seen = seen
        self.steps_run = steps_run


class RobertaSentiment:
    """Fine-tuned ``roberta-base`` sequence classifier.

    Satisfies ``models.protocols.SentimentModel``. ``fit`` trains with no validation set;
    ``fit_with_validation`` is what ``train.py`` actually calls.
    """

    def __init__(
        self,
        *,
        pretrained: str = "roberta-base",
        revision: str | None = None,
        num_labels: int = 2,
        max_len: int = 256,
        batch_size: int = 32,
        epochs: int = 5,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        seed: int = 1337,
        device: torch.device | None = None,
        wall_clock_cap_min: float = 45.0,
        log_every_steps: int = 25,
        num_workers: int = 0,
        random_weight_layers: int | None = None,
        name: str = "roberta",
    ) -> None:
        self.name = name
        self.pretrained = pretrained
        self.revision = revision
        self.num_labels = num_labels
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = device or torch.device("cpu")
        self.wall_clock_cap_min = wall_clock_cap_min
        self.log_every_steps = log_every_steps
        self.num_workers = num_workers
        self.random_weight_layers = random_weight_layers

        self.tokenizer = self._build_tokenizer()
        self.model = self._build_model()
        self.history: list[dict[str, float]] = []
        self.train_report: dict[str, Any] = {}
        self._fitted = False

    # -- construction ---------------------------------------------------------------

    def _build_tokenizer(self) -> Any:
        """Real tokenizer for real runs; an offline hashing tokenizer for the smoke path."""
        if self.random_weight_layers is not None:
            from models.hash_tokenizer import HashTokenizer

            return HashTokenizer(vocab_size=2048)
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.pretrained, revision=self.revision)

    def _build_model(self) -> Any:
        from transformers import RobertaConfig, RobertaForSequenceClassification

        if self.random_weight_layers is not None:
            # Local, tiny, random. No network, no pretrained weights. Smoke path only.
            # num_labels reaches PretrainedConfig through **kwargs; it is absent from the
            # RobertaConfig signature, so mypy cannot see it.
            config = RobertaConfig(  # type: ignore[call-arg]
                vocab_size=getattr(self.tokenizer, "vocab_size", 2048),
                hidden_size=64,
                num_hidden_layers=self.random_weight_layers,
                num_attention_heads=2,
                intermediate_size=128,
                max_position_embeddings=self.max_len + 8,
                num_labels=self.num_labels,
                pad_token_id=1,
                bos_token_id=0,
                eos_token_id=2,
            )
            model = RobertaForSequenceClassification._from_config(
                config, attn_implementation="eager"
            )
        else:
            # D8: attn_implementation belongs on the MODEL. On the config it is a no-op and
            # transformers 5.x silently keeps sdpa, which returns no attentions at all.
            # The transformers model stub types revision as str, although runtime accepts None.
            model = RobertaForSequenceClassification.from_pretrained(
                self.pretrained,
                revision=self.revision,  # type: ignore[arg-type]
                num_labels=self.num_labels,
                attn_implementation="eager",
            )
        if model.config._attn_implementation != "eager":
            raise RuntimeError(
                f"expected eager attention, got {model.config._attn_implementation!r}; "
                "attention figures would be empty"
            )
        return model.to(self.device)

    # -- Protocol -------------------------------------------------------------------

    def fit(self, texts: list[str], labels: list[int]) -> RobertaSentiment:
        return self.fit_with_validation(texts, labels, [], [])

    def predict(self, texts: list[str]) -> np.ndarray:
        logits = self.predict_logits(texts)
        return np.asarray(logits.argmax(axis=1), dtype=np.int64)

    def save(self, path: Path) -> Path:
        """Save the fine-tuned weights. Gitignored — a run artifact."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        return path

    # -- training -------------------------------------------------------------------

    def make_dataset(self, texts: list[str], labels: list[int]) -> ReviewsDataset:
        return ReviewsDataset(texts, labels, self.tokenizer, self.max_len)

    def fit_with_validation(
        self,
        texts: list[str],
        labels: list[int],
        val_texts: list[str],
        val_labels: list[int],
    ) -> RobertaSentiment:
        """Train, selecting the epoch on validation loss and respecting the wall-clock cap."""
        train_ds = self.make_dataset(texts, labels)
        val_ds = self.make_dataset(val_texts, val_labels) if val_texts else None

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            generator=torch_generator(self.seed),
            worker_init_fn=seed_worker,
        )
        val_loader = (
            DataLoader(
                val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
            )
            if val_ds is not None
            else None
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        cap_seconds = self.wall_clock_cap_min * 60.0
        started = time.perf_counter()

        best_val = float("inf")
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        capped = False
        partial_epoch: dict[str, Any] | None = None

        log.info(
            "train.start",
            device=str(self.device),
            n_train=len(train_ds),
            n_val=0 if val_ds is None else len(val_ds),
            steps_per_epoch=len(train_loader),
            epochs=self.epochs,
            wall_clock_cap_min=self.wall_clock_cap_min,
        )

        for epoch in range(1, self.epochs + 1):
            elapsed = time.perf_counter() - started
            if epoch > 1:
                mean_epoch = elapsed / (epoch - 1)
                projected_total = mean_epoch * self.epochs
                log.info(
                    "train.projection",
                    completed_epochs=epoch - 1,
                    elapsed_s=round(elapsed, 1),
                    mean_epoch_s=round(mean_epoch, 1),
                    projected_total_s=round(projected_total, 1),
                    projected_total_min=round(projected_total / 60.0, 1),
                    cap_min=self.wall_clock_cap_min,
                )
                if elapsed + mean_epoch > cap_seconds:
                    log.warning(
                        "train.capped",
                        reason="next epoch projected to exceed WALL_CLOCK_CAP_MIN",
                        elapsed_min=round(elapsed / 60.0, 1),
                        next_epoch_min=round(mean_epoch / 60.0, 1),
                        cap_min=self.wall_clock_cap_min,
                        epochs_run=epoch - 1,
                        epochs_configured=self.epochs,
                    )
                    capped = True
                    break

            epoch_started = time.perf_counter()
            try:
                train_loss = self._train_one_epoch(
                    train_loader,
                    optimizer,
                    epoch,
                    deadline=started + cap_seconds,
                )
            except _WallClockCapReached as stopped:
                capped = True
                partial_epoch = {
                    "epoch": epoch,
                    "train_loss": stopped.mean_loss,
                    "examples_seen": stopped.seen,
                    "steps_run": stopped.steps_run,
                    "steps_total": len(train_loader),
                    "epoch_seconds": time.perf_counter() - epoch_started,
                }
                log.warning(
                    "train.capped_inside_epoch",
                    epoch=epoch,
                    steps_run=stopped.steps_run,
                    steps_total=len(train_loader),
                    examples_seen=stopped.seen,
                    cap_min=self.wall_clock_cap_min,
                )
                break
            val_loss, val_acc = (
                self._evaluate_loss(val_loader)
                if val_loader is not None
                else (float("nan"), float("nan"))
            )
            epoch_seconds = time.perf_counter() - epoch_started

            self.history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "epoch_seconds": epoch_seconds,
                }
            )
            log.info(
                "train.epoch",
                epoch=epoch,
                train_loss=round(train_loss, 4),
                val_loss=None if val_loss != val_loss else round(val_loss, 4),
                val_accuracy=None if val_acc != val_acc else round(val_acc, 4),
                epoch_s=round(epoch_seconds, 1),
            )

            # D4: select on validation loss. With no validation set, keep the last epoch and
            # say so — there is nothing to select on, and using test loss would be leakage.
            score = val_loss if val_loss == val_loss else float(epoch)
            if val_loader is None or score < best_val:
                best_val = score
                best_epoch = epoch
                best_state = {
                    k: v.detach().to("cpu").clone() for k, v in self.model.state_dict().items()
                }

        total_seconds = time.perf_counter() - started
        if best_state is not None and (
            partial_epoch is not None or best_epoch != len(self.history)
        ):
            self.model.load_state_dict(best_state)
            log.info("train.restored_best", epoch=best_epoch, val_loss=round(best_val, 4))

        selection_criterion = (
            "partial epoch only (no validation checkpoint)"
            if best_epoch == 0 and partial_epoch is not None
            else ("min validation loss" if val_loader is not None else "last epoch (no val split)")
        )
        self.train_report = {
            "epochs_configured": self.epochs,
            "epochs_run": len(self.history),
            "selected_epoch": best_epoch,
            "selection_criterion": selection_criterion,
            "wall_clock_capped": capped,
            "wall_clock_cap_min": self.wall_clock_cap_min,
            "train_seconds": total_seconds,
            "epoch_seconds": [h["epoch_seconds"] for h in self.history],
            "history": self.history,
            "partial_epoch": partial_epoch,
            "truncation_train": train_ds.truncation_report(),
            "steps_per_epoch": len(train_loader),
            "device": str(self.device),
            "attn_implementation": self.model.config._attn_implementation,
            "n_parameters": int(sum(p.numel() for p in self.model.parameters())),
        }
        self._fitted = True
        return self

    def _train_one_epoch(
        self,
        loader: DataLoader[Any],
        optimizer: torch.optim.Optimizer,
        epoch: int,
        *,
        deadline: float | None = None,
    ) -> float:
        self.model.train()
        total = 0.0
        seen = 0
        for step, batch in enumerate(loader, start=1):
            if deadline is not None and time.perf_counter() >= deadline:
                raise _WallClockCapReached(
                    mean_loss=(total / seen if seen else None),
                    seen=seen,
                    steps_run=step - 1,
                )
            optimizer.zero_grad(set_to_none=True)
            labels = batch["labels"].to(self.device)
            out = self.model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                labels=labels,
            )
            out.loss.backward()
            optimizer.step()
            batch_size = int(labels.shape[0])
            total += float(out.loss.detach().item()) * batch_size
            seen += batch_size
            if step % self.log_every_steps == 0:
                log.info(
                    "train.step",
                    epoch=epoch,
                    step=step,
                    of=len(loader),
                    loss=round(total / seen, 4),
                )
        return total / seen

    @torch.no_grad()
    def _evaluate_loss(self, loader: DataLoader[Any]) -> tuple[float, float]:
        self.model.eval()
        total = 0.0
        correct = 0
        seen = 0
        for batch in loader:
            labels = batch["labels"].to(self.device)
            out = self.model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                labels=labels,
            )
            batch_size = int(labels.shape[0])
            total += float(out.loss.detach().item()) * batch_size
            correct += int((out.logits.argmax(dim=-1) == labels).sum().item())
            seen += batch_size
        return total / seen, correct / seen

    # -- inference ------------------------------------------------------------------

    @torch.no_grad()
    def predict_logits(self, texts: list[str]) -> np.ndarray:
        """Logits for ``texts``. Batched at the training batch size."""
        self.model.eval()
        ds = self.make_dataset(texts, [0] * len(texts))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        chunks: list[np.ndarray] = []
        for batch in loader:
            out = self.model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
            )
            chunks.append(out.logits.detach().to("cpu").numpy())
        return np.concatenate(chunks, axis=0)

    def evaluate_truncation(self, texts: list[str]) -> dict[str, float]:
        """Truncation report for an arbitrary text set (the test split, in practice)."""
        return self.make_dataset(texts, [0] * len(texts)).truncation_report()


@register("roberta")
def _create_roberta(**kwargs: Any) -> RobertaSentiment:
    return RobertaSentiment(**kwargs)
