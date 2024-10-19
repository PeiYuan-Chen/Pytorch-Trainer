import logging
from typing import Sequence, Mapping, Callable

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import Metric, MetricCollection, MeanMetric
from accelerate import Accelerator
from omegaconf import DictConfig
from hydra.utils import instantiate


from .callbacks import Callback
from .trainer_state import TrainerState
from .trainer_args import TrainerArguments
from .trainer_utils import IntervalStrategy, enable_full_determinism, set_seed

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        args: TrainerArguments,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        eval_loader: DataLoader | None = None,
        lr_scheduler: _LRScheduler | None = None,
        input_getter: Callable | None = None,
        target_getter: Callable | None = None,
        output_getter: Callable | None = None,
        metrics: Metric | Sequence[Metric] | None = None,
        callbacks: Sequence[Callback] | None = None,
    ):
        self.args = args
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.lr_scheduler = lr_scheduler
        self.input_getter = input_getter or (lambda x: x)
        self.target_getter = target_getter or (lambda x: x)
        self.output_getter = output_getter or (lambda x: x)
        # metrics
        self.train_loss_avg = MeanMetric()
        self.eval_loss_avg = MeanMetric() if eval_loader is not None else None
        self.metrics = (
            MetricCollection(metrics, prefix="eval_")
            if eval_loader is not None and metrics is not None
            else None
        )
        # state
        self.state = TrainerState()
        # callbacks
        self.callbacks = callbacks or []
        # accelerator
        self.accelerator: Accelerator | None = None

    @classmethod
    def from_hydra_config(cls, cfg: DictConfig):
        args = instantiate(cfg.get("args"))
        model = instantiate(cfg.get("model"))
        loss_fn = instantiate(cfg.get("loss_fn"))
        optimizer = instantiate(cfg.get("optimizer"), params=model.parameters())
        lr_scheduler = instantiate(cfg.get("lr_scheduler"), optimizer=optimizer)
        train_loader = instantiate(cfg.get("train_loader"))
        eval_loader = instantiate(cfg.get("eval_loader"))
        input_getter = instantiate(cfg.get("input_getter"))
        target_getter = instantiate(cfg.get("target_getter"))
        output_getter = instantiate(cfg.get("output_getter"))
        metrics = instantiate(cfg.get("metrics"))
        callbacks = instantiate(cfg.get("callbacks"))

        return cls(
            args,
            model,
            loss_fn,
            optimizer,
            train_loader,
            eval_loader,
            lr_scheduler,
            input_getter,
            target_getter,
            output_getter,
            metrics,
            callbacks,
        )

    def __enter__(self) -> "Trainer":
        # deterministic
        if self.args.full_determinism:
            enable_full_determinism(self.args.seed)
        else:
            set_seed(self.args.seed)

        # accelerator
        self.accelerator = Accelerator()
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.train_loss_avg,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
            self.train_loss_avg,
        )
        if self.lr_scheduler is not None:
            self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)

        if self.eval_loader is not None:
            self.eval_loader, self.eval_loss_avg = self.accelerator.prepare(
                self.eval_loader, self.eval_loss_avg
            )

        if self.metrics is not None:
            self.metrics = self.accelerator.prepare(self.metrics)

        # callbacks
        for callback in self.callbacks:
            callback.setup(
                self.accelerator, self.model, self.train_loader, self.eval_loader
            )

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.accelerator.end_training()
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.train_loss_avg,
        ) = self.accelerator.free_memory(
            self.model,
            self.optimizer,
            self.train_loader,
            self.train_loss_avg,
        )

        if self.lr_scheduler is not None:
            self.lr_scheduler = self.accelerator.free_memory(self.lr_scheduler)

        if self.eval_loader is not None:
            self.eval_loader, self.eval_loss_avg = self.accelerator.free_memory(
                self.eval_loader, self.eval_loss_avg
            )

        if self.metrics is not None:
            self.metrics = self.accelerator.free_memory(self.metrics)

    def train(self):
        self._on_train_init()
        self._on_train_begin()

        for epoch_idx in range(self.state.epoch, self.args.epochs):
            self._train_epoch(epoch_idx)

        self._on_train_end()

    def _forward(self, batch: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        inputs = self.input_getter(batch)
        targets = self.target_getter(batch)
        if isinstance(inputs, Mapping):
            outputs = self.model(**inputs)
        elif isinstance(inputs, Sequence):
            outputs = self.model(*inputs)
        else:
            outputs = self.model(inputs)
        outputs = self.output_getter(outputs)
        loss = self.loss_fn(outputs, targets)

        return loss, outputs, targets

    def _train_step(self, batch):
        self.model.train()
        # gradient accumulation
        with self.accelerator.accumulate(self.model):
            # forward
            loss, _, _ = self._forward(batch)
            # metrics
            self.train_loss_avg.update(loss.detach())
            # backward
            self.accelerator.backward(loss)

            # step
            if self.accelerator.sync_gradients:
                # gradient clip
                if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm,
                    )
                    # deepspeed does its own clipping?
                # update weight
                self.optimizer.step()
                self.optimizer.zero_grad()
                # update lr
                if not self.accelerator.optimizer_step_was_skipped:
                    if self.lr_scheduler is not None and not isinstance(
                        self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        self.lr_scheduler.step()

                self._on_step_end()

    def _train_epoch(self, epoch_idx: int):
        self._on_epoch_begin()

        for batch in self.train_loader:
            self._train_step(batch)

        self._on_epoch_end()

    @torch.inference_mode()
    def _evaluate_step(self, batch):
        loss, outputs, targets = self._forward(batch)
        # metrics
        self.eval_loss_avg.update(loss)
        self.metrics.update(outputs, targets)

        self._on_eval_step_end()

    def _evaluate(self):
        self._on_eval_begin()

        self.model.eval()
        self.eval_loss_avg.reset()
        self.metrics.reset()

        for batch in self.eval_loader:
            self._evaluate_step(batch)

        self._on_eval_end()

    def _on_train_init(self) -> None:
        for callback in self.callbacks:
            callback.on_train_init(self.args, self.state)

    def _on_train_begin(self) -> None:
        logger.info("Training Begin")

        for callback in self.callbacks:
            callback.on_train_begin(self.args, self.state)

    def _on_train_end(self) -> None:
        for callback in self.callbacks:
            callback.on_train_end(self.args, self.state)

        logger.info("Training End")

    def _on_epoch_begin(self):
        for callback in self.callbacks:
            callback.on_epoch_begin(self.args, self.state)

    def _on_epoch_end(self):
        self.state.epoch += 1

        for callback in self.callbacks:
            callback.on_epoch_end(self.args, self.state)

        # log
        if (
            self.accelerator.is_main_process
            and self.args.log_strategy == IntervalStrategy.EPOCH
            and self.state.epoch % self.args.log_interval == 0
        ):
            logs = {
                "lr": self.optimizer.param_groups[0]["lr"],
                "train_loss": self.train_loss_avg.compute(),
            }
            self.train_loss_avg.reset()
            self._log(logs)

        # evaluation
        if (
            self.eval_loader is not None
            and self.args.eval_strategy == IntervalStrategy.EPOCH
            and self.state.epoch % self.args.eval_interval == 0
        ):

            self._evaluate()

        # save
        if (
            self.accelerator.is_main_process
            and self.args.save_strategy == IntervalStrategy.EPOCH
            and self.state.epoch % self.args.save_interval == 0
        ):
            self._save()

    def _on_step_end(self):
        self.state.global_step += 1

        for callback in self.callbacks:
            callback.on_step_end(self.args, self.state)
        # log
        if (
            self.accelerator.is_main_process
            and self.args.log_strategy == IntervalStrategy.STEP
            and self.state.global_step % self.args.log_interval == 0
        ):
            logs = {
                "lr": self.optimizer.param_groups[0]["lr"],
                "train_loss": self.train_loss_avg.compute(),
            }
            self.train_loss_avg.reset()
            self._log(logs)

        # evaluation
        if (
            self.args.eval_strategy == IntervalStrategy.STEP
            and self.state.global_step % self.args.eval_interval == 0
        ):
            self.evaluation()

        # save
        if (
            self.accelerator.is_main_process
            and self.args.save_strategy == IntervalStrategy.STEP
            and self.state.global_step % self.args.save_interval == 0
        ):
            self._save()

    def _on_eval_begin(self):
        for callback in self.callbacks:
            callback.on_eval_begin(self.args, self.state)

    def _on_eval_step_end(self):
        for callback in self.callbacks:
            callback.on_eval_step_end(self.args, self.state)

    def _on_eval_end(self):
        for callback in self.callbacks:
            callback.on_eval_end(self.args, self.state)

        logs = {
            "eval_loss": self.eval_loss_avg.compute(),
            **self.metrics.compute(),
        }

        # lr scheduler
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            try:
                metric_value = logs[self.args.monitor]
            except KeyError as ex:
                raise KeyError(
                    f"Evaluation logs does not contain the metric {self.args.monitor}."
                ) from ex
            self.lr_scheduler.step(metrics=metric_value)

        # log
        self._log(logs)

    def _log(self, logs: dict):
        for callback in self.callbacks:
            callback.on_log(self.args, self.state, logs)

    def _save(self):
        for callback in self.callbacks:
            callback.on_save(self.args, self.state)
