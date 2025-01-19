import os
from dataclasses import dataclass
from datetime import datetime
from itertools import count
from typing import Callable
import numpy as np
import torch
import torch.optim as optim
from ..models.device import device
from ..models.weights_init import weights_init
from torch.utils.tensorboard import SummaryWriter
from ..trainers.utils import clip_grad, detach


@dataclass
class ModelSaver:
    save_iterations: set[int]
    model_name: str = os.environ.get("MODEL_NAME")
    results_folder: str = os.environ.get("RESULTS_FOLDER")
    writer: SummaryWriter = None
    _date: datetime = datetime.strftime(datetime.now(), "%d-%m-%Y, %H-%M-%S")

    @property
    def dir_name(self) -> str:
        return f"{self.model_name} {self._date}"

    @property
    def weights_folder(self) -> str:
        return os.path.join(self.results_folder, "weights/", self.dir_name)

    def __post_init__(self):
        os.makedirs(self.weights_folder, exist_ok=True)
        if self.writer is None:
            self.writer = SummaryWriter(
                log_dir=os.path.join(self.results_folder, "runs/", self.dir_name),
                flush_secs=30,
            )

    def save(self, model: torch.nn.Module, file_name: str):
        path = os.path.join(self.weights_folder, file_name)
        torch.save(model.state_dict(), path)


@dataclass
class Trainer:
    model: torch.nn.Module
    loss: Callable
    batch_size: int = int(os.environ.get("BATCH_SIZE"))
    lr: float = float(os.environ.get("LR"))
    batches_between_plotting: int = int(os.environ.get("BATCHES_BETWEEN_PLOTTING"))
    model_saver: ModelSaver = None
    optimizer: torch.optim.Optimizer = None
    scheduler: torch.optim.lr_scheduler.LRScheduler = None
    scheduler_step: int = 300

    def __post_init__(self):
        weights_init(self.model)
        self.model.train()

        if self.model_saver is None:
            self.model_saver = ModelSaver(save_iterations=set(range(1_500, 800_000, 250)))

        if self.optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.6, 0.999))

        if self.scheduler is None:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.6, patience=500)

    @staticmethod
    def load_(model, file_name):
        model.load_state_dict(torch.load(file_name, map_location=device))
        model.eval()

    def optimize(self, dataset, iteration: int) -> (float, float):
        x, t = dataset.sample(self.batch_size)

        out = self.model(x)
        loss_ = self.loss(t, out)

        # Model weights update
        self.optimizer.zero_grad()
        loss_.backward()
        clip_grad(self.model)
        self.optimizer.step()

        # Validation
        x, t = dataset.sample(self.batch_size, validation=True)
        out = self.model(x)
        loss_val = self.loss(t, out)

        if not iteration % self.scheduler_step:
            self.scheduler.step(loss_val)

        return float(detach(loss_)), float(detach(loss_val))

    def train(
        self,
        dataset,
        max_iter: int = np.inf,
        callbacks: list[Callable] = None,
        metrics: list[Callable] = None,
    ):
        callbacks = [] if callbacks is None else callbacks
        metrics = [] if metrics is None else metrics

        for it in count():
            loss_, loss_val = self.optimize(dataset, it)
            if it in self.model_saver.save_iterations:
                self.model_saver.save(self.model, file_name=f"weights_it={it}_gloss={loss_val}.pt")

            # plot to tensorboard
            writer = self.model_saver.writer
            # self.writer.add_scalar('Metrics/LR', self.scheduler.get_lr()[0], it)
            writer.add_scalar("Metrics/LR", self.scheduler.state_dict()["_last_lr"][0], it)
            if not it % self.batches_between_plotting:
                writer.add_scalar("Optimization/Train Loss", loss_, it)
                writer.add_scalar("Optimization/Validation Loss", loss_val, it)
                [writer.add_scalar(f"Metrics/{name}", metric(self.model, dataset), it) for name, metric in metrics]

                print(f"Iteration {it + 1}:\nLoss={loss_}\nLoss_val={loss_val}", end="\n\n")
                [callback() for callback in callbacks]

            if it >= max_iter:
                break


pass