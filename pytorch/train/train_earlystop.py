import logging

import torch
from torchmetrics.classification import Accuracy, F1Score
from tqdm import tqdm

from src.config.config import CFG, save_model

from .utils import train_log


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, scaler=None, logger=None, patience=20, delta=0.001):
        self.model = model.to(CFG.DEVICE)
        self.criterion = criterion.to(CFG.DEVICE)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.best_loss = 99999

        """Metric"""
        self.acc_metric = Accuracy(task="multiclass", num_classes=CFG.NUM_CLASS).to(CFG.DEVICE)
        self.f1_metric = F1Score(task="multiclass", average="macro", num_classes=CFG.NUM_CLASS).to(CFG.DEVICE)

        """Early Stopping"""
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.early_stop = False

        if logger:
            train_log(self.model, self.criterion, self.optimizer, self.scheduler)

    def _reset_metrics(self):
        self.acc_metric.reset()
        self.f1_metric.reset()

    def _update_metrics(self, prediction, label):
        self.acc_metric.update(prediction, label)
        self.f1_metric.update(prediction, label)

    def _compute_metrics(self):
        accuracy = self.acc_metric.compute()
        f1_score = self.f1_metric.compute()
        return accuracy, f1_score

    def train_step(self, train_loader):
        self.model.train()
        self._reset_metrics()

        train_loss = 0
        for data, label in tqdm(train_loader, desc="Train Loop", leave=False):
            data, label = data.to(CFG.DEVICE, non_blocking=True), label.to(CFG.DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                prediction = self.model(data)
                loss = self.criterion(prediction, label)

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            train_loss += loss.item() / len(train_loader)
            self._update_metrics(prediction, label)

        return train_loss, *self._compute_metrics()

    def validation_step(self, val_loader):
        self.model.eval()
        self._reset_metrics()
        validation_loss = 0

        with torch.inference_mode():
            for data, label in tqdm(val_loader, desc="Validation Loop", leave=False):
                data, label = data.to(CFG.DEVICE, non_blocking=True), label.to(CFG.DEVICE, non_blocking=True)
                prediction = self.model(data)
                loss = self.criterion(prediction, label)

                validation_loss += loss.item() / len(val_loader)
                self._update_metrics(prediction, label)

        return validation_loss, *self._compute_metrics()

    def fit(self, train_loader, validation_loader):
        for epoch in range(CFG.EPOCHS):
            train_loss, train_accuracy, train_f1_score = self.train_step(train_loader)
            val_loss, val_accuracy, val_f1_score = self.validation_step(validation_loader)

            log_msg = (
                f"Epoch [{epoch + 1}/{CFG.EPOCHS}] "
                f"Training Loss: {train_loss:.4f} "
                f"Training Accuracy: {train_accuracy:.4f} "
                f"Training F1-score: {train_f1_score:.4f} "
                f"Validation Loss: {val_loss:.4f} "
                f"Validation Accuracy: {val_accuracy:.4f} "
                f"Validation F1-score: {val_f1_score:.4f} "
            )

            logging.info(log_msg)

            if self.scheduler is not None:
                self.scheduler.step()

            """Early Stop Logic"""
            if val_loss < self.best_loss - self.delta:
                print("Validation loss decreased, saving model")
                self.best_loss = val_loss
                best_model = self.model
                model_name = self.model.__class__.__name__
                save_model_name = f"{model_name}/{model_name}_{epoch + 1}Epoch.pth"
                save_model(best_model, save_model_name)
                self.counter = 0
            else:
                self.counter += 1
                print(f"Early Stopping Counter {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    logging.info(f"Early Stopping Counter {self.counter}/{self.patience}")
                    logging.info("Early stopping triggered")
                    self.early_stop = True
                    break

        return best_model if not self.early_stop else None
