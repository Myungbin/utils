import logging
from datetime import datetime

import torch
from tqdm import tqdm

from config import cfg, save_model
from .utils import set_logging


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, scaler=None, logger=None):
        """Trainer 클래스의 생성자.

        Args:
            model (nn.Module): 학습할 모델.
            criterion (nn.Module): 손실 함수.
            optimizer (torch.optim.Optimizer): 최적화 함수.
            scheduler (torch.optim.lr_scheduler._LRScheduler): 학습 스케줄러.
            scaler (torch.cuda.amp.GradScaler, optional): Mixed precision(혼합 정밀도)를 사용하는 경우 필요한 스케일러.
            logger (bool, optional): 로깅 여부 (기본값: False).

        """
        self.model = model.to(cfg.DEVICE)
        self.criterion = criterion.to(cfg.DEVICE)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.best_acc = 0

        if logger:
            set_logging('train', self.model, self.criterion, self.optimizer)

    def train(self, train_loader):
        self.model.train()
        train_loss, train_accuracy = 0, 0

        for batch_idx, (data, label) in enumerate(tqdm(train_loader)):
            data, label = data.to(cfg.DEVICE), label.to(cfg.DEVICE)

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
            acc = (prediction.argmax(dim=1) == label).float().mean()
            train_accuracy += acc / len(train_loader)

        return train_loss, train_accuracy

    def validation(self, val_loader):
        self.model.eval()

        validation_loss = 0
        validation_accuracy = 0

        with torch.inference_mode():
            for batch_idx, (data, label) in enumerate(tqdm(val_loader)):
                data, label = data.to(cfg.DEVICE, non_blocking=True), label.to(cfg.DEVICE, non_blocking=True)
                prediction = self.model(data)
                loss = self.criterion(prediction, label)

                validation_loss += loss.item() / len(val_loader)
                acc = (prediction.argmax(dim=1) == label).float().mean()
                validation_accuracy += acc / len(val_loader)

        return validation_loss, validation_accuracy

    def fit(self, train_loader, validation_loader):

        logging.info(f"Train data size: {len(train_loader.dataset)}")
        logging.info(f"Validation data size: {len(validation_loader.dataset)}")

        for epoch in range(cfg.EPOCHS):
            avg_train_loss, train_accuracy = self.train(train_loader)
            avg_val_loss, val_accuracy = self.validation(validation_loader)

            log_msg = (
                f"Epoch [{epoch + 1}/{cfg.EPOCHS}] "
                f"Training Loss: {avg_train_loss:.4f} "
                f"Training Accuracy: {train_accuracy:.4f} "
                f"Validation Loss: {avg_val_loss:.4f} "
                f"Validation Accuracy: {val_accuracy:.4f} "
            )

            logging.info(log_msg)

            if self.scheduler is not None:
                self.scheduler.step()

            if self.best_acc < val_accuracy:
                time = datetime.now().strftime('%Y.%m.%d')
                self.best_acc = val_accuracy
                best_model = self.model
                save_model(best_model, f'{time}{self.model.__class__.__name__}.pth')

        return best_model