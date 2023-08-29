from tqdm import tqdm
import torch
import torch.nn.functional as F

from src.configs.config import cfg


class Trainer:
    def __init__(self, student_model, teacher_model, optimizer):
        self.teacher_model = teacher_model.to(cfg.DEVICE)
        self.student_model = student_model.to(cfg.DEVICE)
        self.optimizer = optimizer

    def train(self, train_loader):
        for batch_idx, (image, label) in enumerate(tqdm(train_loader)):
            image = image.to(cfg.DEVICE, non_blocking=True)

            self.optimizer.zero_grad()

            student_feature = self.student_model(image)

            with torch.no_grad():
                teacher_feature = self.teacher_model(image)

            # Compute the temperature-scaled dot product similarity matrix
            temperature = 0.1
            sim_matrix = torch.matmul(student_feature, teacher_feature.t().detach()) / temperature

            # Compute DINO loss (KL divergence between student and teacher outputs)
            per_example_losses = F.kl_div(torch.log_softmax(sim_matrix, dim=-1),
                                          torch.softmax(sim_matrix.t(), dim=-1).detach(),
                                          reduction="none").sum(dim=1)
            loss = per_example_losses.mean()

            loss.backward()
            self.optimizer.step()

            # Update teacher model with exponential moving average of student model parameters
            with torch.no_grad():
                for student_params, teacher_params in zip(self.student_model.parameters(),
                                                          self.teacher_model.parameters()):
                    teacher_params.data.mul_(cfg.EMA_DECAY).add_(student_params.data, alpha=1 - cfg.EMA_DECAY)
        return loss

    def evaluate(self, validation_loader):
        self.student_model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in tqdm(validation_loader):
                images, labels = images.to(cfg.DEVICE, non_blocking=True), labels.to(cfg.DEVICE, non_blocking=True)

                # Forward pass
                outputs = self.student_model(images)
                predicted = torch.argmax(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                running_loss += loss.item()

        accuracy = 100 * correct / total
        avg_loss = running_loss / len(validation_loader)
        return accuracy, avg_loss, correct

    def fit(self, train_loader, validation_loader):
        for epoch in range(cfg.EPOCHS):
            loss = self.train(train_loader)
            accuracy, avg_loss, correct = self.evaluate(validation_loader)

            print(
                f"Epoch: {epoch + 1}/{cfg.EPOCHS},"
                f" Train Loss: {loss.item()},"
                f" Val Accuracy: {accuracy},"
                f" Val loss:{avg_loss}"
            )
