"""Small shared helpers for the CO5085 Fashion-MNIST exercises."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FOLDER = Path("data")
REPORT_FOLDER = Path("reports")
FIGURE_FOLDER = REPORT_FOLDER / "figures"
CHECKPOINT_FOLDER = REPORT_FOLDER / "checkpoints"
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def set_seed(seed=SEED):
    """Make the split and initial weights repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_split_indices(total=60_000, seed=SEED):
    """Split the official training set into 80% train and 20% validation."""
    ids = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).tolist()
    split = int(total * 0.8)
    return ids[:split], ids[split:]


def get_loaders(batch_size=256):
    """Download Fashion-MNIST once and return train, validation, and test loaders."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    full_train = datasets.FashionMNIST(DATA_FOLDER, train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(DATA_FOLDER, train=False, download=True, transform=transform)
    train_ids, validation_ids = make_split_indices(len(full_train))
    options = {"batch_size": batch_size, "num_workers": 0, "pin_memory": DEVICE.type == "cuda"}
    train_loader = DataLoader(Subset(full_train, train_ids), shuffle=True, **options)
    validation_loader = DataLoader(Subset(full_train, validation_ids), shuffle=False, **options)
    test_loader = DataLoader(test_data, shuffle=False, **options)
    return train_loader, validation_loader, test_loader


def train_epoch(model, loader, loss_function, optimizer):
    """One explicit PyTorch training loop: forward, loss, backward, and step."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        scores = model(images)
        loss = loss_function(scores, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += (scores.argmax(dim=1) == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


def evaluate(model, loader, loss_function):
    """Measure loss and accuracy without changing model weights."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            scores = model(images)
            total_loss += loss_function(scores, labels).item() * len(labels)
            prediction = scores.argmax(dim=1)
            correct += (prediction == labels).sum().item()
            total += len(labels)
            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(prediction.cpu().tolist())

    return total_loss / total, correct / total, true_labels, predicted_labels


def count_parameters(model):
    """Count trainable weights to compare model capacity."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def prepare_output_folders():
    """Create the small folders used by every exercise."""
    for folder in [REPORT_FOLDER, FIGURE_FOLDER, CHECKPOINT_FOLDER]:
        folder.mkdir(parents=True, exist_ok=True)


def save_json(path, content):
    """Save plain Python results for the report and Pages index."""
    Path(path).write_text(json.dumps(content, indent=2), encoding="utf-8")
