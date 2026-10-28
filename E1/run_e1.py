"""E1: compare a linear softmax model, an MLP, and a small CNN."""

import time

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch import nn

from common import (
    CLASS_NAMES,
    DEVICE,
    FIGURE_FOLDER,
    CHECKPOINT_FOLDER,
    count_parameters,
    evaluate,
    get_loaders,
    prepare_output_folders,
    save_json,
    set_seed,
    train_epoch,
)


EPOCHS = 3


class SoftmaxClassifier(nn.Module):
    """A single linear layer after flattening 28 by 28 pixels."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))

    def forward(self, images):
        return self.layers(images)


class MLPClassifier(nn.Module):
    """A small fully connected network using flattened pixels."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, images):
        return self.layers(images)


class CNNClassifier(nn.Module):
    """Two convolution blocks followed by a ten-class classifier."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, images):
        return self.classifier(self.features(images))


def train_model(name, model, train_loader, validation_loader, test_loader):
    """Train one model and return its history and final measurements."""
    model = model.to(DEVICE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = {"train_loss": [], "train_accuracy": [], "validation_loss": [], "validation_accuracy": []}
    start = time.perf_counter()

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = train_epoch(model, train_loader, loss_function, optimizer)
        validation_loss, validation_accuracy, _, _ = evaluate(model, validation_loader, loss_function)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)
        print(f"{name} | epoch {epoch + 1}: validation accuracy = {validation_accuracy:.3f}")

    _, test_accuracy, true_labels, predicted_labels = evaluate(model, test_loader, loss_function)
    return model, history, {
        "parameters": count_parameters(model),
        "seconds": time.perf_counter() - start,
        "validation_accuracy": history["validation_accuracy"][-1],
        "test_accuracy": test_accuracy,
        "true_labels": true_labels,
        "predicted_labels": predicted_labels,
    }


def plot_examples(test_loader):
    """Show six simple Fashion-MNIST examples before training."""
    images, labels = next(iter(test_loader))
    figure, axes = plt.subplots(1, 6, figsize=(12, 2.5))
    for axis, image, label in zip(axes, images[:6], labels[:6]):
        axis.imshow(image.squeeze(), cmap="gray")
        axis.set_title(CLASS_NAMES[label])
        axis.axis("off")
    figure.suptitle("Fashion-MNIST examples")
    figure.tight_layout()
    figure.savefig(FIGURE_FOLDER / "e1_01_examples.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_learning_curves(histories):
    """Plot loss and validation accuracy for all three E1 models."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, history in histories.items():
        epochs = range(1, EPOCHS + 1)
        axes[0].plot(epochs, history["train_loss"], marker="o", label=name)
        axes[1].plot(epochs, history["validation_accuracy"], marker="o", label=name)
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[1].set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(FIGURE_FOLDER / "e1_02_learning_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_comparison(results):
    """Compare capacity and test accuracy without hiding the parameter count."""
    names = list(results)
    accuracies = [results[name]["test_accuracy"] for name in names]
    parameters = [results[name]["parameters"] for name in names]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, accuracies, color=["#e9b949", "#dc5b0b", "#174653"])
    axes[0].set(title="Test accuracy", ylabel="Accuracy", ylim=(0, 1))
    axes[1].bar(names, parameters, color=["#e9b949", "#dc5b0b", "#174653"])
    axes[1].set(title="Trainable parameters", ylabel="Parameters")
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(FIGURE_FOLDER / "e1_03_model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_cnn_errors(test_loader, result):
    """Inspect the CNN confusion matrix and six incorrect test images."""
    true_labels = result["true_labels"]
    predicted_labels = result["predicted_labels"]
    matrix = confusion_matrix(true_labels, predicted_labels, normalize="true")
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axis)
    axis.set(title="CNN normalized test confusion matrix", xlabel="Predicted", ylabel="True")
    figure.tight_layout()
    figure.savefig(FIGURE_FOLDER / "e1_04_cnn_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    images = []
    for batch_images, _ in test_loader:
        images.extend(batch_images)
    wrong_ids = [index for index, (true, predicted) in enumerate(zip(true_labels, predicted_labels)) if true != predicted][:6]
    figure, axes = plt.subplots(1, 6, figsize=(12, 2.5))
    for axis, index in zip(axes, wrong_ids):
        axis.imshow(images[index].squeeze(), cmap="gray")
        axis.set_title(f"true: {CLASS_NAMES[true_labels[index]]}\npred: {CLASS_NAMES[predicted_labels[index]]}", fontsize=8)
        axis.axis("off")
    figure.suptitle("CNN incorrect test predictions")
    figure.tight_layout()
    figure.savefig(FIGURE_FOLDER / "e1_05_cnn_errors.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    set_seed()
    prepare_output_folders()
    train_loader, validation_loader, test_loader = get_loaders()
    plot_examples(test_loader)

    models = {
        "Softmax linear": SoftmaxClassifier(),
        "MLP": MLPClassifier(),
        "CNN": CNNClassifier(),
    }
    histories = {}
    results = {}
    trained_models = {}

    for name, model in models.items():
        set_seed()
        trained_models[name], histories[name], results[name] = train_model(
            name, model, train_loader, validation_loader, test_loader
        )

    plot_learning_curves(histories)
    plot_comparison(results)
    plot_cnn_errors(test_loader, results["CNN"])
    torch.save(trained_models["CNN"].state_dict(), CHECKPOINT_FOLDER / "e1_cnn.pt")

    for result in results.values():
        result.pop("true_labels")
        result.pop("predicted_labels")
    save_json(
        "reports/e1_results.json",
        {
            "dataset": "Fashion-MNIST",
            "split": {"train": 48_000, "validation": 12_000, "test": 10_000, "seed": 42},
            "epochs": EPOCHS,
            "models": results,
            "selected_baseline": "CNN",
        },
    )


if __name__ == "__main__":
    main()
