"""E3: read Fashion-MNIST rows as sequences with LSTM and GRU."""

import os
import json
import time
from pathlib import Path

# Avoid a Windows OpenMP conflict between installed scientific packages.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import torch
from torch import nn

from common import (
    CLASS_NAMES,
    DEVICE,
    FIGURE_FOLDER,
    count_parameters,
    evaluate,
    get_loaders,
    prepare_output_folders,
    save_json,
    set_seed,
    train_epoch,
)


EPOCHS = 3


class RowSequenceClassifier(nn.Module):
    """Treat each image row of 28 pixels as one time step."""

    def __init__(self, kind="lstm", hidden_size=64):
        super().__init__()
        if kind not in {"lstm", "gru"}:
            raise ValueError("kind must be 'lstm' or 'gru'")
        recurrent_layer = nn.LSTM if kind == "lstm" else nn.GRU
        self.recurrent = recurrent_layer(input_size=28, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 10)

    def forward(self, images):
        rows = images.squeeze(1)
        output, _ = self.recurrent(rows)
        return self.classifier(output[:, -1])


def train_model(name, model, train_loader, validation_loader, test_loader):
    """Train one recurrent classifier with the required manual PyTorch loop."""
    model = model.to(DEVICE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = {"train_loss": [], "validation_accuracy": []}
    start = time.perf_counter()

    for epoch in range(EPOCHS):
        train_loss, _ = train_epoch(model, train_loader, loss_function, optimizer)
        _, validation_accuracy, _, _ = evaluate(model, validation_loader, loss_function)
        history["train_loss"].append(train_loss)
        history["validation_accuracy"].append(validation_accuracy)
        print(f"{name} | epoch {epoch + 1}: validation accuracy = {validation_accuracy:.3f}")

    _, test_accuracy, _, _ = evaluate(model, test_loader, loss_function)
    return history, {
        "parameters": count_parameters(model),
        "seconds": time.perf_counter() - start,
        "validation_accuracy": history["validation_accuracy"][-1],
        "test_accuracy": test_accuracy,
    }


def plot_row_sequence(test_loader):
    """Show that one image is read from the top row to the bottom row."""
    images, labels = next(iter(test_loader))
    image = images[0].squeeze().numpy()
    figure, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(image, cmap="gray")
    for row in range(0, 28, 4):
        axis.axhline(row - 0.5, color="#dc5b0b", linewidth=0.8)
    axis.set(title=f"28 row steps: {CLASS_NAMES[labels[0]]}")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(FIGURE_FOLDER / "e3_01_row_sequence.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_history(histories):
    """Plot the LSTM and GRU learning curves separately from the CNN baseline."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, history in histories.items():
        epochs = range(1, EPOCHS + 1)
        axes[0].plot(epochs, history["train_loss"], marker="o", label=name)
        axes[1].plot(epochs, history["validation_accuracy"], marker="o", label=name)
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(FIGURE_FOLDER / "e3_02_learning_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_comparison(results, cnn_baseline):
    """Compare the two recurrent models with the CNN measurement from E1."""
    names = ["CNN baseline", "LSTM rows", "GRU rows"]
    accuracies = [cnn_baseline["test_accuracy"], results["LSTM rows"]["test_accuracy"], results["GRU rows"]["test_accuracy"]]
    parameters = [cnn_baseline["parameters"], results["LSTM rows"]["parameters"], results["GRU rows"]["parameters"]]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, accuracies, color=["#174653", "#e9b949", "#dc5b0b"])
    axes[0].set(title="Test accuracy", ylabel="Accuracy", ylim=(0, 1))
    axes[1].bar(names, parameters, color=["#174653", "#e9b949", "#dc5b0b"])
    axes[1].set(title="Trainable parameters", ylabel="Parameters")
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(FIGURE_FOLDER / "e3_03_rnn_vs_cnn.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def load_cnn_baseline():
    """Read the E1 CNN result instead of training a fourth model for E3."""
    path = Path("reports/e1_results.json")
    if not path.exists():
        raise FileNotFoundError("Run E1/run_e1.py before E3 so the CNN baseline is available.")
    return json.loads(path.read_text(encoding="utf-8"))["models"]["CNN"]


def main():
    set_seed()
    prepare_output_folders()
    train_loader, validation_loader, test_loader = get_loaders()
    cnn_baseline = load_cnn_baseline()
    plot_row_sequence(test_loader)

    models = {"LSTM rows": RowSequenceClassifier("lstm"), "GRU rows": RowSequenceClassifier("gru")}
    histories = {}
    results = {}
    for name, model in models.items():
        set_seed()
        histories[name], results[name] = train_model(name, model, train_loader, validation_loader, test_loader)

    plot_history(histories)
    plot_comparison(results, cnn_baseline)
    selected_name = max(results, key=lambda name: results[name]["validation_accuracy"])
    save_json(
        "reports/e3_results.json",
        {
            "dataset": "Fashion-MNIST",
            "sequence": "Each 28-pixel image row is one time step; sequence length is 28.",
            "epochs": EPOCHS,
            "models": results,
            "cnn_baseline": cnn_baseline,
            "selected_model": selected_name,
        },
    )


if __name__ == "__main__":
    main()
