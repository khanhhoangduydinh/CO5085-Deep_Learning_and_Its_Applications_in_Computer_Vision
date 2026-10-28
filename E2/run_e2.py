"""E2: manual attention, PyTorch attention, and two image tokenizations."""

import math
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
    count_parameters,
    evaluate,
    get_loaders,
    prepare_output_folders,
    save_json,
    set_seed,
    train_epoch,
)


EPOCHS = 2


class PatchTokenizer(nn.Module):
    """Split a 28 by 28 image into 49 non-overlapping 4 by 4 patches."""

    def __init__(self, embed_dim=32):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=4, stride=4)
        self.project = nn.Linear(16, embed_dim)

    def forward(self, images):
        patches = self.unfold(images).transpose(1, 2)
        return self.project(patches)


class CNNStemTokenizer(nn.Module):
    """Use a small CNN feature map; each spatial position becomes one token."""

    def __init__(self, embed_dim=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, images):
        features = self.stem(images)
        return features.flatten(start_dim=2).transpose(1, 2)


class ManualMSA(nn.Module):
    """Multi-head self-attention built from Linear layers and matrix products."""

    def __init__(self, embed_dim=32, heads=2):
        super().__init__()
        if embed_dim % heads != 0:
            raise ValueError("embed_dim must divide equally across heads")
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)

    def split_heads(self, values):
        batch, tokens, _ = values.shape
        return values.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, tokens):
        query = self.split_heads(self.query(tokens))
        key = self.split_heads(self.key(tokens))
        value = self.split_heads(self.value(tokens))
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        attended = weights @ value
        batch, _, token_count, _ = attended.shape
        attended = attended.transpose(1, 2).contiguous().view(batch, token_count, -1)
        return self.output(attended)


class AttentionClassifier(nn.Module):
    """A shallow attention classifier with either manual or PyTorch MSA."""

    def __init__(self, attention_kind="manual", token_kind="patch", embed_dim=32, heads=2):
        super().__init__()
        self.tokenizer = PatchTokenizer(embed_dim) if token_kind == "patch" else CNNStemTokenizer(embed_dim)
        token_count = 49
        self.position = nn.Parameter(torch.zeros(1, token_count, embed_dim))
        self.attention_kind = attention_kind
        self.attention = ManualMSA(embed_dim, heads) if attention_kind == "manual" else nn.MultiheadAttention(
            embed_dim, heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, embed_dim))
        self.norm2 = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, 10)

    def forward(self, images):
        tokens = self.tokenizer(images)
        tokens = tokens + self.position[:, : tokens.shape[1]]
        if self.attention_kind == "manual":
            attended = self.attention(tokens)
        else:
            attended, _ = self.attention(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attended)
        tokens = self.norm2(tokens + self.feed_forward(tokens))
        return self.classifier(tokens.mean(dim=1))


def train_model(name, model, train_loader, validation_loader, test_loader):
    """Train one E2 classifier with the same explicit loop used in E1."""
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

    _, test_accuracy, true_labels, predicted_labels = evaluate(model, test_loader, loss_function)
    return history, {
        "parameters": count_parameters(model),
        "seconds": time.perf_counter() - start,
        "validation_accuracy": history["validation_accuracy"][-1],
        "test_accuracy": test_accuracy,
        "true_labels": true_labels,
        "predicted_labels": predicted_labels,
    }


def plot_tokenization(test_loader):
    """Show how 4 by 4 patches divide one Fashion-MNIST image."""
    images, labels = next(iter(test_loader))
    image = images[0].squeeze().numpy()
    figure, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(image, cmap="gray")
    for position in range(0, 29, 4):
        axis.axhline(position - 0.5, color="#dc5b0b", linewidth=0.8)
        axis.axvline(position - 0.5, color="#dc5b0b", linewidth=0.8)
    axis.set(title=f"Patch tokens: {CLASS_NAMES[labels[0]]}")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(FIGURE_FOLDER / "e2_01_patch_tokens.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_history(histories):
    """Plot the two measurements recorded after every E2 epoch."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, history in histories.items():
        epochs = range(1, EPOCHS + 1)
        axes[0].plot(epochs, history["train_loss"], marker="o", label=name)
        axes[1].plot(epochs, history["validation_accuracy"], marker="o", label=name)
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURE_FOLDER / "e2_02_learning_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_comparison(results):
    """Show accuracy, capacity, and measured training time together."""
    names = list(results)
    values = [
        ("Test accuracy", [results[name]["test_accuracy"] for name in names]),
        ("Parameters", [results[name]["parameters"] for name in names]),
        ("Seconds", [results[name]["seconds"] for name in names]),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, (title, numbers) in zip(axes, values):
        axis.bar(names, numbers, color=["#e9b949", "#dc5b0b", "#174653"])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=18, labelsize=8)
    axes[0].set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(FIGURE_FOLDER / "e2_03_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_selected_confusion(selected_name, selected_result):
    """Use the validation-selected E2 model for one final error view."""
    matrix = confusion_matrix(selected_result["true_labels"], selected_result["predicted_labels"], normalize="true")
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Purples", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axis)
    axis.set(title=f"{selected_name} test confusion matrix", xlabel="Predicted", ylabel="True")
    figure.tight_layout()
    figure.savefig(FIGURE_FOLDER / "e2_04_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    set_seed()
    prepare_output_folders()
    train_loader, validation_loader, test_loader = get_loaders()
    plot_tokenization(test_loader)

    models = {
        "Manual MSA + patch": AttentionClassifier("manual", "patch"),
        "PyTorch MSA + patch": AttentionClassifier("pytorch", "patch"),
        "PyTorch MSA + CNN stem": AttentionClassifier("pytorch", "cnn"),
    }
    histories = {}
    results = {}
    for name, model in models.items():
        set_seed()
        histories[name], results[name] = train_model(name, model, train_loader, validation_loader, test_loader)

    selected_name = max(results, key=lambda name: results[name]["validation_accuracy"])
    plot_history(histories)
    plot_comparison(results)
    plot_selected_confusion(selected_name, results[selected_name])
    for result in results.values():
        result.pop("true_labels")
        result.pop("predicted_labels")
    save_json(
        "reports/e2_results.json",
        {
            "dataset": "Fashion-MNIST",
            "tokenizations": {"patch": "49 tokens of 4x4 pixels", "cnn_stem": "49 feature-map tokens from a small CNN stem"},
            "epochs": EPOCHS,
            "models": results,
            "selected_model": selected_name,
        },
    )


if __name__ == "__main__":
    main()
