"""A beginner-friendly A1 comparison of ResNet-18 and Swin-T on CIFAR-100.

Run from the repository root with: ``python -m A1.run_a1``.
The script downloads CIFAR-100 through Hugging Face only on the first run.
"""

import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, Swin_T_Weights, resnet18, swin_t


SEED = 42
IMAGE_SIZE = 96
CLASS_COUNT = 100
EPOCHS = 3
BATCH_SIZE = 64
LOADER_WORKERS = 4
DATASET_NAME = "uoft-cs/cifar100"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPORT_FOLDER = Path("reports")
FIGURE_FOLDER = REPORT_FOLDER / "figures"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
EXPERIMENTS = [
    {"family": family, "mode": mode}
    for family in ["cnn", "transformer"]
    for mode in ["scratch", "head", "partial", "full"]
]


class Cifar100Dataset(Dataset):
    """Turn one Hugging Face split into the simple PyTorch Dataset interface."""

    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        image = item["img"].convert("RGB")
        return self.transform(image), item["fine_label"]


def make_stratified_indices(labels, train_per_class=400, seed=SEED):
    """Keep the same number of examples from every CIFAR-100 fine class."""
    groups = [[] for _ in range(CLASS_COUNT)]
    for index, label in enumerate(labels):
        groups[label].append(index)

    generator = random.Random(seed)
    train_ids = []
    validation_ids = []
    for group in groups:
        generator.shuffle(group)
        train_ids.extend(group[:train_per_class])
        validation_ids.extend(group[train_per_class:])
    return train_ids, validation_ids


def set_trainable_parameters(model, family, mode):
    """Apply one of the three pretrained freeze modes required by A1."""
    for parameter in model.parameters():
        parameter.requires_grad = mode in ["scratch", "full"]

    if mode == "head":
        modules = [model.fc] if family == "cnn" else [model.head]
    elif mode == "partial":
        modules = [model.layer4, model.fc] if family == "cnn" else [model.features[-1], model.head]
    else:
        modules = []

    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True


def build_model(family, mode, classes=CLASS_COUNT):
    """Load one torchvision backbone and replace its ImageNet classifier head."""
    if family == "cnn":
        weights = None if mode == "scratch" else ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, classes)
    elif family == "transformer":
        weights = None if mode == "scratch" else Swin_T_Weights.IMAGENET1K_V1
        model = swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, classes)
    else:
        raise ValueError("family must be 'cnn' or 'transformer'")

    set_trainable_parameters(model, family, mode)
    return model


def set_seed(seed=SEED):
    """Make the split, shuffle order, and random augmentation repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_loaders():
    """Load CIFAR-100 and return fixed train, validation, and test loaders."""
    dataset = load_dataset(DATASET_NAME)
    train_data = dataset["train"]
    test_data = dataset["test"]
    train_ids, validation_ids = make_stratified_indices(train_data["fine_label"])

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomCrop(IMAGE_SIZE, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    evaluation_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_set = Cifar100Dataset(train_data, train_transform)
    validation_set = Cifar100Dataset(train_data, evaluation_transform)
    test_set = Cifar100Dataset(test_data, evaluation_transform)
    options = {
        "batch_size": BATCH_SIZE,
        "num_workers": LOADER_WORKERS,
        "persistent_workers": True,
        "pin_memory": DEVICE.type == "cuda",
    }
    train_loader = DataLoader(Subset(train_set, train_ids), shuffle=True, **options)
    validation_loader = DataLoader(Subset(validation_set, validation_ids), shuffle=False, **options)
    test_loader = DataLoader(test_set, shuffle=False, **options)
    class_names = train_data.features["fine_label"].names
    return train_loader, validation_loader, test_loader, train_data, test_data, class_names, train_transform


def keep_frozen_batch_norm_in_evaluation_mode(model):
    """Do not update ResNet running statistics inside a frozen backbone."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if not any(parameter.requires_grad for parameter in module.parameters()):
                module.eval()


def train_epoch(model, loader, loss_function, optimizer):
    """One clear training pass: forward, loss, backward, and optimizer step."""
    model.train()
    keep_frozen_batch_norm_in_evaluation_mode(model)
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
    """Measure loss and accuracy without changing parameters."""
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
            predictions = scores.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += len(labels)
            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(predictions.cpu().tolist())
    return total_loss / total, correct / total, true_labels, predicted_labels


def learning_rate_for(mode):
    """Use a larger rate for a new head and a smaller rate for all-layer tuning."""
    return {"scratch": 3e-4, "head": 1e-3, "partial": 3e-4, "full": 1e-4}[mode]


def train_one_experiment(family, mode, train_loader, validation_loader):
    """Train one configuration and keep its best validation checkpoint in RAM."""
    set_seed()
    model = build_model(family, mode).to(DEVICE)
    loss_function = nn.CrossEntropyLoss()
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate_for(mode),
        weight_decay=1e-4,
    )
    history = []
    best_accuracy = -1.0
    best_state = None
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_epoch(model, train_loader, loss_function, optimizer)
        validation_loss, validation_accuracy, _, _ = evaluate(model, validation_loader, loss_function)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        })
        print(f"{family:11} {mode:7} epoch {epoch}/{EPOCHS}: validation accuracy = {validation_accuracy:.4f}")
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    elapsed_seconds = time.perf_counter() - start
    peak_memory_mb = 0.0 if DEVICE.type == "cpu" else torch.cuda.max_memory_allocated() / 1024**2
    record = {
        "family": family,
        "mode": mode,
        "epochs": EPOCHS,
        "learning_rate": learning_rate_for(mode),
        "trainable_parameters": trainable_parameters,
        "validation_accuracy": best_accuracy,
        "time_seconds": elapsed_seconds,
        "peak_memory_mb": peak_memory_mb,
        "history": history,
    }
    model.cpu()
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return record, best_state


def select_best_experiment(records):
    """Select with validation accuracy only; the test result is never inspected here."""
    return max(records, key=lambda record: record["validation_accuracy"])


def to_display_image(tensor):
    """Undo normalization so a saved figure shows ordinary RGB colours."""
    image = tensor.permute(1, 2, 0).cpu().numpy()
    return np.clip(image * np.array(STD) + np.array(MEAN), 0, 1)


def plot_augmentation(train_data, train_transform, class_names):
    """Show original CIFAR-100 images and their random training versions."""
    figure, axes = plt.subplots(2, 5, figsize=(12, 5))
    for column in range(5):
        item = train_data[column]
        axes[0, column].imshow(item["img"])
        axes[0, column].set_title(class_names[item["fine_label"]], fontsize=9)
        axes[1, column].imshow(to_display_image(train_transform(item["img"].convert("RGB"))))
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    axes[0, 0].set_ylabel("original", fontsize=11)
    axes[1, 0].set_ylabel("training\naugmentation", fontsize=11)
    figure.suptitle("CIFAR-100 examples and training augmentation", fontsize=14)
    figure.tight_layout()
    figure.savefig(FIGURE_FOLDER / "a1_01_augmentation.png", dpi=180)
    plt.close(figure)


def plot_training_curves(records):
    """Draw one validation-accuracy graph per model family."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, family in zip(axes, ["cnn", "transformer"]):
        for record in records:
            if record["family"] == family:
                values = [entry["validation_accuracy"] for entry in record["history"]]
                axis.plot(range(1, EPOCHS + 1), values, marker="o", label=record["mode"])
        axis.set_title("ResNet-18" if family == "cnn" else "Swin-T")
        axis.set_xlabel("epoch")
        axis.set_ylabel("validation accuracy")
        axis.set_xticks(range(1, EPOCHS + 1))
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle("Validation learning curves", fontsize=14)
    figure.tight_layout()
    figure.savefig(FIGURE_FOLDER / "a1_02_learning_curves.png", dpi=180)
    plt.close(figure)


def plot_comparison(records):
    """Compare validation accuracy, trainable weights, time, and peak memory."""
    labels = [f"{item['family']}\n{item['mode']}" for item in records]
    colours = ["#e66a13" if item["family"] == "cnn" else "#124e66" for item in records]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    values = [
        ("validation accuracy", [item["validation_accuracy"] for item in records]),
        ("trainable parameters (millions)", [item["trainable_parameters"] / 1e6 for item in records]),
        ("training time (seconds)", [item["time_seconds"] for item in records]),
        ("peak GPU memory (MB)", [item["peak_memory_mb"] for item in records]),
    ]
    for axis, (title, numbers) in zip(axes.flat, values):
        axis.bar(labels, numbers, color=colours)
        axis.set_title(title)
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("A1 comparison: orange = ResNet-18, blue = Swin-T", fontsize=14)
    figure.tight_layout()
    figure.savefig(FIGURE_FOLDER / "a1_03_comparison.png", dpi=180)
    plt.close(figure)


def format_confusion_label(value):
    """Show useful row percentages and leave very small cells empty."""
    return f"{value:.1%}" if value >= 0.01 else ""


def plot_coarse_confusion(true_labels, predicted_labels, test_data):
    """Show a readable 20-superclass matrix for the selected 100-class model."""
    fine_to_coarse = {}
    for fine_label, coarse_label in zip(test_data["fine_label"], test_data["coarse_label"]):
        fine_to_coarse[fine_label] = coarse_label
    true_coarse = [fine_to_coarse[label] for label in true_labels]
    predicted_coarse = [fine_to_coarse[label] for label in predicted_labels]
    matrix = confusion_matrix(true_coarse, predicted_coarse, labels=range(20), normalize="true")
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    axis.set_title("Selected model: normalized confusion matrix of 20 CIFAR-100 superclasses")
    axis.set_xlabel("predicted superclass ID")
    axis.set_ylabel("true superclass ID")
    axis.set_xticks(range(20))
    axis.set_yticks(range(20))
    for row, column in np.ndindex(matrix.shape):
        value = matrix[row, column]
        label = format_confusion_label(value)
        if label:
            colour = "white" if value >= 0.50 else "#102a43"
            axis.text(column, row, label, ha="center", va="center", fontsize=5.5, color=colour)
    figure.colorbar(image, ax=axis, label="row-normalized proportion")
    figure.tight_layout()
    figure.savefig(FIGURE_FOLDER / "a1_04_coarse_confusion_matrix.png", dpi=180)
    plt.close(figure)


def plot_error_examples(true_labels, predicted_labels, test_data, class_names):
    """Display a small set of real wrong predictions from the one final test run."""
    wrong_ids = [index for index, pair in enumerate(zip(true_labels, predicted_labels)) if pair[0] != pair[1]][:12]
    figure, axes = plt.subplots(3, 4, figsize=(10, 7))
    for axis, index in zip(axes.flat, wrong_ids):
        item = test_data[index]
        axis.imshow(item["img"])
        axis.set_title(f"true: {class_names[true_labels[index]]}\npred: {class_names[predicted_labels[index]]}", fontsize=8)
        axis.axis("off")
    for axis in axes.flat[len(wrong_ids):]:
        axis.axis("off")
    figure.suptitle("Examples misclassified by the selected model", fontsize=14)
    figure.tight_layout()
    figure.savefig(FIGURE_FOLDER / "a1_05_error_examples.png", dpi=180)
    plt.close(figure)


def save_results(records, winner, test_accuracy, test_f1):
    """Store only plain JSON values so the website and report are reproducible."""
    result = {
        "dataset": {
            "name": "CIFAR-100",
            "source": DATASET_NAME,
            "classes": CLASS_COUNT,
            "official_train_images": 50_000,
            "official_test_images": 10_000,
            "train_images": 40_000,
            "validation_images": 10_000,
            "image_size_after_resize": IMAGE_SIZE,
        },
        "protocol": {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "device": str(DEVICE),
            "test_used_only_for_selected_validation_winner": True,
        },
        "experiments": records,
        "selected_model": {
            "family": winner["family"],
            "mode": winner["mode"],
            "validation_accuracy": winner["validation_accuracy"],
            "test_accuracy": test_accuracy,
            "test_macro_f1": test_f1,
        },
    }
    (REPORT_FOLDER / "a1_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def main():
    """Run the full A1 matrix, then evaluate the validation winner exactly once."""
    set_seed()
    REPORT_FOLDER.mkdir(exist_ok=True)
    FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)
    loaders = get_loaders()
    train_loader, validation_loader, test_loader, train_data, test_data, class_names, train_transform = loaders
    plot_augmentation(train_data, train_transform, class_names)

    records = []
    states = {}
    for experiment in EXPERIMENTS:
        record, state = train_one_experiment(
            experiment["family"], experiment["mode"], train_loader, validation_loader
        )
        records.append(record)
        states[record["family"], record["mode"]] = state

    winner = select_best_experiment(records)
    model = build_model(winner["family"], winner["mode"]).to(DEVICE)
    model.load_state_dict(states[winner["family"], winner["mode"]])
    _, test_accuracy, true_labels, predicted_labels = evaluate(model, test_loader, nn.CrossEntropyLoss())
    test_f1 = f1_score(true_labels, predicted_labels, average="macro")
    print(f"selected model: {winner['family']} / {winner['mode']}")
    print(f"official test accuracy: {test_accuracy:.4f}; macro F1: {test_f1:.4f}")
    plot_training_curves(records)
    plot_comparison(records)
    plot_coarse_confusion(true_labels, predicted_labels, test_data)
    plot_error_examples(true_labels, predicted_labels, test_data, class_names)
    save_results(records, winner, test_accuracy, test_f1)


if __name__ == "__main__":
    main()
