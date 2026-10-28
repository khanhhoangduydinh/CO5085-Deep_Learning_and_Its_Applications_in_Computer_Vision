"""Small contracts for the beginner CO5085 exercise scripts."""

from pathlib import Path

import torch

from common import make_split_indices
from E1.run_e1 import CNNClassifier, MLPClassifier, SoftmaxClassifier
from E2.run_e2 import CNNStemTokenizer, ManualMSA, PatchTokenizer
from E3.run_e3 import RowSequenceClassifier
from A1.run_a1 import EXPERIMENTS, LOADER_WORKERS, build_model, make_stratified_indices, select_best_experiment


def test_fixed_split_uses_train_images_only_for_training_and_validation():
    """Changing the seed or split size would break fair E1-E3 comparison."""
    train_ids, validation_ids = make_split_indices(total=60_000, seed=42)

    assert len(train_ids) == 48_000
    assert len(validation_ids) == 12_000
    assert set(train_ids).isdisjoint(validation_ids)
    assert set(train_ids) | set(validation_ids) == set(range(60_000))


def test_e1_models_return_ten_class_scores_for_one_image_batch():
    """A missing flatten step or wrong final layer would change this shape."""
    images = torch.zeros(4, 1, 28, 28)

    for model in [SoftmaxClassifier(), MLPClassifier(), CNNClassifier()]:
        assert model(images).shape == (4, 10)


def test_manual_attention_preserves_token_shape():
    """A wrong head reshape would not return one vector per input token."""
    tokens = torch.zeros(2, 49, 32)
    attention = ManualMSA(embed_dim=32, heads=2)

    assert attention(tokens).shape == tokens.shape


def test_two_tokenizers_create_tokens_for_one_fashion_image_batch():
    """Patch and CNN-stem tokenizers must both produce usable token sequences."""
    images = torch.zeros(2, 1, 28, 28)

    patch_tokens = PatchTokenizer(embed_dim=32)(images)
    stem_tokens = CNNStemTokenizer(embed_dim=32)(images)

    assert patch_tokens.shape == (2, 49, 32)
    assert stem_tokens.shape[0] == 2
    assert stem_tokens.shape[2] == 32
    assert stem_tokens.shape[1] > 1


def test_lstm_and_gru_return_ten_class_scores():
    """Changing the sequence shape or classifier head would break E3 outputs."""
    images = torch.zeros(3, 1, 28, 28)

    assert RowSequenceClassifier(kind="lstm")(images).shape == (3, 10)
    assert RowSequenceClassifier(kind="gru")(images).shape == (3, 10)


def test_pages_links_point_to_existing_submission_files():
    """The static page must point to the report and all three exercise files."""
    root = Path(__file__).parents[1]
    page = (root / "index.html").read_text(encoding="utf-8")

    for path in [
        "E1/run_e1.py",
        "E2/run_e2.py",
        "E3/run_e3.py",
        "CO5085%20Report%20Latex/report.pdf",
    ]:
        assert path in page

    assert (root / "CO5085 Report Latex" / "report.pdf").is_file()


def test_public_links_use_the_full_course_repository_name():
    """The site and README must name the current course, not a shortened title."""
    root = Path(__file__).parents[1]
    repository = "CO5085-Deep_Learning_and_Its_Applications_in_Computer_Vision"

    assert repository in (root / "index.html").read_text(encoding="utf-8")
    assert repository in (root / "README.md").read_text(encoding="utf-8")


def test_report_maps_formulas_and_algorithms_to_all_three_exercises():
    """The report must explain the theory and the code workflow for E1-E3."""
    root = Path(__file__).parents[1]
    report = (root / "CO5085 Report Latex" / "report.tex").read_text(encoding="utf-8")

    for heading in [
        "Shared Learning Workflow and Notation",
        "Softmax, Cross-Entropy, and Gradient Update",
        "Algorithm 1: Shared Training and Evaluation Loop",
        "Scaled Dot-Product Attention",
        "Algorithm 2: Manual Multi-Head Self-Attention",
        "LSTM and GRU Gates",
        "Algorithm 3: Row-Sequence Classification",
    ]:
        assert heading in report


def test_shared_theory_is_not_numbered_as_a_zero_section():
    """An unnumbered introduction should not create an awkward 0.1 heading."""
    root = Path(__file__).parents[1]
    report = (root / "CO5085 Report Latex" / "report.tex").read_text(encoding="utf-8")

    assert r"\subsection*{Shared Learning Workflow and Notation}" in report


def test_a1_stratified_split_keeps_400_train_and_100_validation_images_per_class():
    """A1 must use every CIFAR-100 training image without mixing in the test set."""
    labels = [class_id for class_id in range(100) for _ in range(500)]
    train_ids, validation_ids = make_stratified_indices(labels)

    assert len(train_ids) == 40_000
    assert len(validation_ids) == 10_000
    assert set(train_ids).isdisjoint(validation_ids)
    assert {labels[index] for index in train_ids} == set(range(100))
    assert all(sum(labels[index] == class_id for index in train_ids) == 400 for class_id in range(100))
    assert all(sum(labels[index] == class_id for index in validation_ids) == 100 for class_id in range(100))


def test_a1_models_return_100_scores_for_cifar100_images():
    """Changing a classifier head must not silently leave its ImageNet output size."""
    images = torch.zeros(2, 3, 96, 96)

    for family in ["cnn", "transformer"]:
        model = build_model(family, "scratch", classes=100)
        assert model(images).shape == (2, 100)


def test_a1_pretrained_freeze_modes_train_the_expected_amount_of_the_model():
    """Head, partial, and full must be distinct transfer-learning modes."""
    counts = {}
    for family in ["cnn", "transformer"]:
        for mode in ["head", "partial", "full"]:
            model = build_model(family, mode, classes=100)
            counts[family, mode] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

        assert counts[family, "head"] < counts[family, "partial"] < counts[family, "full"]


def test_a1_experiment_matrix_has_both_families_and_every_required_mode():
    """The declared runner matrix implements the full A1 comparison."""
    assert {(item["family"], item["mode"]) for item in EXPERIMENTS} == {
        (family, mode)
        for family in ["cnn", "transformer"]
        for mode in ["scratch", "head", "partial", "full"]
    }


def test_a1_selects_the_validation_winner_without_using_test_accuracy():
    """The official test set must not choose a configuration for A1."""
    records = [
        {"family": "cnn", "mode": "head", "validation_accuracy": 0.60, "test_accuracy": 0.99},
        {"family": "transformer", "mode": "full", "validation_accuracy": 0.70, "test_accuracy": 0.01},
    ]

    winner = select_best_experiment(records)

    assert winner["family"] == "transformer"
    assert winner["mode"] == "full"


def test_a1_loader_uses_parallel_workers_for_cpu_image_preprocessing():
    """Resizing and augmentation should not leave the available GPU idle."""
    assert LOADER_WORKERS == 4


def test_a1_submission_links_and_report_source_are_published():
    """The A1 landing-page entry must lead to code, instructions, and a report."""
    root = Path(__file__).parents[1]
    page = (root / "index.html").read_text(encoding="utf-8")
    report = root / "CO5085 Report Latex" / "a1_report.tex"

    for path in [
        "A1/run_a1.py",
        "A1/README.md",
        "CO5085%20Report%20Latex/a1_report.pdf",
    ]:
        assert path in page

    assert (root / "A1" / "README.md").is_file()
    assert report.is_file()
    assert (root / "CO5085 Report Latex" / "a1_report.pdf").is_file()
    report_text = report.read_text(encoding="utf-8")
    for heading in [
        "CIFAR-100 Dataset and Protocol",
        "ResNet-18 and Swin-T Theory",
        "Transfer Learning and Freeze Modes",
        "Experiment Algorithm",
        "Measured Results and Discussion",
    ]:
        assert heading in report_text
