"""Small contracts for the beginner CO5085 exercise scripts."""

from pathlib import Path

import torch

from common import make_split_indices
from E1.run_e1 import CNNClassifier, MLPClassifier, SoftmaxClassifier
from E2.run_e2 import CNNStemTokenizer, ManualMSA, PatchTokenizer
from E3.run_e3 import RowSequenceClassifier


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
