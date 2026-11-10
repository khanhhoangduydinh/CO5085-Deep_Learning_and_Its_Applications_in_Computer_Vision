"""Draw report diagrams from the implemented CO5085 model configurations.

Run from any directory with Python and matplotlib. No training or downloads.
PDFs keep text and arrows as vectors; PNGs are convenient GitHub previews.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT = Path(__file__).resolve().parent
INK, BLUE, PALE, GREEN = "#102a43", "#1954a6", "#eaf2fb", "#e8f3ed"
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})


def canvas(title, subtitle, height):
    fig, ax = plt.subplots(figsize=(12, height))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 12), ylim=(0, height))
    ax.axis("off")
    ax.text(0.2, height - 0.35, title, fontsize=21, weight="bold", color=INK)
    ax.text(0.2, height - 0.75, subtitle, fontsize=13, color=INK)
    return fig, ax


def box(ax, x, y, text, width=2.55, height=1.0, fill=PALE, size=14):
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.06,rounding_size=0.09",
        edgecolor=BLUE, facecolor=fill, linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=size,
            color=INK, linespacing=1.4)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                mutation_scale=14, color=BLUE, linewidth=1.3))


def row(ax, y, labels, fill=PALE, height=1.0):
    positions = [1.5, 4.5, 7.5, 10.5]
    for x, label in zip(positions, labels):
        box(ax, x, y, label, height=height, fill=fill)
    for left, right in zip(positions, positions[1:]):
        arrow(ax, (left + 1.34, y), (right - 1.34, y))


def note(ax, y, text):
    ax.text(0.2, y, text, fontsize=13, color=INK, va="top", linespacing=1.5)


def save(fig, name):
    fig.savefig(OUTPUT / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUTPUT / f"{name}.png", dpi=180)
    plt.close(fig)
    print(name)


def e1():
    fig, ax = canvas("E1 | Three image classifiers",
                     "Fashion-MNIST • Shapes omit the batch dimension • Source: E1/run_e1.py", 7.4)
    note(ax, 6.2, "SOFTMAX LINEAR  •  7,850 parameters")
    row(ax, 5.45, ["Image\n1 × 28 × 28", "Flatten\n784 values",
                   "Linear\n784 → 10", "10 logits"], height=0.85)
    note(ax, 4.6, "MLP  •  109,386 parameters • Same image input")
    row(ax, 3.85, ["Flatten\n784 values", "Linear 784 → 128\nReLU",
                   "Linear 128 → 64\nReLU", "Linear 64 → 10\n10 logits"], height=0.85)
    note(ax, 3.0, "CNN  •  105,866 parameters • Same image input")
    row(ax, 2.1, ["Conv 3×3: 1 → 16\nReLU + MaxPool 2\n16 × 14 × 14",
                  "Conv 3×3: 16 → 32\nReLU + MaxPool 2\n32 × 7 × 7",
                  "Flatten: 1,568\nLinear 1,568 → 64\nReLU",
                  "Linear 64 → 10\n10 logits"], height=1.2)
    note(ax, 1.1, "Convolutions: stride 1, padding 1. Pooling: kernel 2, stride 2.\n"
         "All heads return raw logits; CrossEntropyLoss applies log-softmax internally.")
    save(fig, "e1_classifiers")


def e2():
    fig, ax = canvas("E2 | Image tokens and self-attention",
                     "Source: E2/run_e2.py • One encoder block • Embedding width 32 • No CLS token", 9.3)
    note(ax, 8.15, "CHOOSE ONE TOKENIZER  •  Input: 1 × 28 × 28")
    for x, label in [(2, "Patch (measured)\nUnfold 4×4, stride 4\nLinear 16 → 32\n49 × 32 tokens"),
                     (6, "CNN stem (measured)\nConv/ReLU/Pool × 2\n1 → 16 → 32 channels\n49 × 32 tokens"),
                     (10, "Row (implemented)\n28 rows of 28 pixels\nLinear 28 → 32\n28 × 32 tokens")]:
        box(ax, x, 6.95, label, width=3.5, height=1.65,
            fill=GREEN if x == 10 else PALE)
        arrow(ax, (x, 6.06), (x, 5.7))
    ax.plot([2, 10], [5.7, 5.7], color=BLUE, linewidth=1.3)
    arrow(ax, (6, 5.7), (6, 5.35))
    box(ax, 6, 4.85, "X = tokens + learned positional embedding  •  N × 32",
        width=10.9, height=0.8)
    arrow(ax, (6, 4.39), (6, 4.05))
    box(ax, 6, 3.4, "A = MSA(X)  •  ManualMSA or nn.MultiheadAttention\n"
        "2 heads × 16 channels: softmax(QKᵀ / √16)V → concatenate → projection\n"
        "Y = LayerNorm(X + A)", width=10.9, height=1.2)
    arrow(ax, (6, 2.74), (6, 2.4))
    box(ax, 6, 1.95, "Z = LayerNorm(Y + FFN(Y))  •  FFN: Linear 32 → 64 → ReLU → Linear 64 → 32",
        width=10.9, height=0.75, size=13)
    arrow(ax, (6, 1.51), (6, 1.2))
    box(ax, 6, 0.72, "Mean over N tokens → 32 values → Linear 32 → 10 → 10 logits",
        width=10.9, height=0.7, fill=GREEN)
    save(fig, "e2_attention")


def e3():
    fig, ax = canvas("E3 | Recurrent classification by image rows",
                     "Source: E3/run_e3.py • One unidirectional recurrent layer • Hidden width 64", 7.1)
    row(ax, 5.4, ["Image\n1 × 28 × 28", "Remove channel axis\n28 steps × 28 pixels",
                  "LSTM or GRU\nOutput: 28 × 64", "Last output h₂₈\n64 values"], height=1.1)
    arrow(ax, (10.5, 4.79), (10.5, 4.15))
    box(ax, 10.5, 3.55, "Linear 64 → 10\n10 logits", fill=GREEN)
    note(ax, 4.25, "UNROLLED RECURRENCE  •  Shared weights at every time step")
    for x, label in [(1.6, "Row 1\n28 pixels"), (4.0, "Row 2\n28 pixels"),
                     (7.1, "Row 28\n28 pixels")]:
        box(ax, x, 3.2, label, width=1.9, height=0.7)
        arrow(ax, (x, 2.79), (x, 2.35))
        box(ax, x, 1.85, "Recurrent cell\nhₜ ∈ ℝ⁶⁴", width=1.9, height=0.8)
    arrow(ax, (2.61, 1.85), (2.99, 1.85))
    arrow(ax, (5.01, 1.85), (5.35, 1.85))
    ax.text(5.55, 1.85, "…", ha="center", va="center", fontsize=18)
    arrow(ax, (5.76, 1.85), (6.09, 1.85))
    note(ax, 1.05, "Initial states are zero. LSTM carries hidden hₜ and cell cₜ; GRU carries hₜ only.\n"
         "LSTM: 24,714 parameters. GRU: 18,698 parameters. CNN baseline: the E1 CNN.")
    save(fig, "e3_recurrent")


def resnet():
    fig, ax = canvas("A1 | ResNet-18 on CIFAR-100",
                     "Source: A1/run_a1.py + torchvision.models.resnet18 • 11,227,812 parameters", 9.0)
    row(ax, 7.35, ["RGB image\n3 × 32 × 32", "Resize + normalize\n3 × 96 × 96",
                   "Conv 7×7, stride 2\nBN + ReLU\n64 × 48 × 48",
                   "MaxPool 3×3\nstride 2, padding 1\n64 × 24 × 24"], height=1.2)
    note(ax, 6.35, "RESIDUAL STAGES  •  The stem output enters layer1 below")
    row(ax, 5.45, ["layer1: 2 blocks\n64 × 24 × 24", "layer2: 2 blocks\n128 × 12 × 12",
                   "layer3: 2 blocks\n256 × 6 × 6", "layer4: 2 blocks\n512 × 3 × 3"], height=1.05)
    ax.plot([11.84, 11.92, 11.92, 0.08, 0.08],
            [7.35, 7.35, 6.65, 6.65, 5.45], color=BLUE, linewidth=1.3)
    arrow(ax, (0.08, 5.45), (0.16, 5.45))
    arrow(ax, (10.5, 4.86), (10.5, 4.25))
    box(ax, 10.5, 3.55, "Global average pool\n512 values\nFC 512 → 100", height=1.2, fill=GREEN)
    note(ax, 4.35, "BASIC BLOCK  •  Two 3×3 convolutions per block")
    box(ax, 1.2, 3.25, "x", width=1.3, height=0.75)
    box(ax, 4.25, 3.25, "Conv → BN → ReLU\nConv → BN\nF(x)", width=3.0, height=1.2)
    box(ax, 7.65, 3.25, "Add + ReLU\ny = ReLU(F(x) + S(x))",
        width=2.5, height=1.0, size=12)
    arrow(ax, (1.91, 3.25), (2.69, 3.25))
    arrow(ax, (5.81, 3.25), (6.34, 3.25))
    ax.plot([1.2, 1.2, 7.65], [2.82, 2.05, 2.05], color=BLUE, linewidth=1.3)
    arrow(ax, (7.65, 2.05), (7.65, 2.69))
    ax.text(4.5, 2.15, "S(x): identity or 1×1 Conv + BN", ha="center", fontsize=13, color=INK)
    note(ax, 1.35, "First blocks of layer2–4 downsample with stride 2 and a projected shortcut.\n"
         "Head output: 100 logits. The original ImageNet stem is retained (not a CIFAR-specific stem).\n"
         "Partial fine-tuning trains layer4 + fc; frozen BatchNorm statistics remain fixed.")
    save(fig, "a1_resnet18")


def swin():
    fig, ax = canvas("A1 | Swin-T on CIFAR-100",
                     "Source: A1/run_a1.py + torchvision.models.swin_t • 27,596,254 parameters", 10.1)
    box(ax, 3, 8.65, "RGB 32×32 → resize 96×96\nImageNet normalization", width=5.2, height=0.95)
    box(ax, 9, 8.65, "Patch embedding: Conv 4×4, stride 4\nPermute + LayerNorm → 24 × 24 × 96",
        width=5.2, height=0.95)
    arrow(ax, (5.66, 8.65), (6.34, 8.65))
    note(ax, 7.8, "HIERARCHICAL STAGES  •  Shapes: H × W × C  •  Patch embedding enters Stage 1")
    row(ax, 6.85, ["Stage 1: 2 blocks\n3 heads\n24 × 24 × 96",
                   "Stage 2: 2 blocks\n6 heads\n12 × 12 × 192",
                   "Stage 3: 6 blocks\n12 heads\n6 × 6 × 384",
                   "Stage 4: 2 blocks\n24 heads\n3 × 3 × 768"], height=1.25)
    ax.plot([11.66, 11.92, 11.92, 0.08, 0.08],
            [8.65, 8.65, 7.97, 7.97, 6.85], color=BLUE, linewidth=1.3)
    arrow(ax, (0.08, 6.85), (0.16, 6.85))
    note(ax, 5.9, "Between stages: Patch Merging = concatenate 2×2 tokens → LayerNorm(4C) → Linear 4C → 2C.")
    arrow(ax, (10.5, 6.16), (10.5, 5.4))
    box(ax, 6, 4.85, "LayerNorm → permute to C × H × W → global average pool → 768 values\n"
        "Linear head 768 → 100 → 100 logits", width=10.9, height=0.9, fill=GREEN)
    note(ax, 4.05, "SWIN BLOCK  •  Pre-norm residual paths")
    box(ax, 6, 3.2, "Y = X + StochasticDepth(WindowAttention(LayerNorm(X)))\n"
        "Z = Y + StochasticDepth(MLP(LayerNorm(Y)))  •  MLP: C → 4C → GELU → C",
        width=10.9, height=1.0, size=14)
    note(ax, 2.35, "Attention uses 7×7 windows with relative position bias; successive blocks use shifts 0 and 3.\n"
         "Window padding: 24→28, 12→14, 6→7 and 3→7; outputs are cropped back after attention.\n"
         "At 96×96 input, shifts are disabled in the 6×6 and 3×3 stages (one padded window).\n"
         "Patch merging is separate from attention padding. There is no CLS token.\n"
         "Partial fine-tuning trains features[-1] (Stage 4) + head; the final norm stays frozen.")
    save(fig, "a1_swin_t")


if __name__ == "__main__":
    for draw in (e1, e2, e3, resnet, swin):
        draw()
