# CO5085 Exercises: Deep Learning for Computer Vision

This repository completes E1, E2, E3, and Assignment A1 of CO5085. The code is intentionally direct and written for a beginner to read.

Repository: [CO5085-Deep_Learning_and_Its_Applications_in_Computer_Vision](https://github.com/khanhhoangduydinh/CO5085-Deep_Learning_and_Its_Applications_in_Computer_Vision)

| Exercise | Main comparison | Best test accuracy |
| --- | --- | ---: |
| E1 | Softmax, MLP, CNN | CNN: 0.8686 |
| E2 | Manual MSA, PyTorch MSA, patch/CNN-stem tokens | PyTorch MSA + CNN stem: 0.7578 |
| E3 | LSTM rows, GRU rows, CNN baseline | CNN baseline: 0.8686 |
| A1 | ResNet-18 vs Swin-T on CIFAR-100; scratch and ImageNet transfer modes | Swin-T full fine-tune: 0.8173 test accuracy |

## Run the code

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khanhhoangduydinh/CO5085-Deep_Learning_and_Its_Applications_in_Computer_Vision/blob/main/CO5085_Exercises_E1_E2_E3.ipynb)

```bash
pip install -r requirements.txt
python -m E1.run_e1
python -m E2.run_e2
python -m E3.run_e3
python -m A1.run_a1
```

Run commands from the repository root, or open [`CO5085_Exercises_E1_E2_E3.ipynb`](CO5085_Exercises_E1_E2_E3.ipynb) to execute interactively on Google Colab with line-by-line pedagogical explanations. Fashion-MNIST or CIFAR-100 downloads automatically on first use. Each script writes figures and JSON metrics to `reports/`. A1 also downloads official torchvision ImageNet weights on first use.

## Report and website

- [Interactive Colab Notebook (E1–E3)](CO5085_Exercises_E1_E2_E3.ipynb)
- [E1-E3 PDF report (exercises.pdf)](reports/exercises.pdf)
- [LaTeX source](CO5085%20Report%20Latex/report.tex)
- [PDF report](CO5085%20Report%20Latex/report.pdf)
- [A1 LaTeX source](CO5085%20Report%20Latex/a1_report.tex)
- [A1 PDF report](CO5085%20Report%20Latex/a1_report.pdf)
- [GitHub Pages source](index.html)

E1-E3 use a shared 48,000 / 12,000 / 10,000 train-validation-test split with seed 42. A1 uses a stratified 40,000 / 10,000 split of the official CIFAR-100 training partition, then one final 10,000-image test evaluation. See both reports for model details, learning curves, confusion matrices, and an explanation of each result.
