# CO5085 Exercises: Deep Learning for Computer Vision

This repository completes E1, E2, and E3 of CO5085 with Fashion-MNIST. The code is intentionally short and written for a beginner to read.

| Exercise | Main comparison | Best test accuracy |
| --- | --- | ---: |
| E1 | Softmax, MLP, CNN | CNN: 0.8686 |
| E2 | Manual MSA, PyTorch MSA, patch/CNN-stem tokens | PyTorch MSA + CNN stem: 0.7578 |
| E3 | LSTM rows, GRU rows, CNN baseline | CNN baseline: 0.8686 |

## Run the code

```bash
pip install -r requirements.txt
python -m E1.run_e1
python -m E2.run_e2
python -m E3.run_e3
```

Run commands from the repository root. Fashion-MNIST downloads automatically on first use. Each script writes figures and JSON metrics to `reports/`.

## Report and website

- [LaTeX source](CO5085%20Report%20Latex/report.tex)
- [PDF report](CO5085%20Report%20Latex/report.pdf)
- [GitHub Pages source](index.html)

The models use a shared 48,000 / 12,000 / 10,000 train-validation-test split with seed 42. See the report for model details, learning curves, confusion matrices, and an explanation of each result.
