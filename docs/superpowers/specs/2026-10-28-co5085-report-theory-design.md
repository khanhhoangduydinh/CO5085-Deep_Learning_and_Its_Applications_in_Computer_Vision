# CO5085 Report Theory Expansion Design

## Intent

Make the submitted report strong enough for a beginner to explain in a presentation or defence. The report must state what every model does, why the formula is used, and how the corresponding Python code moves data through training and evaluation.

## Decision

Rename the public repository to `CO5085-Deep_Learning_and_Its_Applications_in_Computer_Vision`, the exact English course name after its course code. Keep the existing Fashion-MNIST experiments and measured results unchanged.

## Approaches considered

1. Add a separate theory appendix. This keeps the results short, but separates formulas from the E1--E3 code that uses them.
2. Add theory inside each exercise section. This directly connects a formula, an algorithm, and the observed result. **Selected.**
3. Replace code with a long textbook-style derivation. This would not suit the simple beginner code and would make the report harder to defend.

## Report changes

- Add a shared workflow section: tensor representation, fixed split, cross-entropy, softmax, gradient descent, and accuracy.
- Add E1 theory: linear classifier, MLP ReLU, convolution, pooling, and the exact training-loop stages.
- Add E2 theory: patch/CNN-stem tokenization, Q/K/V, scaled dot-product attention, multi-head merging, residual connection, and manual-versus-library comparison.
- Add E3 theory: row sequence representation, LSTM cell state, GRU update gate, last hidden state classification, and the reason CNN is an appropriate baseline.
- Use short formulas with every symbol defined in plain language; use numbered algorithms that match `common.py`, `E1/run_e1.py`, `E2/run_e2.py`, and `E3/run_e3.py`.
- Keep actual output metrics, figures, and the existing title/header layout unchanged.

## Constraints

- Do not claim data augmentation, pretraining, more epochs, or metrics that were not executed.
- Keep wording simple and avoid unexplained mathematical notation.
- Compile the PDF twice and visually inspect pages containing the formulas and algorithms.
- Update the Pages repository hyperlink after the GitHub rename and verify both the new Pages URL and report link.
