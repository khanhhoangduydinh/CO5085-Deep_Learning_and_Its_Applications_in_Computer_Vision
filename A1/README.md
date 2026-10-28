# A1: CNN vs Transformer on CIFAR-100

This folder is the A1 comparison required by CO5085.  It uses CIFAR-100, which
has 100 fine classes, 50,000 official training images, and 10,000 official test
images.  The original dataset is by Krizhevsky, Nair, and Hinton.  Its public
download page does not state an explicit licence, so this project uses it only
for coursework and cites the original technical report.

## What is compared?

| Family | torchvision model | `scratch` | `head` | `partial` | `full` |
| --- | --- | --- | --- | --- | --- |
| CNN | ResNet-18 | random weights, train all | ImageNet, train head | ImageNet, train `layer4` + head | ImageNet, train all |
| Transformer | Swin-T | random weights, train all | ImageNet, train head | ImageNet, train final stage + head | ImageNet, train all |

All eight rows use the same fixed 40,000 / 10,000 train-validation split from
the official CIFAR-100 training partition.  The official 10,000-image test set
is used only once, after validation selects the winning configuration.

## Run

From the repository root:

```bash
pip install -r requirements.txt
python -m A1.run_a1
```

The first run downloads `uoft-cs/cifar100` through Hugging Face and pretrained
ImageNet weights through torchvision.  The script needs an NVIDIA GPU for a
reasonable runtime; it can run on CPU but will be much slower.

The run writes `reports/a1_results.json` and five figures under
`reports/figures/`.  It uses seed 42, 96 x 96 inputs, batch size 64, and three
epochs for every configuration.  These are a controlled coursework budget, not
a claim of a fully tuned CIFAR-100 benchmark.

## Read the code in this order

1. `make_stratified_indices` makes the same 400-train / 100-validation split
   for each of the 100 fine classes.
2. `get_loaders` applies random crop and horizontal flip only to training
   images; validation and test images receive deterministic transforms.
3. `build_model` replaces the 1,000-class ImageNet head with a 100-class head,
   then applies the chosen freeze mode.
4. `train_epoch` performs forward pass, cross-entropy loss, backward pass, and
   `optimizer.step()` explicitly.
5. `select_best_experiment` uses validation accuracy only.  The selected model
   is evaluated once on the official test partition.

See `CO5085 Report Latex/a1_report.pdf` for the formulas, experiment protocol,
measured results, error analysis, limitations, and references.
