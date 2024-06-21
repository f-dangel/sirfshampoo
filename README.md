# SIRFShampoo: Structured Inverse- and Root-Free Shampoo

This package contains the official PyTorch implementation of our inverse- and
square-root free Shampoo optimizer from our [ICML
paper](https://arxiv.org/abs/2402.03496) 'Can We Remove the Square-Root in
Adaptive Gradient Methods? A Second-Order Perspective' (the 'IF-Shampoo'
optimizer in [Fig. 3](https://openreview.net/pdf?id=vuMD71R20q)).

Some highlights of the optimizer:

- Numerically stable, even in `bfloat16`, due to a fully matrix-multiplication
  based update (no matrix decompositions)
- Compatible with any architecture as the pre-conditioner only uses mini-batch
  gradients
- Kronecker factors can have structures to reduce memory and computation,
  thanks to [our previous SINGD work](https://arxiv.org/pdf/2312.05705)

## Installation

- Stable (recommended):
  ```bash
  pip install sirfshampoo
  ```

- Latest version from GitHub `main` branch:
  ```bash
  pip install git+https://github.com/f-dangel/sirfshampo.git@main
  ```

## Usage

 - [Basic
   example](https://sirfshampoo.readthedocs.io/en/latest/generated/gallery/example_01_basic/)

## Limitations

- `SIRFShampoo` assumes that the objective is an **average** over per-example losses.

- The code has stabilized only recently. Expect things to break and help us
  improve by filing issues.

## Citation

If you find this code useful for your research, consider citing the paper:

```bib

@inproceedings{lin2024can,
  title =        {Can We Remove the Square-Root in Adaptive Gradient Methods? A
                  Second-Order Perspective},
  author =       {Wu Lin and Felix Dangel and Runa Eschenhagen and Juhan Bae and
                  Richard E. Turner and Alireza Makhzani},
  booktitle =    {International Conference on Machine Learning (ICML)},
  year =         2024,
}

```
