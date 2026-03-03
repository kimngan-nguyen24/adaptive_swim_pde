# ADAPTIVE SWIM-PDE

This repository contains the research code developed for the Bachelor's thesis [[1]](#ref1).

The project extends the **Frozen-PINNs** framework [[2]](#ref2) for solving partial differential equations (PDEs) using sampled
neural networks. Frozen-PINNs approximate PDE solutions as a linear combination of basis functions induced by randomly
sampled weights and biases. The corresponding coefficients are determined by solving ordinary differential equations
(ODEs) that depend only on time.

This work enhances the original framework by introducing **adaptive activation functions** into the construction of
basis functions. In particular, it incorporates activation functions with trainable parameters, including rational
activation functions, to improve approximation quality and flexibility.

The folder `examples/` contains Jupyter notebooks reproducing the numerical experiments presented in [[1]](#ref1).

---

## Installation

Clone the repository and install the package from the root directory:

```bash
pip install .
```

The dependencies [`swimpde`](https://gitlab.com/fd-research/swimpde#id1) and [`swimnetworks`](https://gitlab.com/fd-research/swimnetworks) are installed automatically.
It is recommended to use a virtual environment.

## Citation

If you use this code in your research, please cite:

> <a id="ref1">[1]</a> N. Nguyen, "Solving Partial Differential Equations by Sampled Neural Networks with Adaptive Activation Functions", 
Bachelor's Thesis, Technical University of Munich, 2026.

> <a id="ref2">[2]</a> C. Datar, T. Kapoor, A. Chandra, Q. Sun, I. Burak, E. L. Bolager, A. Veselovska, M.
Fornasier, and F. Dietrich, “Solving partial differential equations with sampled neural
networks”, arXiv preprint arXiv:2405.20836, 2024.

> <a id="ref3">[3]</a> E. L. Bolager, I. Burak, C. Datar, Q. Sun, and F. Dietrich, “Sampling weights of deep
neural networks”, Advances in Neural Information Processing Systems, vol. 36, pp. 63 075–
63 116, 2023.