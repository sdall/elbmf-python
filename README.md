# Efficiently Factorizing Boolean Matrices using Proximal Gradient Descent

This repository provides a Python library that implements the _Elastic Boolean Matrix Factorization_ (Elbmf) algorithm using PyTorch. It provides an efficient methods for factorizing binary matrices into low-rank matrices using a continuous relaxation, an elastic-net inspired Boolean regularization, and proximal gradient descent.

## Acknowledgments

This project is based on the research paper 'Efficiently Factorizing Boolean Matrices using Proximal Gradient Descent'

```bibtex
@inproceedings{dalleiger2022efficiently,
    title={Efficiently Factorizing Boolean Matrices using Proximal Gradient Descent},
    author={Sebastian Dalleiger and Jilles Vreeken},
    booktitle={Thirty-Sixth Conference on Neural Information Processing Systems (NeurIPS)},
    year={2022}
}
```

## Installation

```sh
pip install torch
pip install git+https://github.com/sdall/elbmf-python
```

## Usage

```python
import torch
from elbmf import elbmf

X = torch.randint(0, 2, (100, 100))

U, V = elbmf(
    X                   = X,                  # Boolean input matrix 
    n_components        = 20,                 # number of components 
    l1reg               = 0.01,               # l1 coefficient 
    l2reg               = 0.02,               # l2 coefficient 
    regularization_rate = lambda t: 1.02**t,  # monotonically increasing regularization-rate (function)  
    maxiter             = 3000,               # max number of iterations 
    tolerance           = 1e-8,               # the absolute allowed difference between the current and previous losses determines the convergence of elbmf. 
    beta                = 0.0001,             # inertial coefficient of iPALM 
    callback            = None,               # e.g. lambda t, U, V, fn: ... 
    with_rounding       = True)               # rounds U and V in case of early stopping. 
```

If the resulting reconstruction is unexpected or not ideal, 
you might want to try to tweak `l1reg`, `l2reg`, `maxiter`, and most importantly the `regularization_rate`,
with disabled `with_rounding` (for debugging purposes) and enabled `callback`.

## Contributing

Contributions to Elbmf are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the GitHub repository: https://github.com/sdall/elbmf-python 

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact

If you have any questions or inquiries, please contact sdalleig@mpi-inf.mpg.de.

Feel free to reach out with any feedback or suggestions.

