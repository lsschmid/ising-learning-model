# Ising Learning Model

![License](https://img.shields.io/badge/License-MIT-blue.svg)

This repository contains the code and resources for the Ising Learning Model, a general learning scheme for classical and quantum Ising machines. It represents a ready to use Python package of the idea proposed in the corresponding ArXiv paper.


## Directory Structure

The repository is organized as follows:

```
├── LICENSE.md
├── README.md
├── pyproject.toml
└── src
    └── ising_learning_model
        ├── __init__.py
        ├── data.py
        ├── exact_model.py
        ├── model.py
        ├── qpu_model.py
        ├── sim_anneal_model.py
        └── utils.py
```

- `LICENSE.md`: The license file for this project.
- `README.md`: You are currently reading this file, which serves as an introduction and documentation for the project.
- `pyproject.toml`: A configuration file for Python dependencies and project information.

The `src` directory contains the Python source code for the Ising Learning Model. The structure of the `ising_learning_model` package is as follows:

- `__init__.py`: Initialization file for the package.
- `data.py`: Module for handling data related to the Ising Learning Model. In particular methods to automatically create Datasets for the function approximation, and the bars and stripes taks described in the paper.
- `exact_model.py`: Implementation using a brute force samples as the underlying Ising machine.
- `model.py`: Implementation of the Ising Learning Model, serving as an abstract base class for the other models.
- `qpu_model.py`: Implementation of the Ising Learning Model using a quantum processing unit (QPU) as the underlying Ising machine.
- `sim_anneal_model.py`: Implementation of the Ising Learning Model using simulated annealing as the underlying Ising machine.
- `utils.py`: Utility functions and tools used in the Ising Learning Model, in particular, helper functions to facilitate hidden nodes initialization etc.

## Getting Started

To get started with the Ising Learning Model, you can refer to the code comments within the source code files (exact_model.py, model.py, qpu_model.py, sim_anneal_model.py, and utils.py) for detailed explanations of the functionality and usage of each module.

Additionally, you can find more comprehensive documentation and a mathematical characterization of the training process in the corresponding scientific paper.

Installation
To install the Ising Learning Model package locally, you can use pip:

```bash
pip install .
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.