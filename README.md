# TD2C (Time-Dependency to Causality)
**TD2C (Time-Dependency to Causality)** is a library for time series causal discovery. It focuses on computing asymmetric conditional mutual information terms, known as descriptors, within the Markov blankets of variable pairs.

## Setup
To get started with TD2C, follow these steps:

1. Create a virtual environment with the tool of your choice:
    1. Use `pyenv` (recommended):
        ```
        curl https://pyenv.run | bash
        pyenv virtualenv 3.10 td2c
        pyenv shell td2c
        ```
    2. Use `conda`:
        ```
        conda create --name td2c python=3.8.19 ipython
        conda activate td2c
        ```

2. Install Poetry for dependencies management:
    ```
    pip install --upgrade pip
    pip install poetry
    ```

3. Install dependencies from `poetry.lock` file:
    ```
    poetry install
    ```

4. If the above fails, you can resolve dependencies and try again. This will take a few minutes. 
    ```
    poetry lock
    poetry install
    ```

5. Setup completed. 

## Usage
Check the notebooks in the [notebooks folder](./notebooks) for detailed examples and explanations:

- [00_data_generation.ipynb](./notebooks/00_data_generation.ipynb): Demonstrates how to generate synthetic data for testing the TD2C library.
- [01_descriptors_computation.ipynb](./notebooks/01_descriptors_computation.ipynb): Shows how to compute the descriptors using the TD2C library.
- [02_run_competitors.ipynb](./notebooks/02_run_competitors.ipynb): Provides examples of running competitor algorithms for comparison.
- [03_collect_results.ipynb](./notebooks/03_collect_results.ipynb): Illustrates how to collect and analyze the results from the experiments.
