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


## How to update the documentation? 
1. Make changes to `docs/` content on the `main` branch
2. re-build the book with `jupyter-book build docs/` 
3. from the `docs/` folder run `ghp-import -n -p -f _build/html` to push the newly built HTML to the `gh-pages` branch
4. 