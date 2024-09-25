# TD2C (Time-Dependency to Causality)
**TD2C (Time-Dependency to Causality)** is a library for time series causal discovery. It focuses on computing asymmetric conditional mutual information terms, known as descriptors, within the Markov blankets of variable pairs.

## Clone 
Make sure you only get the latest commit 
```
git clone --depth 1 https://github.com/gmpal/td2c
```

## Setup
To get started with TD2C, follow these steps:

```
sudo apt update
sudo apt install -y build-essential libssl-dev libffi-dev python3-dev zlib1g-dev libbz2-dev libsqlite3-dev libreadline-dev libgdbm-dev liblzma-dev tk-dev
```

1. Create a virtual environment with the tool of your choice:
    1. Use `pyenv` (recommended, tested):
        ```
        curl https://pyenv.run | bash
        ```

    2. Load pyenv automatically by appending the following to ~/.bash_profile if it exists, otherwise ~/.profile (for login shells) and  ~/.bashrc (for interactive shells)
     
        ```
        export PYENV_ROOT="$HOME/.pyenv"
        [[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init -)"
        ```
    3. Restart the shell. Then 
        pyenv install 3.10
        pyenv virtualenv 3.10 td2c
        pyenv shell td2c
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
