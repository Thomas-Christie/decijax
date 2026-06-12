# Contributing Guidelines

Thanks for your interest in contributing to `decijax`! We welcome contributions from the community to help improve the project. Please follow the guidelines below to ensure a smooth contribution process.

## Issue Tracking

Before starting work on a new feature or significant change, please open an issue to discuss the proposal. This helps avoid duplicate effort and ensures your contribution aligns with the project's direction. For bug fixes, check if an existing issue already describes the problem before creating a new one.

## How to Contribute

1. **Fork the Repository**: Start by forking the `decijax` repository on GitHub to your own account.
2. **Clone Your Fork**: Clone your forked repository to your local machine using:
   ```
   git clone <your-fork-url>
   ```
3. **Install Dependencies**: Navigate to the project directory and install the required dependencies. We use [uv](https://docs.astral.sh/uv/) for managing dependencies, and highly recommend that you do too. To get started quickly, after installing `uv`, you can simply run:
    ```
    uv sync --all-extras
    ```
    and this will create a virtual environment and install all necessary packages for development.

4. **Set up pre-commit hooks**: We use pre-commit hooks to maintain code quality. To initialise pre-commit hooks, run the following command:
   ```
   uv run pre-commit install
   ```
5. **Create a Branch**: Create a new branch for your feature or bug fix:
   ```
   git checkout -b <new-feature-branch>
   ```
6. **Make Changes**: Implement your changes in the codebase. Please ensure that your code includes appropriate tests.
7. **Run Tests**: Before committing your changes, run the unittests to ensure everything is working correctly:
   ```
   uv run pytest
   ```
8. **Commit Changes**: Commit your changes with a clear and descriptive commit message.
9. **Push Changes**: Push your changes to your forked repository:
   ```
   git push origin <new-feature-branch>
   ```
10. **Create a Pull Request**: Open a pull request in the [original `decijax` repository](https://github.com/Thomas-Christie/decijax).

## Notebooks

We use `jupytext` to manage Jupyter notebooks, storing them as `.py` files for better version control. In order to convert between `.ipynb` and `.py` formats, you can use the following commands:

- To convert a `.ipynb` file to a `.py` file:
  ```
  jupytext --to py:percent <notebook>.ipynb
  ```

- To convert a `.py` file back to a `.ipynb` file:
  ```
  jupytext --to ipynb <notebook>.py
  ```

Note that you can prepend `uv run` to these commands if you have `jupytext` installed in your `uv`-managed virtual environment.
