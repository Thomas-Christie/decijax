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

## Building Documentation Locally

If you made changes that affect the documentation, build the docs locally to
verify them before opening a pull request.

### Previewing the docs

To start a live-reloading development server at http://127.0.0.1:8000:

```
uv run mkdocs serve
```

To reproduce exactly what CI runs (a strict build that executes every example
notebook and fails on any broken link or output):

```
DECIJAX_DOCS_EXECUTE=1 DECIJAX_DOCS_STRICT=1 uv run mkdocs build --strict
```

Executing the notebooks can be slow, and `mkdocs serve` re-runs them on every
reload. While you're iterating on non-notebook content, you can skip execution
by setting `DECIJAX_DOCS_EXECUTE=0`, which transcribes the notebook source
without running it:

```
DECIJAX_DOCS_EXECUTE=0 uv run mkdocs serve
```

### Adding a new example notebook

Example notebooks live in the `examples/` directory as `jupytext` "percent"
`.py` files (see [Notebooks](#notebooks) above for converting to and from
`.ipynb`). At docs-build time, `docs/scripts/convert_examples.py` (wired in via
the `gen-files` plugin in `mkdocs.yml`) runs each example, renders it to
Markdown, and writes it into the virtual `_examples/` tree — there is no
checked-in Markdown to keep in sync.

To add a new example:

1. Place the `.py` notebook in `examples/`.
2. Add a link to the rendered page under the appropriate section of `nav` in
   `mkdocs.yml`, pointing at `_examples/<notebook-name>.md`. For instance, the
   intro example is listed as:
   ```yaml
   nav:
     - 💡 Background:
         - Intro to Bayesian Optimisation: _examples/intro_to_bo.md
   ```
3. Preview with `uv run mkdocs serve` to confirm it executes and renders as
   expected.
