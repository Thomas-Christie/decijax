# decijax

`decijax` is an early-stage library which aims to provide algorithms for a
variety of sequential decision-making problems. It currently provides implementations
of several acquisition functions for Bayesian optimisation, including
probability of improvement, expected improvement and Thompson sampling. The
implementations are built upon the JAX library, enabling automatic differentiation,
vectorisation, and just-in-time (JIT) compilation for high performance. This allows for efficient research, development, and deployment of decision-making agents.

> **⚠️ Warning**
>
> `decijax` is currently under active development, and the API is likely to
> change in the near future.

-----

## Table of Contents

- [decijax](#decijax)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Documentation](#documentation)
  - [Contributing](#contributing)
  - [License](#license)

## Installation

`decijax` requires Python 3.11 or later. Install the latest release from PyPI with:

```bash
pip install decijax
```

## Documentation

Available at [https://thomas-christie.github.io/decijax/](https://thomas-christie.github.io/decijax/).

## Contributing

Please refer to the [contributing guidelines](https://github.com/Thomas-Christie/decijax/blob/main/CONTRIBUTING.md) file for guidelines on how to contribute to the
project.

## License

`decijax` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
