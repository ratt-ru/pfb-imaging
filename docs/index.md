# pfb-imaging

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://ratt-ru.github.io/pfb-imaging/)

`pfb-imaging` is a flexible radio interferometric imaging suite based on the preconditioned forward-backward algorithm. It exposes common steps in a typical radio astronomy imaging pipeline as individual applications (a.k.a **workers**) which can be run from the command line or strung together into a highly customizable imager using [`stimela`](https://github.com/caracal-pipeline/stimela). It is part of the **africanus** ecosystem documented in the paper following series:

- [**Africanus I. Scalable, distributed and efficient radio data processing with Dask-MS and Codex Africanus**](https://arxiv.org/abs/2412.12052)
- [**Africanus II. QuartiCal: calibrating radio interferometer data at scale using Numba and Dask**](https://arxiv.org/abs/2412.10072)
- [**Africanus III. pfb-imaging -- a flexible radio interferometric imaging suite**](https://arxiv.org/abs/2412.10073)
- [**Africanus IV. The Stimela2 framework: scalable and reproducible workflows, from local to cloud compute**](https://arxiv.org/abs/2412.10080)


## Key Features

- Uses [`dask-ms`](https://github.com/ratt-ru/dask-ms) to interface with measurement set-like objects exposed as `xarray` datasets 
- Interfaces with [`QuartiCal`](https://github.com/ratt-ru/QuartiCal) gain tables to apply gains on the fly
- On the fly averaging using the averaging modules in [`codex-africanus`](https://github.com/ratt-ru/codex-africanus)
- Exposes `stimela` cabs for each worker making it easy to construct custom imaging pipelines. 

## Quick Start

### Installation

```bash
# Install from PyPI
pip install pfb-imaging

# Or install from source
git clone https://github.com/ratt-ru/pfb-imaging.git
cd pfb-imaging
pip install -e .
```

### Basic Usage

```bash
# Get the list of available workers 
pfb --help

# Get help on a specific worker
pfb <worker> --help

# Or using stimela
stimela doc -l pfb::stimela_cabs.yml
stimela doc pfb::stimela_cabs.yml pfb.init
```

A typical imaging pipeline strings these workers together in a specific order. For example:

```mermaid
graph LR
    A[pfb init] --> B[pfb grid]
    B --> C[pfb kclean/sara]
    C --> D[pfb restore]
    D --> E[pfb degrid]
```

## Getting Help

- 📖 **Documentation**: [https://ratt-ru.github.io/pfb-imaging/](https://ratt-ru.github.io/pfb-imaging/)
- 🐛 **Issues**: [GitHub Issues](https://github.com/ratt-ru/pfb-imaging/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/ratt-ru/pfb-imaging/discussions)

## Contributing

We welcome contributions! Please see our [Contributing Guide](contributing.md) for details on how to get started.

## Citation

If you use `pfb-imaging` in your research, please cite:

```bibtex
@misc{bester2024africanusiiipfbimaging,
      title={Africanus III. pfb-imaging -- a flexible radio interferometric imaging suite}, 
      author={Hertzog L. Bester and Jonathan S. Kenyon and Audrey Repetti and Simon J. Perkins and Oleg M. Smirnov and Tariq Blecher and Yassine Mhiri and Jakob Roth and Ian Heywood and Yves Wiaux and Benjamin V. Hugo},
      year={2024},
      eprint={2412.10073},
      archivePrefix={arXiv},
      primaryClass={astro-ph.IM},
      url={https://arxiv.org/abs/2412.10073}, 
}
```

## License

`pfb-imaging` is licensed under the MIT License. See [LICENSE](https://github.com/ratt-ru/pfb-imaging/blob/main/LICENSE) for details.