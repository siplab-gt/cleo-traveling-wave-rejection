# Cleo traveling wave rejection experiment

This is the code for the traveling wave rejection experiment in the Cleo paper ([preprint](https://www.biorxiv.org/content/10.1101/2023.01.27.525963)).
It is an implementation of the model in [Moldakarimov et al., 2018](https://pubmed.ncbi.nlm.nih.gov/29712831/)

## Installation and use

We use [uv](https://docs.astral.sh/uv/) to manage dependencies and virtual environments.
It is much faster than conda/mamba and includes dependency management features.

First set up a virtual environment:
```bash
uv venv && source .venv/bin/activate
```

Then install dependencies and the source code `cleo_pe1` (PE1 refers to prospective experiment 1) package:
```bash
uv pip install .
```

Run simulations and produce figure with [Task](https://taskfile.dev/) (a modern Make alternative):
```bash
task
```
