# Cleo traveling wave rejection experiment

This is the code for the traveling wave rejection experiment in the Cleo paper ([preprint](https://www.biorxiv.org/content/10.1101/2023.01.27.525963)).
It is an implementation of the model in [Moldakarimov et al., 2018](https://pubmed.ncbi.nlm.nih.gov/29712831/)

Install dependencies and source code with [`uv pip`](https://docs.astral.sh/uv/):
```bash
uv pip install .
```
And set up a virtual environment:
```bash
uv venv && source .venv/bin/activate
```

Run simulations and produce figure with [Taskfile](https://taskfile.dev/) (a modern Make replacement):
```bash
task
```
