# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Python library implementing causal inference algorithms (DML, MIM-DRCFR, SRCVAE). No web services, databases, or external dependencies beyond Python packages. All code lives under `causal_inference_project/`.

### Dependencies

No `requirements.txt` or `pyproject.toml` exists. Dependencies are inferred from imports:
- **torch** (CPU-only variant is sufficient): used by DRCFR and SRCVAE algorithms
- **scikit-learn**: used by DML algorithm
- **numpy**: core numerical library (usually pre-installed)
- **pytest**: test runner
- **pandas**: optional, imported in `dml_example.py`

Install PyTorch CPU separately from PyPI torch index, then other packages from default PyPI:
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn pytest pandas
```

### Running tests

```bash
python3 -m pytest causal_inference_project/tests/ -v
```

All 29 tests should pass.

### Running examples

```bash
python3 causal_inference_project/examples/dml_example.py
python3 causal_inference_project/examples/drcfr_example.py
python3 causal_inference_project/examples/srcvae_example.py  # has a pre-existing unpacking bug
```

### Running the Streamlit UI

```bash
streamlit run causal_inference_project/app.py --server.port 8501 --server.headless true
```

### Gotchas

- PyTorch must be installed from the CPU-only index (`https://download.pytorch.org/whl/cpu`) separately before installing other packages, because mixing `--index-url` with default PyPI causes resolution failures for non-torch packages.
- Ensure `~/.local/bin` is on PATH for `pytest` CLI access (or use `python3 -m pytest`).
- No linter is configured in this project.
