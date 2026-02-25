# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Pure Python causal inference library implementing three algorithms (DML, MIM-DRCFR, SRCVAE) under `causal_inference_project/`. No web services, databases, or Docker needed. See `causal_inference_project/README.md` for algorithm details.

### Running tests

```
python3 -m pytest causal_inference_project/tests/ -v
```

**Pre-existing test failures (not environment issues):**
- `test_dml_binary_treatment` — statistical tolerance assertion fails (estimate outside ±0.2 of true effect)
- `test_reconstruction_bce_loss` — missing `import torch.nn.functional as F` in `test_srcvae.py`

### Running examples

```
python3 causal_inference_project/examples/dml_example.py
python3 causal_inference_project/examples/drcfr_example.py
```

**Note:** `srcvae_example.py` has a pre-existing bug (unpacking mismatch from `train_test_split`).

### Dependencies

No `requirements.txt` exists in the repo. Dependencies are inferred from imports: `numpy`, `torch`, `scikit-learn`, `pytest`, `pandas`. The update script installs these via pip.

### Gotchas

- `python` is not available on the VM; use `python3`.
- pip installs to `~/.local/bin` which may not be on PATH; use `python3 -m pytest` instead of bare `pytest`.
- No linter is configured in this project (no flake8, ruff, pylint, mypy config).
