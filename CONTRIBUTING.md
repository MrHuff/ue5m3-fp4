# Contributing

Thank you for helping improve the UE5M3 FP4 reference. The project treats
numerical behavior and experiment identity as public API: a change that looks
like a performance refactor can change the represented format, scale refresh,
rounding distribution, or measured result.

## Before opening a change

1. Open or reference an issue describing the behavior being changed.
2. Keep private datasets, checkpoints, credentials, bucket names, and cluster
   configuration out of commits, examples, logs, and issue text.
3. Separate portable numerical code from architecture, runtime, and launcher
   integrations.
4. Do not claim native execution, throughput equivalence, or paper-result
   reproduction unless the corresponding backend and evidence are included.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
python -m pytest
python examples/tiny_mlp.py
python -m build
```

Before requesting release review, install the built wheel into a clean
environment and repeat the tests and example. GPU or native-backend changes
must also record the hardware, PyTorch, CUDA, kernel/backend, and matmul-policy
versions used for numerical checks.

## Numerical changes

A numerical change should include:

- a test that fails before the change and passes afterward;
- an explicit description of affected operands and GEMMs;
- the format, block orientation, scale target, refresh rule, and rounding mode;
- whether checkpoint or process-local state changes;
- updated provenance/schema output when result identity changes; and
- a note explaining whether prior measurements remain comparable.

Do not silently replace periodic sample-and-hold with a rolling maximum,
`StochasticFast` with an ideal Bernoulli draw, or decoded-Torch matmul with a
different accumulation/output model.

## Model integrations

Architecture integrations must provide a reviewed selector or exact allowlist
of converted linears. They should fail closed on unsupported subclasses,
aliases, attention backends, and checkpoint layouts. Include a test that
asserts the complete converted-module set; a count alone is not sufficient.

## Documentation and results

Keep BF16 controls, software FP4 fake quantization, probe-matched emulation,
and native NVFP4 results clearly separated. Published measurements should have
immutable checkpoint/data identities and machine-readable provenance. Report
single-run evidence as single-run evidence and do not infer statistical
significance from checkpoints along one trajectory.

Security-sensitive reports must follow [`SECURITY.md`](SECURITY.md), not a
public issue.
