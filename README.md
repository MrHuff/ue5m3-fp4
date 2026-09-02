# UE5M3 FP4

Portable PyTorch reference code for **UE5M3 block scaling with E2M1 FP4**.
The package covers the numerical recipe used for low-precision pretraining and
the explicit scale lifecycle needed to evaluate a learned checkpoint with FP4
fake quantization after loading it.

This repository is a **public alpha**. It is useful for inspecting, testing,
and integrating the numerical method, but it is not a one-command reproduction
of the paper's Nemotron-H 8B experiments. In particular, this implementation
decodes fake-quantized operands and calls a PyTorch matrix multiplication; it
does not claim native UE5M3 execution or hardware throughput.

## What is implemented

The reference recipe makes each numerical choice explicit:

| Choice | Proposed reference setting |
|---|---|
| Payload | signed E2M1 FP4 |
| Block scale | unsigned E5M3 (UE5M3) |
| Block size | 16 values |
| Default tensor-scale target | 448 |
| Delayed tensor scaling | periodic sample-and-hold, D=50 |
| Activation and weight rounding | round-to-nearest, ties-to-even |
| Backward rounding | 8-bit-midpoint `StochasticFast`, only for `dY` |
| Weight scaling | two-dimensional |
| Randomized Hadamard transform | disabled |
| Inference activation scaling | current D=1, cold D=50 replay, or calibrated/frozen |

The D=50 cache is not a rolling maximum. Each operand's current global `amax`
is sampled on steps 1, 51, 101, and so on, and that exact value is held between
refreshes. Activations, weights, and upstream gradients have independent
per-linear state.

The package includes:

- Torch implementations of E2M1 payload and UE5M3 scale quantization;
- the exact eight-bit-midpoint stochastic-rounding rule used by the extracted
  implementation;
- an autograd-aware FP4 linear reference with selective `dY` rounding;
- strict YAML parsing for the proposed training recipe;
- deterministic post-load inference controllers for three activation-scale
  strategies;
- explicit model-conversion coverage and fail-closed lifecycle checks;
- a local, content-addressed next-token validation-loss evaluator; and
- CPU unit tests for formats, scaling, backward paths, conversion, inference,
  and evaluation provenance.

The public alpha intentionally excludes TorchTitan and Nemotron-H model
adapters, distributed checkpoint conversion, cloud/cluster launchers,
Transformer Engine, native kernels, private datasets, and internal artifact
locations. The probe-matched GEMM emulator used for the paper's closest native
comparison is also not in this first slice.

## Repository layout

```text
src/ue5m3_fp4/
  formats.py             E2M1/UE5M3 representation and block fake quantization
  recipe.py              strict training-recipe schema
  recipes/               YAML recipes shipped in both wheel and sdist
  nn/                    FP4 linear and explicit module conversion
  scaling/training.py    D=50 periodic sample-and-hold training state
  scaling/inference.py   post-load inference lifecycle and provenance
  eval/validation.py     exact local next-token NLL
examples/tiny_mlp.py     checkpoint-reload smoke example
docs/numerics.md         detailed numerical contract
docs/inference.md        evaluation semantics and callback examples
```

## Which configuration should I use?

The canonical YAML files are intended to make experiment identity reviewable.
They do not choose a model architecture or silently decide which linears are
eligible for FP4.

| Goal | Configuration | Important condition |
|---|---|---|
| Train with the proposed recipe | `proposed_b16_d50.yaml` | Call `begin_step` once per logical optimizer step. |
| Order-independent quantized validation | `inference/current_tensor_d1.yaml` | Recompute each activation operand's `amax`; recommended starting point. |
| Reproduce cold delayed-scale inference | `inference/training_replay_d50.yaml` | Fix batch size and exact evaluation order; advance once per complete forward. |
| Evaluate frozen activation scales | `inference/calibrated_frozen.yaml` | Use an immutable calibration stream disjoint from measurement data. |

These are stable package-resource names, so the same code works from a source
checkout and an installed wheel. List or inspect them with:

```python
from ue5m3_fp4.recipes import available_recipes, load_recipe_config

print(available_recipes())
current_tensor_config = load_recipe_config("inference/current_tensor_d1.yaml")
```

The training YAML has a strict typed loader through `UE5M3Recipe`. The three
inference YAMLs are declarative experiment manifests: `load_recipe_config`
returns their mappings for inspection, but does not instantiate a controller
or validate a runtime protocol. Select the corresponding
`activation_mode` explicitly when constructing
`FP4InferenceScalingController`; the explicit Python arguments are the runtime
source of truth in this release.

Training caches are process-local and are deliberately absent from a model
checkpoint. Consequently, D=50 inference starts cold; it never infers a cache
or optimizer step from the checkpoint. BF16 evaluation is a separate control
path and is not an FP4 result.

## Install

Python 3.11 or newer is required by package metadata. The current verification
record covers one CPU-reference environment rather than a complete
Python/PyTorch compatibility matrix; see the release checklist. Install the
editable package and development tools from a fresh virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Run the complete CPU reference suite and the checkpoint-reload example:

```bash
python -m pytest
python examples/tiny_mlp.py
python examples/tiny_mlp.py --activation-mode training_replay
python examples/tiny_mlp.py --activation-mode calibrated_frozen
```

The default example uses current-tensor D=1. The two additional commands run a
cold D=50 replay and a synthetic disjoint-calibration/frozen-scale lifecycle,
respectively. These examples validate control flow, not model quality.

The tests can also run directly from a checkout without installation:

```bash
PYTHONPATH=src python -m pytest
PYTHONPATH=src python examples/tiny_mlp.py
```

## Minimal training integration

Convert linears before constructing the optimizer, and make model coverage an
explicit architecture decision. For a toy model where every linear is meant
to use FP4:

```python
from ue5m3_fp4.nn import convert_linear_modules, select_all_linears
from ue5m3_fp4.recipe import UE5M3Recipe
from ue5m3_fp4.recipes import recipe_path
from ue5m3_fp4.scaling import TrainingScaleState

with recipe_path("proposed_b16_d50.yaml") as path:
    recipe = UE5M3Recipe.from_yaml(path)
scale_state = TrainingScaleState(recipe)

coverage = convert_linear_modules(
    model,
    recipe=recipe,
    scale_state=scale_state,
    selector=select_all_linears,
)
print([record.module_name for record in coverage])

optimizer = make_optimizer(model.parameters())
for optimizer_step, batch in enumerate(train_loader, start=1):
    scale_state.begin_step(optimizer_step)
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

For a language model, do not use `select_all_linears` without checking the
architecture. Supply an exact allowlist or a reviewed selector so the intended
language-model head and other exemptions remain in high precision. The generic
converter rejects `nn.Linear` subclasses and aliased module objects because
they require architecture-specific adapters.

The implementation applies stochastic rounding only to the upstream gradient
operand `dY` in the data-gradient and weight-gradient GEMMs. Forward
activations, weights, saved activations, and block scales use deterministic
ties-to-even rounding. The target 2,048 override is limited to `dY` in the
weight-gradient GEMM for `mixer.down_proj` in Nemotron layers 44--51; all other
operands use 448.

## Save and reload checkpoints

Save learned parameters normally. `TrainingScaleState.state_dict()` is empty
by design because the delayed-amax cache is runtime numerical state, not a
learned parameter:

```python
checkpoint = model.state_dict()
assert scale_state.state_dict() == {}
```

For evaluation, load master weights into a fresh model, convert the same
reviewed linear set, call `eval()`, and then create fresh inference state. The
controller enforces this order:

```python
from ue5m3_fp4.scaling import FP4InferenceScalingController

evaluation_model.load_state_dict(checkpoint)
convert_linear_modules(
    evaluation_model,
    recipe=recipe,
    selector=select_all_linears,  # replace with the reviewed model selector
)
evaluation_model.eval()

controller = FP4InferenceScalingController(
    evaluation_model,
    activation_mode="current_tensor",
    checkpoint_identity={"id": "checkpoint-step-30000"},
)
controller.reset_after_checkpoint_load()
controller.calibrate_and_freeze_weights()
controller.begin_measurement()

with torch.inference_mode():
    output = evaluation_model(input_ids)

result_provenance = controller.provenance()
```

The weights' global `amax` values are sampled once after load and frozen,
because the learned weights do not change during evaluation. The forward
activations still undergo FP4 fake quantization according to the selected
policy.

## Inference activation strategies

### 1. Current-tensor scaling (D=1)

Use `activation_mode="current_tensor"` as in the example above. Every forward
activation is scaled from its own current `amax`. This avoids order-dependent
cache reuse and is the clearest baseline for checkpoint comparison.

### 2. Cold periodic replay (D=50)

Use this only when the ordered inference work unit is part of the experiment.
Advance the controller exactly once immediately before every complete model
forward—not once per layer or token:

```python
controller = FP4InferenceScalingController(
    evaluation_model,
    activation_mode="training_replay",
    checkpoint_identity={"id": "checkpoint-step-30000"},
    replay_work_unit={"kind": "fixed_forward_batch", "size": 1},
)
controller.reset_after_checkpoint_load()
controller.calibrate_and_freeze_weights()
controller.begin_measurement(
    evaluation_order={"order": "validation manifest order", "seed": None}
)

with torch.inference_mode():
    for input_ids in ordered_batches:
        controller.advance_training_replay_work_unit(
            input_ids,
            effective_token_count=input_ids.numel(),
        )
        output = evaluation_model(input_ids)
```

The first work unit refreshes each activation cache; work units 2--50 reuse it;
unit 51 refreshes it again. Results can change if input order or batch size
changes, so both are recorded in provenance.

### 3. Disjoint calibration, then frozen scales

Collect a per-linear maximum over an identified calibration stream, freeze it,
and only then begin measurement:

```python
controller = FP4InferenceScalingController(
    evaluation_model,
    activation_mode="calibrated_frozen",
    checkpoint_identity={"id": "checkpoint-step-30000"},
)
controller.reset_after_checkpoint_load()
controller.calibrate_and_freeze_weights()
controller.begin_activation_calibration(
    {"manifest_sha256": calibration_manifest_sha256}
)

with torch.inference_mode():
    for input_ids in ordered_calibration_batches:
        controller.record_activation_calibration_batch(input_ids)
        evaluation_model(input_ids)

controller.freeze_activation_scales()
controller.begin_measurement()

with torch.inference_mode():
    for input_ids in validation_batches:
        output = evaluation_model(input_ids)
```

The calibration and validation streams must be disjoint. Persist the
calibration-manifest identity alongside the result rather than relying on a
human-readable dataset name. The controller records calibration inputs but
does not automatically prove that a separately supplied validation set is
disjoint; the evaluator or experiment manifest must enforce that check.

## Validation loss

`evaluate_validation` computes exact next-token negative log-likelihood from
local, already-tokenized tensors or safetensors files. Each row must contain
`sequence_length + 1` tokens: the evaluator forwards `tokens[:-1]` and compares
the logits with `tokens[1:]`. It retains per-sequence sums and content hashes
for paired analysis.

```python
from ue5m3_fp4.eval import evaluate_validation

validation = evaluate_validation(
    evaluation_model,
    ["validation-shard-000.safetensors"],
    checkpoint_id="checkpoint-step-30000",
    device="cuda",
    batch_size=1,
)
validation["model"] = controller.provenance()
```

For the BF16 control, do not convert the model to `UE5M3Linear`. Attach the
explicit control-path record instead:

```python
from ue5m3_fp4.scaling.inference import learned_weight_bf16_numeric_path

bf16_validation["model"] = learned_weight_bf16_numeric_path()
```

For D=50 replay, use `before_forward_callback` to call
`advance_training_replay_work_unit`; the complete pattern is in
[`docs/inference.md`](docs/inference.md). Never relabel a BF16 checkpoint
evaluation as quantized inference: `controller.provenance()` records whether
FP4 quantization was actually applied and identifies the numerical path.

## Reproducing the reference implementation

There are three different reproduction targets, and they should not be
conflated:

1. **Portable numerical reference.** Install this package, run the unit tests,
   and run `examples/tiny_mlp.py`. This exercises block quantization, delayed
   training state, backward quantization, checkpoint reload, and current-tensor
   FP4 inference on synthetic data.
2. **Model integration.** Add an architecture adapter with an explicit eligible
   linear allowlist, use the packaged `proposed_b16_d50.yaml`, and retain the emitted
   conversion coverage and scale-state reports with each run. Evaluate BF16
   and at least current-tensor FP4 from the same learned checkpoint.
3. **Paper-scale experiment.** This additionally requires the exact
   Nemotron-H/TorchTitan integration, tokenizer and immutable data manifests,
   distributed checkpoint conversion, probe-matched GEMM model, runtime
   versions, and launch configuration. Those assets are not yet included, so
   this repository alone cannot reproduce the paper's 8B loss or throughput
   numbers.

For a reviewable run, record at minimum:

- Git commit and recipe-file SHA-256;
- Python, PyTorch, CUDA, and device versions;
- exact converted-module names;
- checkpoint identity and content hash;
- random seeds and deterministic settings;
- training step or ordered inference work-unit definition;
- validation/calibration manifest identities; and
- `TrainingScaleState.report()` or `controller.provenance()` output.

## Development and release checks

Run the pinned formatter/linter, tests, and package build before proposing a
change:

```bash
ruff check .
ruff format --check .
python -m pytest
python -m build
```

Then install the wheel into a clean environment and rerun the suite and
example. Before publishing a tagged release, repeat the full-history secret
scan and the operational checks in
[`docs/release_checklist.md`](docs/release_checklist.md). CUDA qualification
and paper-scale reproduction are separate future integration targets, not
claims made by this portable decoded-Torch reference. The exact checks run on
the public-alpha CPU reference are recorded in
[`docs/verification.md`](docs/verification.md).

## Numerical and result interpretation

This package genuinely rounds selected operands to values representable by
E2M1 with UE5M3 block scales. It then decodes those operands and uses the
runtime PyTorch matmul path. Provenance records that path and the relevant
PyTorch matmul settings. It is therefore suitable for checking operand
quantization and scale-lifecycle behavior, but not for claiming native UE5M3
throughput or the paper's probe-matched accumulation behavior.

Native Transformer Engine NVFP4 is a separate hardware/software path using
E2M1 payloads with E4M3 block scales. Likewise, loading a checkpoint trained
with FP4 and evaluating its learned weights in BF16 is a useful control, but it
is not quantized inference.

## Provenance, citation, and license

The extraction is based on `gc-training` commit
`99a96f2a345ab4a9d37904cfdcdf93777458106d`. It uses fresh Git history rather
than publishing monorepo history. File-level and third-party provenance is
recorded in [`NOTICE`](NOTICE) and
[`SOURCE_PROVENANCE.md`](SOURCE_PROVENANCE.md).

Citation metadata for both the software and accompanying report is provided in
[`CITATION.cff`](CITATION.cff). The preferred citation currently identifies the
report by title and authors; its arXiv identifier can be added after assignment.
For numerical comparisons, also record the exact repository commit used.

The code is provided under the Apache License 2.0. Graphcore approved this
standalone extraction for public release under that license with the current
[`NOTICE`](NOTICE).

Contributions should follow [`CONTRIBUTING.md`](CONTRIBUTING.md), especially
the numerical-change and result-provenance requirements.

## Contact

For correspondence about the project, email `robert.stats.hu@gmail.com`.
