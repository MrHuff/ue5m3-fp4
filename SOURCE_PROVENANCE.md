# Source provenance

The standalone tree is an allowlist-based extraction from `gc-training` commit
`99a96f2a345ab4a9d37904cfdcdf93777458106d`.  No Git objects, environment files,
credentials, internal job manifests, or private storage locations are copied.

| Standalone component | Monorepo source | Extraction treatment |
|---|---|---|
| FP4/UE5M3 formats and rounding | `low_bits_training/quantization/triton_quantization_compat.py` and `fused_quant_triton_v2.py` | Reduced to E2M1, unsigned E5M3, ties-to-even, and exact 8-bit-midpoint `StochasticFast`; implemented as a portable Torch reference. |
| Training scale lifecycle | delayed-amax logic in `fused_quant_triton_v2.py` | Replaced the implicit environment step with an explicit `begin_step(step)` API while preserving periodic sample-and-hold semantics. |
| Post-load inference scaling | `low_bits_training/evaluation/fp4_inference_scaling.py` | Adapted as a generic Torch-only controller; cloud and checkpoint download code excluded. |
| FP4 linear and conversion | `fused_quant_triton_v2.py` and `mxfp_custom.py` | Reduced to the proposed recipe with explicit typed configuration and a generic `nn.Linear` converter. |

The extraction snapshot below freezes the initial source and output hashes;
future changes remain visible in the standalone Git history.

## Extraction snapshot

The source snapshot was frozen at the commit above before extraction. SHA-256
digests make the review boundary explicit; extracted files are adaptations and
are not expected to be byte-identical to their monorepo sources.

| File | SHA-256 |
|---|---|
| Source `fused_quant_triton_v2.py` | `66e3e8f4c29f5924dfa905f9f2b5fec2e651ca5bca16eb4c11a0ca7d4cfa05c2` |
| Source `triton_quantization_compat.py` | `8efa00bfa4195cbd54cb7212261ef8904faee0651ab87b92e3f20b3be3866d2c` |
| Source `mxfp_custom.py` | `e9de5525bb232f49834d806eb3459cedaa6ea6c9d62d1ed537b10767f0601b78` |
| Source `fp4_inference_scaling.py` | `50c520c22d8461b4c48e859c337eb7f81a7f4fead71224189a1d1587d3701632` |
| Source `validation_loss.py` | `65e77ddbeab0b03bdc4129b7820a897dab0b7b12bf4c7c67e123ad2be817edd0` |
| Extracted `formats.py` | `45b689f975a83421464b37506108e8cd80f55a40ff379de82d170162acf2127c` |
| Extracted `recipe.py` | `8da6b42535347e570ea5c0b48b11c10d01cba601b03ceb5671e309aa16121ac0` |
| Extracted `scaling/training.py` | `d22de7ad4c25b562f003eaf06c0de5c0ab78b5e0a14e4fd3042b5e2b9e8aaaca` |
| Extracted `scaling/inference.py` | `da7577dfa069fd242b76ca8bd09e57b1fc6154acfc0952e3351ef840c551094e` |
| Extracted `nn/linear.py` | `ba79c3e4231bdd070579659eb0b16173f3e9ccf3a1070352b5613e107391bf43` |
| Extracted `nn/convert.py` | `db9a68e9f3c310082c11e9af6bce6ff23ed22b0060440bafc1d4cd2537004f6c` |
| Extracted `eval/validation.py` | `30b16a5ef87e933dfc4aae2491a96f25cb6c12ba1e8330072fd8a7ac3eb5c032` |
