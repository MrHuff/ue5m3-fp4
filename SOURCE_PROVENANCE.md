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

The snapshot below records the monorepo inputs and the current standalone
release-candidate outputs. Refresh the extracted-file digests whenever one of
these files changes; subsequent changes remain visible in the standalone Git
history.

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
| Extracted `formats.py` | `2fd4ccbdd0e98cf8eae3f05f78ec0e85fad6c864ffef77d99163d5adbc240c6b` |
| Extracted `recipe.py` | `3a633cb936759a2c360da4d34e7803b5a29eb1c324d858c5c7f2cac9cf765fde` |
| Extracted `scaling/training.py` | `d22de7ad4c25b562f003eaf06c0de5c0ab78b5e0a14e4fd3042b5e2b9e8aaaca` |
| Extracted `scaling/inference.py` | `24ef2d0842e3ff234b051bcfe09bbdb52c96737e44ca605f83d891bd902b3caf` |
| Extracted `nn/linear.py` | `7bd7fc393fecd02a4fa915432eccdc021393d966ee662169f4f240d9ac829c7f` |
| Extracted `nn/convert.py` | `1b20982830fe003cdb381f4609b21427443f3bad5d0903b35218dffe765a1311` |
| Extracted `eval/validation.py` | `1aa246aed38c3a05a5fa6957f8befdf00c54f9cfb3f23ff996835b27b3e950b4` |
| Packaged recipe-resource API | `ebd70d60af4fe00296cb4ad35dfbab71a60f31d325a72c0c581ced43ac3a10e5` |
| Packaged `proposed_b16_d50.yaml` | `5007da1d7955188426c955b793eef03a0228581a5669bd24013f10eded9e4ac3` |
| Packaged `current_tensor_d1.yaml` | `d3a865047eb7ff74baaa675320bc4777e24495586dc9bd69fe430345465570b4` |
| Packaged `training_replay_d50.yaml` | `b4708c5330227ee436153b52ab21105fd75e2706abe315a58645605d609b0c6d` |
| Packaged `calibrated_frozen.yaml` | `be414e8bca60d1bfb18307f911fcacd5fc13aaaeb188b43fd2e8c8f054862266` |
