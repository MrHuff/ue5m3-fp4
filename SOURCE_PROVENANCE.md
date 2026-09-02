# Source provenance

This repository is an allowlist-based public extraction and adaptation of the
Graphcore-approved UE5M3 research implementation. The committed source baseline
is identified by revision `99a96f2a345ab4a9d37904cfdcdf93777458106d`; the
source files used from that snapshot are additionally identified by SHA-256
below. This is important because the original research repository is not a
public dependency of this release.

No source-control objects from the original repository, credentials, private
storage locations, model checkpoints, token data, internal job manifests, or
cluster-specific orchestration are included. The public TorchTitan path replaces
the original launcher and storage glue. It preserves the released numerical
method and reported model/configuration choices, but its public data ordering has
a new identity and is not a byte-identical replay of a historical training run.

## Source snapshot

The released modules are adaptations rather than byte-for-byte copies. The
source hashes define the reviewed extraction boundary.

| Source input | SHA-256 |
|---|---|
| FP4 quantization, autograd, and GEMM implementation | `66e3e8f4c29f5924dfa905f9f2b5fec2e651ca5bca16eb4c11a0ca7d4cfa05c2` |
| Floating-point compatibility implementation | `8efa00bfa4195cbd54cb7212261ef8904faee0651ab87b92e3f20b3be3866d2c` |
| Original FP4 module-selection wrapper | `e9de5525bb232f49834d806eb3459cedaa6ea6c9d62d1ed537b10767f0601b78` |
| Nemotron-H training wrapper | `236ea6909cbba2fc7fba5016fa10a5d5fe44cf36b5f8972e47fbfc206b7670cb` |
| Checkpoint conversion implementation | `f904236ad8440f6187dfd203ecec368f77dda998eba9ccbb71b366298818ba78` |
| Post-load FP4 inference scaling | `50c520c22d8461b4c48e859c337eb7f81a7f4fead71224189a1d1587d3701632` |
| Post-load FP4 model conversion | `3d39d9e021c89dd6538ddd8fdfcb6ef99bf002ce72ae4adc3e06e30ba8af091c` |
| Quantized-inference protocol definitions | `c45ecb11ad0dc990c61392cd4e4acd89af8a874ae8ea38c62b779594f2c02cd5` |
| Calibration-data implementation | `9773e83950f96be754acfc26d0665b52c878dbd6f2a7addf5eec420a82365c2b` |
| Quantized-inference pilot implementation | `a2db99185d3fbe3433c0f004c1e5d4398a03ff965cd5d5e5bdff4d0d557abfa0` |
| Held-out validation evaluator | `65e77ddbeab0b03bdc4129b7820a897dab0b7b12bf4c7c67e123ad2be817edd0` |
| Validation sweep implementation | `4964dbf5fc1bcc9ee9b6b68eefdde7c2534071f8425c09c006934cd24b9288b8` |
| Grouped validation implementation | `b5c48e3bfcc35a06f6abaa1e1cc16b15a08e9a508145bdea420b6c987d7f6204` |
| Runtime-wheel provenance implementation | `1a80cf7183d8198370b94d03656ca990990eaaa50bf4355443e24c4bcb673a89` |

## Extraction treatment

| Public component | Treatment |
|---|---|
| E2M1 and UE5M3 formats | Reduced to the formats and rounding modes used in the paper, with a portable Torch oracle and exhaustive/sampled representation tests. |
| Proposed Triton backend | Split into explicit quantization and GEMM modules. It retains encoded operands, block-16/block-32 scaling, stochastic midpoint rounding, K=64 issue grouping, round-toward-zero cross-group accumulation, and the final `1/1024` snap. Focused CUDA differentials compare it with the source snapshot. |
| Training integration | Re-expressed through pinned public TorchTitan. Delayed scaling advances once per optimizer step, synchronizes each current amax across the default distributed group, and uses periodic sample-and-hold rather than a rolling maximum. |
| Nemotron-H integration | Rebuilt against pinned public Hugging Face remote code. A hash-locked patch prevents the Mamba fused training path from bypassing converted `out_proj` modules. Model shape, module inventory, precision placement, and state-dict roots fail closed. |
| Comparator paths | Native Transformer Engine NVFP4 and the UE5M3-with-Transformer-Engine-settings comparator are explicit converters. Native execution requires the exact pinned Transformer Engine runtime and supported Blackwell hardware; no BF16/software fallback is allowed. |
| Evaluation | Adapted to local safetensors/Hugging Face checkpoints, explicit post-load FP4 scale lifecycles, ordered data identities, FP32 cross entropy, FP64 accumulation, and storage-neutral provenance. |
| OLMES | Adapted to a pinned public OLMES ancestor with a fail-closed compatibility hook that restores the relevant historical filename behavior. Public task reconstruction creates a new request identity; byte-identical request replay is enabled only for caller-supplied artifacts matching the recorded immutable hashes. |
| Paper evidence | Sanitized collector tables, diagnostic summaries, and value-distribution histograms retain numerical values and hashes while removing private storage and job metadata. They are archived evidence, not outputs of a public rerun. |

## External immutable inputs

| Dependency or asset | Public location and identity |
|---|---|
| TorchTitan | `https://github.com/pytorch/torchtitan.git`, commit `e37f83f58b35fdbceed9a5916b3490c16247ac9c` |
| Transformer Engine | `https://github.com/NVIDIA/TransformerEngine.git`, commit `01aef4fc721bd12fd09cd56d53a314aee1b953d6`, expected package version `2.16.0.dev0+01aef4fc` |
| OLMES | Public ancestor `https://github.com/allenai/olmes.git` at commit `8e2743734066b073c5d8498d1b8220f67a21a2d6`, tree `991940b9a2b37f8491ff29d1d22487b209fe750f`; the released runtime restores the historical descendant's colon-free request/result filenames |
| Container base | `nvcr.io/nvidia/pytorch:25.10-py3`, Linux arm64 manifest digest `sha256:5c8302e4628ac326c412368675cc462b8aee2f96326a3bf817304a83816f179a` (multi-architecture index `sha256:42263b2424fc237b34c4fc4a91c30d603c57eed36e37d31ff6d9a4f1f801edee`) |
| Nemotron-H configuration/tokenizer/code | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base`, revision `78dc93a79e2533922ac8ad2c16f79b7fb747970d`; model source changes from SHA-256 `8fed3b30c627bc5c58f1f17f5941fa2641d1ea69bf52c40bac31ec0dd67dd4a9` to `9498e7b4b28592fc03d9b00e74ae5484672a842fd8e322b69eabe1edfa14689a` under the public dispatch patch |
| Public OLMo Mix reconstruction | `allenai/olmo-mix-1124`, revision `8162bd79c6dc4fea470506531a8d791badc06b4b` |
| OLMES auxiliary sources | AI2 OLMo commit `090253dac6688f2532509daa7aa2eb5fae50e956`; AlpacaEval commit `db85f8065408b842100436a45f56c65d3a0dd6a6` |

The three Git dependencies are recorded as Git submodules, so their commit
identities are also part of the standalone repository tree. The historical
OLMES evaluation used descendant revision
`3d80ebb0f08706a5d2dd3fb0be72100735b5f5c6`, tree
`6403093f39b09e3dd6980bee7d60863a7714de8f`, whose remote is not an anonymous
public dependency. For the three reported suites, its changes relative to the
pinned public ancestor affect unselected HELMET/judge paths, dependency
declarations fixed by the released lock, inactive default chat handling, and
colon-free storage names. The public hook restores the last behavior and
records both identities. The container and Python locks fix the released
aarch64/CPython 3.12 runtime boundary.

## Released-file snapshot

These hashes cover the paper-critical release files. `tests/test_provenance.py`
checks every row in this section.

| Released file | SHA-256 |
|---|---|
| `src/ue5m3_fp4/formats.py` | `2fd4ccbdd0e98cf8eae3f05f78ec0e85fad6c864ffef77d99163d5adbc240c6b` |
| `src/ue5m3_fp4/recipe.py` | `b7e4062518a4db3b53af410c22ac5a9bd94c80ecd485d2ae7a11cca86d2cedce` |
| `src/ue5m3_fp4/scaling/training.py` | `258170700d7aeb8a411450949e9c70347e1dd894099f5b602202ea734cc6814b` |
| `src/ue5m3_fp4/scaling/inference.py` | `2e3352ad45c1f016305dbd254c0c5e53c49aaf32dafe04aedeb9938af698a026` |
| `src/ue5m3_fp4/nn/linear.py` | `5db120576818f773d8213bab43e89d6896bb45e93c1a3fd6a689d9b22c00e650` |
| `src/ue5m3_fp4/nn/convert.py` | `ce057895aff3d3118faf3731aa90b91541de9b33ba1d78c42c9ad67424d74ae4` |
| `src/ue5m3_fp4/backends/triton/_rounding.py` | `abb6b5e3ca0411ece66b35a920fe62a3fb9953593c54dbc63b20fb04df2a60b0` |
| `src/ue5m3_fp4/backends/triton/quantization.py` | `0921ad475801c903378b7d6d780ed66c035859926691af116c46101dd591a388` |
| `src/ue5m3_fp4/backends/triton/gemm.py` | `8621454eea0c45c0b3e2421c0bd303930dd9bc2ca9c71fd510ecae1eebf76a7c` |
| `src/ue5m3_fp4/backends/triton/api.py` | `67306b9915119585722a53cb0386ab458025f7b11b2599f1eff363641a8ed396` |
| `src/ue5m3_fp4/integrations/torchtitan/nemotron_h.py` | `06aaab63d1a7f87ef6c18972710d6761aaa56ceba04898978a3b9dc4cdf7a881` |
| `src/ue5m3_fp4/integrations/torchtitan/remote_code.py` | `74dd5b4d29b767cc91151d07f40af6c7a85e1943a004651bd2fbdde0ae4d8d55` |
| `src/ue5m3_fp4/integrations/torchtitan/selection.py` | `a36ba7616f25e2460644a036679e415fd1e9cf1a3100af6be6e2adea139f56a3` |
| `src/ue5m3_fp4/integrations/torchtitan/comparators.py` | `47c1a4d466fb69aca69606e662a185ff347b3f62b26ed4b3ddd269f38799df21` |
| `src/ue5m3_fp4/integrations/torchtitan/linear_backend.py` | `25ce14471f57ad8543f316b76bd87193443ea2c0a024fa44d8376d5e879e9541` |
| `src/ue5m3_fp4/integrations/torchtitan/registration.py` | `f90d10e98fdc1837fe4cfbddf421afbec9aaab01669dbc90980077eb11b9c9c5` |
| `src/ue5m3_fp4/integrations/torchtitan/config.py` | `a4b9420c7a48ca21424d2cdc255c5950c0e9e11f3aa6966be23d3e59027d80fb` |
| `src/ue5m3_fp4/integrations/torchtitan/data.py` | `2e5e8cf679b5d719d145683e321407671414d1f00c4757dd7d186e47afcdf9d1` |
| `src/ue5m3_fp4/integrations/torchtitan/state_dict.py` | `29cd3be3af856fd22c721491e66e45bd795bb42d1093ead5c59441458066da15` |
| `src/ue5m3_fp4/integrations/torchtitan/trainer.py` | `6f2171b6a34488d2ec5f4a6d85dcfc4cc15585403a172c482f42409c5cd2c08c` |
| `src/ue5m3_fp4/checkpoint.py` | `711ebce1d9c6b1520eed92ec7e8f3777100a788bc48f4a19c42cafaea57aa7f8` |
| `src/ue5m3_fp4/cli/evaluate.py` | `9f5fd344a03ff095388c935e7e99595aa85fa786c3448ccd8c915d93b6ef4915` |
| `src/ue5m3_fp4/olmes_runtime.py` | `e74d9e5c4394c208c43f81b1bb23d7c21fed2c84802b814469357ba8caa75de7` |
| `reproduce/configs/nemotron_h_8b_bf16.toml` | `cbdac3ca285a413a433c13bd3498e368b2ea978a76c1aa57164b0c18e2878572` |
| `reproduce/configs/nemotron_h_8b_nvfp4_no_rht_all_linears.toml` | `f2b6d24869d4dd8493c75491a4668ad9b2044871707b0861bde894dd7a19def8` |
| `reproduce/configs/nemotron_h_8b_nvfp4_te.toml` | `93ef0c016824ef2ae2e36d286fe426dad6876894f0f15cfd82e95e8a2ddd3c8c` |
| `reproduce/configs/nemotron_h_8b_ue5m3_b16.toml` | `b0d5e04c3405a778373280ed52d36b2936a09454d6a94f96a9a98524db0a2c56` |
| `reproduce/configs/nemotron_h_8b_ue5m3_b32.toml` | `98adad4e360a279aa0768cca15e73b246269e8ec524a34c158553026e489139f` |
| `reproduce/configs/nemotron_h_8b_ue5m3_te_settings.toml` | `96ecba35525ece7b099cfac2cc2d8bbe29fcb9e07b960b16ea079299bec72434` |
| `reproduce/configs/nemotron_h_8b_ue5m3_torch_control.toml` | `92b9d0e45b63c9172c777dbd515f1bf40c0f93678e47c96b3b5040b5f0b885be` |
| `reproduce/manifests/data_preparation.yaml` | `96ca201083d8bbd400fe468b2dab53a639c2f6835935b9d0e51c6c7230daee88` |
| `reproduce/manifests/reported_experiments.yaml` | `d10f6255fda08b3a519fb1185fc203a01eb1df4db78b47bd2d9a8e36588d7c14` |
| `reproduce/manifests/validation.yaml` | `ae1871f482a648f635b953f2042b8a8595e495f40fc31d95ace04b6b6a62d066` |
| `reproduce/manifests/olmes.yaml` | `09fd965022cb9167c9dcbf23aa0274200d8cab022697026f5ed6395e9d4156b9` |
| `reproduce/reference_results/provenance.json` | `37f14e317dc6a0fb12ef1b8f229e7a198348f1abedd7fd827ecf89e1e7fef326` |
| `reproduce/reference_results/generated/artifact_manifest.json` | `b77b05824f5f49a424face92a21a41f60ef9add1658b01c98c639cb75581754a` |
| `reproduce/diagnostics/archived/report_summary.json` | `b92b68b96189f689655e811e5e184adcc06f008d2fd2c67a4c6bf4b0e2f4ee8e` |
| `reproduce/scale_target/archived/manifest.json` | `987d747ee58a511c1c52caaf92651095ec80b7efcc98450ed9e7cdce9034aa0e` |
| `reproduce/scale_target/historical_350m_provenance.json` | `701309318411675b8a4c331bd7756a44e485da534aa7e12c5ac775772ebef223` |
| `reproduce/Dockerfile` | `c10b6a8549421a7ebd952a4e64f3daa3b91eb735a2666af1efaf9d15aa97aaa2` |
| `reproduce/environment/requirements-ngc-25.10.lock` | `ecc6ebeb68655bf9f236ac1744db585bb7c3dc6b56bd497cf47e2c2c838205fc` |
| `reproduce/environment/compiled-requirements-ngc-25.10.lock` | `da1d0ecceb06b6c6d0b57da57dc70c84aa6b5cb702d73ba91256b1e5afc041b6` |
| `reproduce/environment/olmes-cp312-linux-aarch64.lock` | `ccb078f4c1c87766931e2dafa0c6002640c2d35d5c7e41965ea3531f3bc16da5` |
| `src/ue5m3_fp4/recipes/__init__.py` | `ebd70d60af4fe00296cb4ad35dfbab71a60f31d325a72c0c581ced43ac3a10e5` |
| `src/ue5m3_fp4/recipes/proposed_b16_d50.yaml` | `5007da1d7955188426c955b793eef03a0228581a5669bd24013f10eded9e4ac3` |
| `src/ue5m3_fp4/recipes/inference/current_tensor_d1.yaml` | `d3a865047eb7ff74baaa675320bc4777e24495586dc9bd69fe430345465570b4` |
| `src/ue5m3_fp4/recipes/inference/training_replay_d50.yaml` | `b4708c5330227ee436153b52ab21105fd75e2706abe315a58645605d609b0c6d` |
| `src/ue5m3_fp4/recipes/inference/calibrated_frozen.yaml` | `be414e8bca60d1bfb18307f911fcacd5fc13aaaeb188b43fd2e8c8f054862266` |

The standalone Git commit and tagged release archive remain the primary
identity for the complete tree. This table provides a human-reviewable boundary
for the implementation and evidence most directly tied to the paper.
