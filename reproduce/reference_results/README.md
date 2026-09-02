# Sanitized reference results

These tables are the reviewed numerical artifacts used by the UE5M3 report,
with private checkpoint and result-storage locations removed. Values, labels,
confidence-interval fields, and result-content hashes are otherwise copied
without numerical transformation.

- `validation_metrics.csv`: 84 held-out-loss points (seven trajectories at 12
  checkpoints).
- `validation_comparisons.csv`: 144 same-step comparisons reported by the
  collector.
- `olmes_aggregate_scores.csv`: the complete seven-configuration by
  three-benchmark aggregate matrix.
- `olmes_leaf_task_metrics.csv`: all 146 leaf-task metrics for each of the
  seven step-30,000 configurations (1,022 rows).
- `olmes_paired_differences.csv`: 36 reviewed paired aggregate differences.
- `provenance.json`: source/output SHA-256 hashes, removed columns, exact
  inventories, and the single-seed limitation.

These are reference outputs, not model checkpoints or an OLMES request bundle.
The sequence bootstrap intervals in the tables quantify held-out-example
variation; they do not estimate training run-to-run variability. Every training
configuration has one independent seed.

To regenerate from a checkout containing the reviewed paper data:

```bash
python reproduce/reference_results/generate_reference_results.py \
  /path/to/paper/reports/ue5m3_fp4_training/data
```

The generator fails if row counts, the 7x3 OLMES matrix, expected private
columns, or sanitized-output checks differ from the reviewed source.

Render the BF16-referenced validation-loss/percent-loss curves, the common-axis
OLMES comparison, and compact final tables with:

```bash
python reproduce/reference_results/render_reference_artifacts.py
```

This writes PDF/PNG figures, CSV tables, and a hash manifest below `generated/`.
The validation percentage uses
`100 * (BF16 NLL - candidate NLL) / BF16 NLL`, so positive values mean lower
validation loss than the same-step BF16 reference.
