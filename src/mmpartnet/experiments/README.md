# `mmpartnet.experiments` — reusable experiment orchestration

Experiment modules keep the training/evaluation logic inside the package, while
`scripts/` stays as a thin command-line layer. This makes branch-specific
workflows easier to merge into a shared pipeline without duplicating loaders,
model construction, metrics, and checkpoint handling across entry points.

| file | role |
|------|------|
| `film_multitask.py` | FiLM multitask/profile-only/binary-only training and evaluation workflow |
| `recover_demo_finetune.py` | demo recovery fine-tuning workflow from the main branch |
| `recover_demo_profile.py` | demo profile recovery workflow from the main branch |

## FiLM branch note

`film_multitask.py` is the reusable implementation behind:

```text
scripts/train_film_profile.py
scripts/eval_film_multitask.py
```

It builds flattened RNA-window/RBP-cell batches from `mmpartnet.data.multimodal`,
feeds frozen PARNET RNA features into `ProteinCellFiLMProfileHead`, and records
profile Pearson plus binary binding metrics for the FiLM baseline and its
single-task ablations.
