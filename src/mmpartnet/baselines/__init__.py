"""Baseline classifiers — simplest multimodal models that validate the data pipeline.

Anything fancier (FiLM, cross-attention) must beat these or the fancy is doing nothing.
"""
from .early_fusion import (
    EarlyFusion,
    EarlyFusionDataset,
    ProteinEmbeddings,
    build_gene_to_idx,
    one_hot_meanpool,
    train_one_epoch,
    evaluate,
    protein_shuffle_mapping,
)

__all__ = [
    "EarlyFusion",
    "EarlyFusionDataset",
    "ProteinEmbeddings",
    "build_gene_to_idx",
    "one_hot_meanpool",
    "train_one_epoch",
    "evaluate",
    "protein_shuffle_mapping",
]
