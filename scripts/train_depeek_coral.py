"""De-peek CORAL trainer (three-way family split): TRAIN + EARLY-STOP-VAL + untouched TEST.

Patch of the CORAL fork's scripts/train.py for a NO-TEST-PEEK protocol:
  - reads fold_dir/{train,val,test}.csv (VAL = family-disjoint early-stop set; TEST = held families, never
    used for selection);
  - each epoch computes train/val/TEST metrics -> {train,val,test}_metrics.csv;
  - saves the best-VAL-F1 PEFT adapter to <out>/best_adapter and the final to <out>/last_adapter (reusable
    weights for predict.py per-pair scoring + attentions);
  - writes selection.json {best_epoch, best_val_f1}.
De-peek headline = TEST metrics AT the best-VAL epoch (honest), reported next to last-epoch and best-on-TEST.
Deploy: copy into the CORAL fork's scripts/ (as train_depeek.py); run from the coral repo root on a CUDA GPU.
"""
import argparse, json, os, random, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from torch.amp import autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from coral.data import CoralDataCollator, seed_worker, specificity_score, tokenize_function
from coral.model import CoralConfig, CoralPretraining


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--data-root", default="Data/datasets")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--rna-encoder", default="./DNABERT-2-117M")
    p.add_argument("--protein-encoder", default="facebook/esm2_t30_150M_UR50D")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--warmup-pct", type=float, default=0.05)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--mlm-scaling", type=float, default=1.0)
    p.add_argument("--num-threads", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_loader(df, tok_kwargs, collator, bs, g, threads, shuffle):
    ds = Dataset.from_pandas(df)
    tds = ds.map(tokenize_function, batched=True, num_proc=threads,
                 remove_columns=["RNA_seqs", "Prot_seqs"], fn_kwargs=tok_kwargs)
    return DataLoader(tds, shuffle=shuffle, batch_size=bs, collate_fn=collator,
                      worker_init_fn=seed_worker, generator=g)


def run_eval(peft_model, loader, device):
    peft_model.eval(); preds, labels = [], []
    for batch in loader:
        matched = batch["labels"].to(device)
        rna = {"input_ids": batch["rna_input_ids"].to(device),
               "attention_mask": batch["rna_attention_mask"].to(device),
               "labels_for_MLM": batch["rna_labels_for_MLM"].to(device)}
        prot = {"input_ids": batch["protein_input_ids"].to(device),
                "attention_mask": batch["protein_attention_mask"].to(device),
                "labels_for_MLM": batch["protein_labels_for_MLM"].to(device)}
        with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
            _, _, predictions, _ = peft_model(rna, prot, matched, task_mask_lm=True, mlm_scaling_factor=1.0)
        preds.extend(predictions.cpu().tolist()); labels.extend(matched.cpu().tolist())
    return labels, preds


def write_metrics(path, fold, epoch, y, p):
    if not path.exists():
        path.write_text("Accuracy,F1,MCC,Recall,Precision,Specificity,Fold,Epoch\n")
    with path.open("a") as f:
        f.write(f"{accuracy_score(y,p)},{f1_score(y,p)},{matthews_corrcoef(y,p)},{recall_score(y,p)},"
                f"{precision_score(y,p)},{specificity_score(y,p)},{fold},{epoch}\n")


def main():
    a = parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True
    g = torch.Generator(); g.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out = Path(a.output_dir) / a.split / str(a.fold); out.mkdir(parents=True, exist_ok=True)
    fold_dir = Path(a.data_root) / a.split / f"fold_{a.fold}"
    cols = ["RNA_seqs", "Prot_seqs", "labels"]
    train_df = pd.read_csv(fold_dir / "train.csv")[cols]
    val_df = pd.read_csv(fold_dir / "val.csv")[cols]
    test_df = pd.read_csv(fold_dir / "test.csv")[cols]

    rna_tok = AutoTokenizer.from_pretrained(a.rna_encoder, trust_remote_code=True)
    prot_tok = AutoTokenizer.from_pretrained(a.protein_encoder, trust_remote_code=True)
    tk = {"rna_tokenizer": rna_tok, "protein_tokenizer": prot_tok}
    collator = CoralDataCollator(rna_tokenizer=rna_tok, protein_tokenizer=prot_tok, mlm_probability=0.15)
    train_loader = make_loader(train_df, tk, collator, a.batch_size, g, a.num_threads, True)
    val_loader = make_loader(val_df, tk, collator, a.batch_size, g, a.num_threads, False)
    test_loader = make_loader(test_df, tk, collator, a.batch_size, g, a.num_threads, False)
    print(f"batches train/val/test = {len(train_loader)}/{len(val_loader)}/{len(test_loader)}", flush=True)

    model = CoralPretraining(config=CoralConfig(rna_encoder_checkpoint=a.rna_encoder,
                                                protein_encoder_checkpoint=a.protein_encoder)).to(device)
    target, save = [], []
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.Linear, torch.nn.Embedding)):
            if name.startswith("bert.protein_encoder") or name.startswith("bert.rna_encoder"): target.append(name)
            elif any(name.startswith(p) for p in ("bert.cross_encoder", "cls", "bert.projection", "bert.pooler")): save.append(name)
    peft_model = get_peft_model(model, LoraConfig(r=a.lora_rank, target_modules=target, modules_to_save=save))
    peft_model.print_trainable_parameters()
    opt = AdamW(peft_model.parameters(), lr=a.lr)
    steps = a.epochs * len(train_loader)
    sched = get_scheduler("linear", optimizer=opt, num_warmup_steps=int(a.warmup_pct * steps), num_training_steps=steps)
    bar = tqdm(range(steps))

    best_val_f1, best_epoch = -1.0, -1
    for epoch in range(a.epochs):
        peft_model.train(); tr_p, tr_y = [], []
        for batch in train_loader:
            matched = batch["labels"].to(device)
            rna = {"input_ids": batch["rna_input_ids"].to(device), "attention_mask": batch["rna_attention_mask"].to(device), "labels_for_MLM": batch["rna_labels_for_MLM"].to(device)}
            prot = {"input_ids": batch["protein_input_ids"].to(device), "attention_mask": batch["protein_attention_mask"].to(device), "labels_for_MLM": batch["protein_labels_for_MLM"].to(device)}
            opt.zero_grad()
            with autocast("cuda", dtype=torch.bfloat16):
                loss, _, predictions, _ = peft_model(rna, prot, matched, task_mask_lm=True, mlm_scaling_factor=a.mlm_scaling)
            loss.backward(); opt.step(); sched.step()
            tr_p.extend(predictions.cpu().tolist()); tr_y.extend(matched.cpu().tolist()); bar.update(1)
        write_metrics(out / "train_metrics.csv", a.fold, epoch, tr_y, tr_p)
        vy, vp = run_eval(peft_model, val_loader, device); write_metrics(out / "val_metrics.csv", a.fold, epoch, vy, vp)
        ty, tp = run_eval(peft_model, test_loader, device); write_metrics(out / "test_metrics.csv", a.fold, epoch, ty, tp)
        vf1 = f1_score(vy, vp)
        print(f"epoch {epoch}: valF1={vf1:.3f} testMCC={matthews_corrcoef(ty,tp):.3f}", flush=True)
        if vf1 > best_val_f1:
            best_val_f1, best_epoch = vf1, epoch
            peft_model.save_pretrained(str(out / "best_adapter"))
    peft_model.save_pretrained(str(out / "last_adapter"))
    (out / "selection.json").write_text(json.dumps({"best_epoch": best_epoch, "best_val_f1": best_val_f1,
                                                     "epochs": a.epochs, "lr": a.lr, "lora_rank": a.lora_rank,
                                                     "batch_size": a.batch_size, "seed": a.seed}, indent=1))
    print(f"DONE fold {a.fold}: best_epoch={best_epoch} best_val_f1={best_val_f1:.3f}", flush=True)


if __name__ == "__main__":
    main()
