"""Confound analysis for the famfull family-diversity finding (reusable, writes JSON to mmpartnet_out/).

The famfull held-out-FAMILY result (MCC 0.29-0.37) is only a *protein-family transfer* claim if it is NOT
explained by RNA-side leakage. This script quantifies the confounds on the CPU (no GPU) and writes a
reusable artifact:

  1. RNA overlap    -- fraction of held RNAs that also appear in train (should be LOW for a clean claim).
  2. positive-RNA overlap -- fraction of held POSITIVE RNAs already positive (bound by some other protein)
     in train (a bindable-RNA prior the model can memorize).
  3. protein overlap -- fraction of held proteins in train (should be ~0: family-disjoint by construction).
  4. RNA-only baseline -- predict each held label from ONLY the train bindability P(bound | RNA), protein
     IGNORED. If its MCC ~ CORAL's famfull MCC, the "family transfer" is really RNA-memorization.

Run on the node with the CORAL data (env-overridable paths); writes famfull_confounds.json. Fetch that
into mmpartnet_out/ for the notebook.

  ML4RG_CORAL_PROT_FASTA / ML4RG_CLUST_TSV / ML4RG_CORAL_DATASETS / ML4RG_FAMFULL_OUT / ML4RG_OUT_JSON
"""
from __future__ import annotations
import csv, glob, json, math, os, random
from collections import defaultdict

HOME = os.path.expanduser("~")
PROT_FASTA = os.environ.get("ML4RG_CORAL_PROT_FASTA", HOME + "/coral_prot.fasta")
CLUST_TSV = os.environ.get("ML4RG_CLUST_TSV", HOME + "/clust30_cluster.tsv")
DATASETS = os.environ.get("ML4RG_CORAL_DATASETS", HOME + "/coral-verify/Data/datasets")
FAMFULL_OUT = os.environ.get("ML4RG_FAMFULL_OUT", HOME + "/famfull_out/famfull")
OUT_JSON = os.environ.get("ML4RG_OUT_JSON", HOME + "/famfull_confounds.json")
NFOLD, H, MINPOS = 5, 60, 5


def mcc(tp, tn, fp, fn):
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d else 0.0


def eval_thr(score, y, thr):
    tp = sum(1 for s, l in zip(score, y) if s >= thr and l == 1)
    tn = sum(1 for s, l in zip(score, y) if s < thr and l == 0)
    fp = sum(1 for s, l in zip(score, y) if s >= thr and l == 0)
    fn = sum(1 for s, l in zip(score, y) if s < thr and l == 1)
    return mcc(tp, tn, fp, fn), (tp + tn) / max(1, len(y))


def coral_famfull_mcc(k):
    f = f"{FAMFULL_OUT}/{k}/val_metrics.csv"
    if not os.path.exists(f):
        return None, None
    rows = list(csv.DictReader(open(f)))
    m = [float(r["MCC"]) for r in rows if r.get("MCC") not in (None, "")]
    return (max(m), m[-1]) if m else (None, None)


def main():
    id2seq = {}; name = None
    for line in open(PROT_FASTA):
        if line[0] == ">":
            name = line[1:].strip()
        else:
            id2seq[name] = line.strip()
    mem2fam = {}
    for line in open(CLUST_TSV):
        rep, mem = line.rstrip("\n").split("\t"); mem2fam[mem] = rep
    seq2fam = {id2seq[m]: mem2fam[m] for m in id2seq if m in mem2fam}
    pairs = {}
    for f in glob.glob(DATASETS + "/*/*/*.csv"):
        if "/famfull/" in f.replace("\\", "/"):
            continue
        for r in csv.DictReader(open(f)):
            k = (r["RNA_id"], r["Protein_id"])
            if k not in pairs and r.get("Prot_seqs") in seq2fam:
                r["_fam"] = seq2fam[r["Prot_seqs"]]; pairs[k] = r
    fam2pos = defaultdict(list); fam2neg = defaultdict(list)
    for r in pairs.values():
        (fam2pos if str(r["labels"]) == "1" else fam2neg)[r["_fam"]].append(r)
    fams = [f for f in fam2pos if len(fam2pos[f]) >= MINPOS]
    random.seed(0); random.shuffle(fams)

    folds = []
    for k in range(NFOLD):
        TEST = set(fams[k * H:(k + 1) * H]); TRAIN = [f for f in fams if f not in TEST]
        tr = [r for f in TRAIN for r in (fam2pos[f] + fam2neg[f])]
        # overlap stats
        tr_rna = set(r["RNA_id"] for r in tr)
        tr_pos_rna = set(r["RNA_id"] for r in tr if str(r["labels"]) == "1")
        tr_prot = set(r["Protein_id"] for r in tr)
        te_all = [r for f in TEST for r in (fam2pos[f] + fam2neg[f])]
        te_rna = set(r["RNA_id"] for r in te_all)
        te_pos_rna = set(r["RNA_id"] for r in te_all if str(r["labels"]) == "1")
        te_prot = set(r["Protein_id"] for r in te_all)
        # RNA-only bindability baseline (protein ignored), on the balanced held set
        rna_pos = defaultdict(int); rna_tot = defaultdict(int)
        for r in tr:
            rna_tot[r["RNA_id"]] += 1; rna_pos[r["RNA_id"]] += (str(r["labels"]) == "1")
        bind = {rna: rna_pos[rna] / rna_tot[rna] for rna in rna_tot}
        prior = sum(1 for r in tr if str(r["labels"]) == "1") / max(1, len(tr))
        rng = random.Random(200 + k)
        pos = [r for f in TEST for r in fam2pos[f]]; neg = [r for f in TEST for r in fam2neg[f]]
        rng.shuffle(neg); neg = neg[:len(pos)]; te = pos + neg
        y = [1 if str(r["labels"]) == "1" else 0 for r in te]
        score = [bind.get(r["RNA_id"], prior) for r in te]
        rna_best = max(eval_thr(score, y, t / 20.0)[0] for t in range(1, 20))
        rna_m5, rna_a5 = eval_thr(score, y, 0.5)
        cbest, clast = coral_famfull_mcc(k)
        folds.append({
            "fold": k, "n_test_fams": len(TEST), "n_train_fams": len(TRAIN),
            "rna_overlap_held_in_train": len(te_rna & tr_rna) / max(1, len(te_rna)),
            "pos_rna_overlap": len(te_pos_rna & tr_pos_rna) / max(1, len(te_pos_rna)),
            "protein_overlap": len(te_prot & tr_prot) / max(1, len(te_prot)),
            "rna_only_baseline_bestMCC": rna_best, "rna_only_baseline_MCC@0.5": rna_m5,
            "rna_only_baseline_Acc@0.5": rna_a5,
            "coral_famfull_bestMCC": cbest, "coral_famfull_lastMCC": clast,
        })
        print(f"fold {k}: RNAov={folds[-1]['rna_overlap_held_in_train']:.3f} "
              f"posRNAov={folds[-1]['pos_rna_overlap']:.3f} protov={folds[-1]['protein_overlap']:.3f} "
              f"| RNA-only bestMCC={rna_best:.3f} vs CORAL best={cbest}", flush=True)

    def mean(key, src=folds):
        v = [f[key] for f in src if f.get(key) is not None]
        return sum(v) / len(v) if v else None
    summary = {
        "rna_overlap_mean": mean("rna_overlap_held_in_train"),
        "pos_rna_overlap_mean": mean("pos_rna_overlap"),
        "protein_overlap_mean": mean("protein_overlap"),
        "rna_only_baseline_bestMCC_mean": mean("rna_only_baseline_bestMCC"),
        "coral_famfull_bestMCC_mean": mean("coral_famfull_bestMCC"),
        "coral_famfull_lastMCC_mean": mean("coral_famfull_lastMCC"),
    }
    verdict = ("CONFOUNDED: RNAs are shared train<->held and an RNA-only baseline approaches the CORAL "
               "number -> the family-diversity result is (at least partly) RNA-bindability memorization, "
               "not protein-family transfer. A clean test must hold out RNAs too (RNA+family disjoint) "
               "and/or a protein-shuffle control on the held set."
               if (summary["rna_only_baseline_bestMCC_mean"] or 0) >= 0.6 * (summary["coral_famfull_bestMCC_mean"] or 1)
               else "RNA-only baseline is well below CORAL -> protein signal contributes beyond RNA memorization "
                    "(still verify with a protein-shuffle control + RNA-disjoint split).")
    out = {"provenance": {"computed_on": "TUM cluster (CPU); CORAL fork data", "task": "famfull 5-fold "
           "held-out-family, balanced random-repair negatives", "note": "RNA-only baseline ignores the "
           "protein entirely; compares to CORAL famfull MCC"},
           "per_fold": folds, "summary": summary, "verdict": verdict}
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nSUMMARY:", json.dumps(summary, indent=1), flush=True)
    print("VERDICT:", verdict, flush=True)
    print("wrote", OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
