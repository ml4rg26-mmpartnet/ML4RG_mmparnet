"""Near-vs-far mechanism: does held-out-family transfer decay with sequence distance to the nearest
TRAINING family? Interpolation (dense-neighbour) predicts MCC rises with nearest-train identity; genuine
long-range generalization would hold even for far (isolated) families.

For each de-peek fold: per held family, (a) MCC from the per-pair predictions (test_preds.csv), and
(b) nearest_id = max % identity of a held-family representative to any TRAIN protein (mmseqs easy-search).
Pools ~300 held families across folds, fits MCC ~ nearest_id (Pearson + slope), and compares the
far-bin (nearest_id < 30%, i.e. truly isolated) mean MCC to the near-bin.

Runs on a host with the mmseqs binary + the CORAL family assets (env-overridable). Writes JSON.
  ML4RG_MMSEQS / ML4RG_CORAL_PROT_FASTA / ML4RG_CLUST_TSV / ML4RG_FAMFULL3_SPLITS / ML4RG_FAMFULL3_OUT / ML4RG_OUT_JSON
"""
from __future__ import annotations
import csv, json, math, os, subprocess, tempfile
from collections import defaultdict

HOME = os.path.expanduser("~")
MMSEQS = os.environ.get("ML4RG_MMSEQS", HOME + "/mmseqs")
PROT_FASTA = os.environ.get("ML4RG_CORAL_PROT_FASTA", HOME + "/coral_prot.fasta")
CLUST_TSV = os.environ.get("ML4RG_CLUST_TSV", HOME + "/clust30_cluster.tsv")
SPLITS = os.environ.get("ML4RG_FAMFULL3_SPLITS", HOME + "/coral-verify/Data/datasets/famfull3")
OUTDIR = os.environ.get("ML4RG_FAMFULL3_OUT", HOME + "/famfull3_out/famfull3")
OUT_JSON = os.environ.get("ML4RG_OUT_JSON", HOME + "/famfull3_nearvsfar.json")
NFOLD, MINPOS = 5, 5


def mcc(tp, tn, fp, fn):
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d else 0.0


def main():
    id2seq = {}; name = None
    for line in open(PROT_FASTA):
        if line[0] == ">": name = line[1:].strip()
        else: id2seq[name] = line.strip()
    mem2fam = {}
    for line in open(CLUST_TSV):
        rep, mem = line.rstrip("\n").split("\t"); mem2fam[mem] = rep
    seq2fam = {id2seq[m]: mem2fam[m] for m in id2seq if m in mem2fam}

    rows_all = []
    for K in range(NFOLD):
        pf = f"{OUTDIR}/{K}/test_preds.csv"
        if not os.path.exists(pf):
            continue
        preds = list(csv.DictReader(open(pf)))
        byfam = defaultdict(list); fam_rep = {}
        for r in preds:
            f = seq2fam.get(r["Prot_seqs"])
            if not f:
                continue
            byfam[f].append((int(float(r["prediction"])), int(float(r["true_label"]))))
            fam_rep.setdefault(f, r["Prot_seqs"])
        # nearest-train identity via mmseqs easy-search (held family reps vs this fold's train proteins)
        train = list(csv.DictReader(open(f"{SPLITS}/fold_{K}/train.csv")))
        trainseqs = sorted(set(r["Prot_seqs"] for r in train))
        td = tempfile.mkdtemp()
        with open(f"{td}/held.fasta", "w") as fh:
            for f, s in fam_rep.items():
                fh.write(f">{f}\n{s}\n")
        with open(f"{td}/train.fasta", "w") as fh:
            for i, s in enumerate(trainseqs):
                fh.write(f">t{i}\n{s}\n")
        m8 = f"{td}/hits.m8"
        subprocess.run([MMSEQS, "easy-search", f"{td}/held.fasta", f"{td}/train.fasta", m8, f"{td}/tmp",
                        "--format-output", "query,target,pident,qcov", "-c", "0.5", "-s", "6"],
                       check=False, capture_output=True, text=True)
        nearest = defaultdict(float)
        if os.path.exists(m8):
            for line in open(m8):
                p = line.rstrip("\n").split("\t")
                if len(p) >= 3:
                    nearest[p[0]] = max(nearest[p[0]], float(p[2]) * 100 if float(p[2]) <= 1 else float(p[2]))
        for f, lst in byfam.items():
            if len(lst) < MINPOS:
                continue
            tp = sum(1 for pr, y in lst if pr == 1 and y == 1); tn = sum(1 for pr, y in lst if pr == 0 and y == 0)
            fp = sum(1 for pr, y in lst if pr == 1 and y == 0); fn = sum(1 for pr, y in lst if pr == 0 and y == 1)
            rows_all.append({"fold": K, "family": f, "n": len(lst),
                             "mcc": mcc(tp, tn, fp, fn), "nearest_id": nearest.get(f, 0.0)})

    xs = [r["nearest_id"] for r in rows_all]; ys = [r["mcc"] for r in rows_all]; n = len(xs)
    summary = {"n_families": n}
    if n > 2:
        mx = sum(xs) / n; my = sum(ys) / n
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = sum((x - mx) ** 2 for x in xs); vy = sum((y - my) ** 2 for y in ys)
        summary["pearson_mcc_vs_nearest_id"] = cov / math.sqrt(vx * vy) if vx * vy > 0 else float("nan")
        summary["slope_mcc_per_pctid"] = cov / vx if vx else float("nan")
        far = [r["mcc"] for r in rows_all if r["nearest_id"] < 30]
        near = [r["mcc"] for r in rows_all if r["nearest_id"] >= 30]
        summary["far_lt30id_n"] = len(far); summary["far_lt30id_meanMCC"] = (sum(far) / len(far)) if far else None
        summary["near_ge30id_n"] = len(near); summary["near_ge30id_meanMCC"] = (sum(near) / len(near)) if near else None
    json.dump({"provenance": {"metric": "per-held-family MCC vs nearest-train-family % identity (mmseqs "
               "easy-search, held rep vs fold train proteins); CORAL de-peek folds"},
               "per_family": rows_all, "summary": summary}, open(OUT_JSON, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print("wrote", OUT_JSON)


if __name__ == "__main__":
    main()
