"""Shared visualization helpers for the mmpartnet result notebooks/figures. Field-inspired style
(RBPNet/TFBindFormer-like clean panels) with a VIOLIN + inner-BOX hybrid as the default distribution
plot -- it shows the full per-RBP distribution (shape, tails, median, IQR) instead of a mean+CI bar, so we
read much more from each comparison. Pure matplotlib (no seaborn dependency)."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RC = {"figure.dpi": 130, "savefig.dpi": 150, "figure.facecolor": "white", "font.family": "DejaVu Sans",
      "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
plt.rcParams.update(RC)
PALETTE = {"film": "#1f9d6b", "xattn": "#7e57c2", "perres": "#d84315", "bidir": "#1565c0",
           "concat": "#9e9e9e", "real": "#7e57c2", "shuf": "#bdbdbd", "fam": "#f0a35e",
           "string": "#1f9d6b", "esm": "#7e57c2"}


def violin_box(ax, data_lists, positions, colors, width=0.7, points=True, point_max=60):
    """Violin + inner boxplot (quartiles) + median marker + faint jittered points. data_lists: list of 1D
    arrays (one per position). Returns nothing; draws on ax. Robust to empty/short arrays."""
    rng = np.random.default_rng(0)
    clean = [np.asarray([v for v in d if v == v], float) for d in data_lists]  # drop nan
    valid_pos = [p for p, d in zip(positions, clean) if len(d) >= 2]
    valid_dat = [d for d in clean if len(d) >= 2]
    if valid_dat:
        vp = ax.violinplot(valid_dat, positions=valid_pos, widths=width, showmeans=False,
                           showmedians=False, showextrema=False)
        for i, b in enumerate(vp["bodies"]):
            c = colors[positions.index(valid_pos[i])] if valid_pos[i] in positions else "#888"
            b.set_facecolor(c); b.set_alpha(0.35); b.set_edgecolor(c); b.set_linewidth(1.0)
    for p, d in zip(positions, clean):
        if len(d) == 0:
            continue
        c = colors[positions.index(p)]
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        ax.add_patch(plt.Rectangle((p - 0.06, q1), 0.12, q3 - q1, facecolor=c, edgecolor="black",
                                   linewidth=0.8, alpha=0.9, zorder=3))
        lo = max(d.min(), q1 - 1.5 * (q3 - q1)); hi = min(d.max(), q3 + 1.5 * (q3 - q1))
        ax.plot([p, p], [lo, hi], color="black", lw=0.8, zorder=2)
        ax.plot([p], [med], marker="o", color="white", markeredgecolor="black", markersize=4, zorder=4)
        if points and len(d) <= 400:
            n = min(len(d), point_max)
            sub = d if len(d) <= point_max else rng.choice(d, point_max, replace=False)
            ax.scatter(p + 0.10 + rng.uniform(-0.03, 0.03, n), sub, s=6, color=c, alpha=0.5,
                       edgecolor="none", zorder=2)


def grouped_violins(ax, groups, cats, colors, gap=1.0, sub=0.32, title="", ylabel=""):
    """groups: dict cat -> list of per-series arrays (series = e.g. real/shuf/fam). cats on x; series within.
    colors: list aligned to series. Returns the x tick centers."""
    nser = len(colors); centers = []
    for gi, cat in enumerate(cats):
        base = gi * gap; positions = [base + (j - (nser - 1) / 2) * sub for j in range(nser)]
        centers.append(base)
        violin_box(ax, groups[cat], positions, colors)
    ax.set_xticks(centers); ax.set_xticklabels(cats)
    if title:
        ax.set_title(title, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel)
    return centers


def legend(ax, labels, colors, **kw):
    h = [plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=c, markersize=9, markeredgecolor=c)
         for c in colors]
    ax.legend(h, labels, **{"fontsize": 8.5, "frameon": True, **kw})


def col(name):
    """Stable colour for a method/rep/condition label (palette hit, else hashed pastel)."""
    key = name.split("/")[-1].replace("+string", "string").replace("-bidir", "").replace("-bce", "")
    if key in PALETTE:
        return PALETTE[key]
    import colorsys
    h = (abs(hash(name)) % 997) / 997.0
    return colorsys.hsv_to_rgb(h, 0.45, 0.8)


def baseline_bars(ax, values, lines=None, markers=None, ylabel="", title="", ylim=None):
    """Generic: bars for {name:val}, horizontal baseline lines {name:(val,style,colour)}, marker dashes
    {name:val}. Schema-agnostic so any team's result dict plots the same way."""
    names = list(values); xs = np.arange(len(names))
    ax.bar(xs, [values[n] for n in names], 0.6, color=[col(n) for n in names], alpha=0.9, edgecolor="white")
    for nm, spec in (lines or {}).items():
        v, ls, c = spec if isinstance(spec, tuple) else (spec, "--", "#000")
        ax.axhline(v, color=c, ls=ls, lw=1.3, label=f"{nm} {v:.3f}")
    for i, n in enumerate(names):
        if markers and n in markers:
            ax.scatter([i], [markers[n]], marker="_", s=300, color="#bdbdbd", zorder=4)
    ax.set_xticks(xs); ax.set_xticklabels(names)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title, fontsize=11)
    if ylim: ax.set_ylim(*ylim)
    if lines: ax.legend(fontsize=8)


def delta_violins(ax, series, colors=None, zero=True, ylabel="", title=""):
    """Generic per-item delta distribution: series = {label: 1D-array}. Draws one violin-box per label."""
    labs = list(series); colors = colors or [col(l) for l in labs]
    for i, l in enumerate(labs):
        violin_box(ax, [np.asarray(series[l], float)], [i], [colors[i]])
    if zero: ax.axhline(0, color="#000", lw=1.0, ls="--")
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title, fontsize=11)


# ── high-level figure builders (one call per result JSON; adapt to whatever it contains) ──────────
def fig_fair(d):
    """binding_fair.json -> (gain-vs-RNA-only violins | absolute auPRC vs baselines)."""
    M = d["methods"]; B = d.get("baselines", {}); ms = list(M)
    fig, (a, c) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.1, 1]})
    delta_violins(a, {m: [r["vs_rna"] for r in M[m]["rows"]] for m in ms},
                  ylabel="per-RBP auPRC gain over RNA-only", title="Fair test: method − RNA-only (0 = baseline)")
    lines = {}
    if "rna_only_multitask" in B: lines["RNA-only multitask"] = (B["rna_only_multitask"], "--", "#000")
    if "rna_only_randombody" in B: lines["random-body (rest=leakage)"] = (B["rna_only_randombody"], "-.", "#c62828")
    if "rna_only_bindability" in B: lines["protein-agnostic"] = (B["rna_only_bindability"], ":", "#888")
    baseline_bars(c, {m: M[m]["real"] for m in ms}, lines=lines,
                  markers={m: M[m]["shuffle"] for m in ms}, ylabel="mean per-RBP auPRC",
                  title="Absolute auPRC vs baselines (— = shuffle)")
    fig.tight_layout(); return fig


def fig_dimsweep(d, ref_name="perres32_ref", prefix="perres_full_"):
    """P1 dim-sweep (binding_x_p1_dimsweep.json): per-residue auPRC gain over the RNA-only baseline vs the
    protein-projection dim, with bootstrap CI -- the CORAL-direction lever 'does widening the only head that
    beats baseline help?'. The lab's 32-d reduced rep is drawn as a reference line."""
    M = d["methods"]; pts = []
    for m in M:
        if m.startswith(prefix):
            try:
                dim = int(m.split("_")[-1])
            except ValueError:
                dim = M[m].get("dp", 0)
            pts.append((dim, M[m]))
    pts.sort()
    dims = [p[0] for p in pts]; g = [p[1]["gap_vs_rna_only"] for p in pts]
    lo = [p[1]["gap_vs_rna_only"] - p[1]["gap_vs_rna_only_ci"][0] for p in pts]
    hi = [p[1]["gap_vs_rna_only_ci"][1] - p[1]["gap_vs_rna_only"] for p in pts]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(dims, g, yerr=[lo, hi], marker="o", color=PALETTE["perres"], capsize=4, lw=1.6,
                label="per-residue ESM-2 (PCA→dim)")
    if ref_name in M:
        rg = M[ref_name]["gap_vs_rna_only"]
        ax.axhline(rg, color=PALETTE["bidir"], ls="--", lw=1.3, label=f"lab 32-d reduced ({rg:+.4f})")
    ax.axhline(0, color="#000", lw=1.0, ls=":")
    if dims:
        ax.set_xscale("log", base=2); ax.set_xticks(dims); ax.set_xticklabels(dims)
    ax.set_xlabel("per-residue protein projection dim"); ax.set_ylabel("auPRC gain over RNA-only baseline")
    ax.set_title("P1: does widening the per-residue protein lift the winning head?", fontsize=11)
    ax.legend(fontsize=9); fig.tight_layout(); return fig


def fig_mechanism(dm, dn2=None):
    """binding_mechanism.json (+ optional binding_n2_xattn.json) -> ladder violins | N2 objective bars."""
    mech = dm["mechanisms"]; order = [k for k in ("concat", "film", "xattn") if k in mech]
    n = 2 if dn2 else 1
    fig, axs = plt.subplots(1, n, figsize=(6 * n, 5)); axs = np.atleast_1d(axs)
    delta_violins(axs[0], {k: [r["real_mean"] - r["shuffle_mean"] for r in mech[k]["rows"]] for k in order},
                  ylabel="per-RBP multimodal gain (auPRC real − shuffle)", title="Binding: conditioning ladder")
    if dn2:
        cells = list(dn2["cells"]); xs = np.arange(len(cells)); b = axs[1]
        b.bar(xs - 0.2, [dn2["cells"][c]["n2_real"] for c in cells], 0.4, color="#1f9d6b", label="real", edgecolor="white")
        b.bar(xs + 0.2, [dn2["cells"][c]["n2_perm"] for c in cells], 0.4, color="#bdbdbd", label="permuted", edgecolor="white")
        b.axhline(0.5, color="#888", ls=":", lw=1); b.set_xticks(xs); b.set_xticklabels([c.replace("-", "\n") for c in cells], fontsize=8)
        b.set_ylim(0.45, 0.75); b.set_ylabel("N2-auROC"); b.legend(fontsize=8); b.set_title("N2 task: objective is the lever", fontsize=11)
    fig.tight_layout(); return fig


def fig_faithfulness(d):
    """xattn_faithfulness.json -> attn-vs-ISM agreement (real vs shuffle) | example window track."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.3]})
    delta_violins(a, {"real protein": [x["sp_real"] for x in d["rows"]], "shuffled": [x["sp_shuf"] for x in d["rows"]]},
                  colors=[PALETTE["real"], PALETTE["shuf"]], zero=True,
                  ylabel="Spearman(attention, ISM)", title="Attention–ISM agreement (per window)")
    ex = d.get("example")
    if ex:
        nrm = lambda v: (np.array(v) - min(v)) / (max(v) - min(v) + 1e-9); x = np.arange(len(ex["ism"]))
        b.plot(x, nrm(ex["ism"]), color="#000", lw=1.6, label="ISM (perturbation)")
        b.plot(x, nrm(ex["attn_real"]), color=PALETTE["real"], lw=1.6, label="attention (real)")
        b.plot(x, nrm(ex["attn_shuf"]), color="#bdbdbd", lw=1.3, ls="--", label="attention (shuffled)")
        b.set_xlabel("RNA position (downsampled)"); b.set_ylabel("normalized importance")
        b.set_title(f"Example window ({ex['rbp']}): attention peaks where ISM does", fontsize=10.5); b.legend(fontsize=8)
    fig.tight_layout(); return fig


def fig_domains(d):
    """binding_xattn_perres.json domain_rows -> RRM/KH attention enrichment vs control (sorted barh)."""
    dr = sorted(d["domain_rows"], key=lambda r: -r["enrichment"]); names = [r["rbp"] for r in dr]
    y = np.arange(len(dr))[::-1]
    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.42 * len(dr))))
    ax.barh(y + 0.18, [r["enrichment"] for r in dr], 0.36, color=PALETTE["perres"], alpha=0.9, label="RRM/KH domain", edgecolor="white")
    ax.barh(y - 0.18, [r["ctrl_mean"] for r in dr], 0.36, xerr=[r.get("ctrl_std", 0) for r in dr], capsize=2,
            color="#bdbdbd", alpha=0.9, label="random control", edgecolor="white")
    ax.axvline(1.0, color="#444", ls="--", lw=1); ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("attention-mass enrichment"); ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title(f"Per-residue attention reads RRM/KH domains (mean {np.mean([r['enrichment'] for r in dr]):.2f}×)", fontsize=10.5)
    fig.tight_layout(); return fig


def fig_rep(d):
    """binding_ribex.json -> per-RBP binding-gain violins by protein representation."""
    C = d["conds"]
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(C)), 5))
    delta_violins(ax, {c: [r["real"] - r["der"] for r in C[c]["rows"]] for c in C},
                  ylabel="per-RBP multimodal gain (auPRC real − shuffle)", title="Protein representation ablation")
    return fig


def fig_competence(d):
    """binding_competence.json -> reliability scatters (gain vs OOD distance | vs binding strength)."""
    rows = d["rows"]; ood = np.array([r["ood"] for r in rows]); gain = np.array([r["gain"] for r in rows])
    strg = np.array([r["pos_rate"] for r in rows]); labs = [r["rbp"] for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, xv, sp, xl in [(a, ood, d["spearman_ood_gain"], "protein-OOD distance"),
                           (b, strg, d["spearman_strength_gain"], "binding strength (test pos rate)")]:
        ax.scatter(xv, gain, s=55, color=PALETTE["xattn"] if ax is a else PALETTE["perres"], edgecolor="white", zorder=3)
        for x, yv, l in zip(xv, gain, labs): ax.annotate(l, (x, yv), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color="#888", lw=0.8); ax.set_xlabel(xl); ax.set_title(f"vs {xl.split('(')[0].strip()} (Spearman {sp:+.2f})", fontsize=10.5)
    a.set_ylabel("multimodal gain"); fig.tight_layout(); return fig


def fig_structure(d):
    """binding_structure.json -> per-RBP delta violin | body vs body+struct scatter."""
    rows = d["rows"]; delta = np.array([r["delta"] for r in rows]); body = np.array([r["body"] for r in rows]); bs = np.array([r["body_struct"] for r in rows])
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1, 1.2]})
    delta_violins(a, {"body+struct − body": delta}, colors=[PALETTE["bidir"]],
                  ylabel="per-RBP auPRC delta", title=f"delta {d['delta_mean']:+.4f} CI[{d['delta_ci'][0]:+.4f},{d['delta_ci'][1]:+.4f}]")
    lim = max(body.max(), bs.max()) * 1.08; b.plot([0, lim], [0, lim], "--", color="#888", lw=1)
    b.scatter(body, bs, s=28, color=PALETTE["bidir"], alpha=0.7, edgecolor="white")
    b.set_xlim(0, lim); b.set_ylim(0, lim); b.set_xlabel("body auPRC"); b.set_ylabel("body+struct auPRC"); b.set_title("per-RBP", fontsize=10.5)
    fig.tight_layout(); return fig


def fig_m2(cells):
    """cells = [(name, m2_profile_dict), ...] -> per-cell grouped profile-Pearson violins (real/shuf/fam)."""
    fig, axes = plt.subplots(1, len(cells), figsize=(6 * len(cells), 5), sharey=True); axes = np.atleast_1d(axes)
    cols = [PALETTE["real"], PALETTE["shuf"], PALETTE["fam"]]
    for ax, (nm, dd) in zip(axes, cells):
        archs = list(dd["archs"])
        groups = {a: [np.array([x["pearson_real"] for x in dd["archs"][a]["rows"]]),
                      np.array([x["pearson_shuf"] for x in dd["archs"][a]["rows"]]),
                      np.array([x["pearson_fam"] for x in dd["archs"][a]["rows"]])] for a in archs}
        grouped_violins(ax, groups, archs, cols, ylabel=("per-RBP profile Pearson" if ax is axes[0] else ""))
        ax.axhline(0, color="#888", lw=0.8); ax.set_title(f"{nm} (K={dd['K']})", fontsize=11)
    legend(axes[-1], ["real", "shuffled", "within-family"], cols, loc="upper right")
    fig.tight_layout(); return fig
