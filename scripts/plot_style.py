"""Paper-grade figure system for mmparnet (RBPNet / CORAL / TFBindFormer aesthetic). Matplotlib-only (no
seaborn dependency, per the env), publication-style: clean spines, consistent color-blind-safe palette,
numeric effect sizes + exact p on-panel, controls always shown beside treatment.

Helpers: apply_style(), despine(), panel_label(); gap_violin() (per-RBP protein vs shuffle vs within-family),
trend_with_ci() (depth/dim trend, M1-vs-M2 contrast), decomp_bars() (coarse vs fine per-RBP gap),
profile_track() (eCLIP observed/predicted/shuffled + fine/coarse overlay), attention_heatmap() (protein-residue
attention with RRM/KH domain bars). Used by the notebook builders so all figures render identically.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
from matplotlib.font_manager import FontProperties

# --- Marsico/Gagneur house palettes (single source of truth) ---
NT_COLORBLIND = {"A": "#009E73", "C": "#0072B2", "G": "#E69F00", "U": "#D55E00", "T": "#D55E00"}  # default logos
BASE_COL = NT_COLORBLIND                                          # logos use the colorblind-safe set
TRACK = {"observed": "#3A3A3A", "target": "#2CA02C", "control": "#D62728", "total": "#1F77B4"}    # RBPNet mixture
DOMAIN = {"RRM": "#0072B2", "KH": "#D55E00", "ZnF": "#009E73", "Helicase": "#CC79A7",
          "linker": "#BDBDBD", "other": "#999999"}
_LOGO_FP = FontProperties(family="DejaVu Sans", weight="bold")
try:
    import logomaker as _logomaker
except Exception:
    _logomaker = None


def set_house_style():
    """Marsico/Gagneur publication look (editable-text PDF, colorblind-safe, clean spines)."""
    apply_style()
    plt.rcParams.update({
        "savefig.dpi": 600, "savefig.pad_inches": 0.02, "font.size": 7, "axes.labelsize": 8,
        "axes.titlesize": 9, "axes.titleweight": "bold", "legend.fontsize": 6.5, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.default": "regular",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    })


def save_fig(fig, stem):
    """Write both stem.pdf and stem.png (editable-text + raster) for paper use."""
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")

# color-blind-safe core (Okabe-Ito-compatible)
PALETTE = {
    "protein": "#2C6FBB",      # protein-conditioned / treatment (deep blue)
    "shuffle": "#9AA3AD",      # shuffled-protein control (neutral grey)
    "family":  "#E08214",      # within-family shuffle (hard control, amber)
    "rna_only": "#5C5C5C",     # RNA-only baseline
    "random":  "#C0C0C0",      # random-body control
    "eclip":   "#B2182B",      # eCLIP target track (red)
    "control": "#2166AC",      # SMInput / control track (blue)
    "cleaned": "#404040",      # cleaned p_target
    "coarse":  "#7FB3D5",      # coarse-envelope component
    "fine":    "#2C6FBB",      # fine single-nt component
    "rrm":     "#1B9E77", "kh": "#D95F02", "linker": "#BDBDBD",
    "hepg2":   "#1B7837", "k562": "#762A83",
    "m1":      "#5C5C5C", "m2": "#2C6FBB",
}
COND = {"real": "protein", "protein": "protein", "shuf": "shuffle", "shuffle": "shuffle",
        "fam": "family", "within-family": "family"}


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 300, "figure.facecolor": "white", "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "axes.titleweight": "bold", "axes.edgecolor": "#333333",
    })


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_label(ax, s):
    ax.text(-0.12, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="left")


def _violin(ax, x, data, color, width=0.62):
    data = np.asarray([v for v in data if v == v], float)
    if len(data) >= 2:
        vp = ax.violinplot([data], positions=[x], widths=width, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(color); b.set_alpha(0.30); b.set_edgecolor(color); b.set_linewidth(1.0)
    if len(data):
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        ax.add_patch(plt.Rectangle((x - 0.045, q1), 0.09, q3 - q1, facecolor=color, edgecolor="black",
                                   linewidth=0.7, alpha=0.92, zorder=3))
        ax.plot([x], [med], "o", color="white", mec="black", ms=4, zorder=4)


def gap_violin(ax, series, title="", ylabel="per-RBP profile Pearson", paired=True, p_annot=None, ref=None):
    """series: ordered dict-like {label: 1D per-RBP array}. Half-violin + paired points + connecting lines.
    Controls (shuffle/family) auto-colored. p_annot: (i,j,text) significance bracket between two positions."""
    labels = list(series); cols = [PALETTE[COND.get(l, "protein")] for l in labels]
    arrs = [np.asarray(series[l], float) for l in labels]
    for i, (a, c) in enumerate(zip(arrs, cols)):
        _violin(ax, i, a, c)
    if paired and len({len(a) for a in arrs}) == 1:
        n = len(arrs[0]); rng = np.random.default_rng(0)
        jit = {i: i + 0.16 + rng.uniform(-0.02, 0.02, n) for i in range(len(arrs))}
        for k in range(n):
            ys = [a[k] for a in arrs]
            ax.plot([jit[i][k] for i in range(len(arrs))], ys, color="#999999", lw=0.4, alpha=0.45, zorder=2)
        for i, (a, c) in enumerate(zip(arrs, cols)):
            ax.scatter(jit[i], a, s=7, color=c, alpha=0.7, edgecolor="none", zorder=3)
    if ref is not None:
        ax.axhline(ref, color=PALETTE["rna_only"], ls="--", lw=1.0)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if p_annot:
        i, j, txt = p_annot; y = max(np.nanmax(arrs[i]), np.nanmax(arrs[j])) * 1.04
        ax.plot([i, i, j, j], [y, y * 1.02, y * 1.02, y], lw=0.8, color="black")
        ax.text((i + j) / 2, y * 1.03, txt, ha="center", va="bottom", fontsize=6.5)
    despine(ax)


def trend_with_ci(ax, x, series, xlabel="", ylabel="gap vs shuffled protein", logx=False, title=""):
    """series: {label: (means[], ci_lo[], ci_hi[], style)}. style optional dict {color,ls,marker}. The
    M1-vs-M2 capacity-contrast figure: pass both trends so divergence is visible on one axis."""
    for lab, v in series.items():
        m = np.asarray(v[0], float); lo = np.asarray(v[1], float); hi = np.asarray(v[2], float)
        st = v[3] if len(v) > 3 and isinstance(v[3], dict) else {}
        c = st.get("color", PALETTE.get(lab, PALETTE["protein"])); ls = st.get("ls", "-"); mk = st.get("marker", "o")
        ax.plot(x, m, ls=ls, marker=mk, color=c, lw=1.5, label=lab, zorder=3)
        ax.fill_between(x, lo, hi, color=c, alpha=0.15, zorder=1)
    ax.axhline(0, color="#000", lw=0.8, ls=":")
    if logx:
        ax.set_xscale("log", base=2); ax.set_xticks(x); ax.set_xticklabels(x)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(frameon=True); despine(ax)


def decomp_bars(ax, rows, comp_keys=("coarse_gap_shuf", "fine_gap_shuf"),
                comp_labels=("coarse envelope", "fine single-nt"), title=""):
    """Per-RBP grouped bars of the gap decomposition, sorted by total gap. The FINE component is the
    load-bearing nt-resolution quantity."""
    rows = sorted(rows, key=lambda r: -(r.get(comp_keys[0], 0) + r.get(comp_keys[1], 0)))
    names = [r["rbp"] for r in rows]; y = np.arange(len(rows))[::-1]
    cols = [PALETTE["coarse"], PALETTE["fine"]]
    h = 0.38
    for ci, (k, lab) in enumerate(zip(comp_keys, comp_labels)):
        off = (ci - 0.5) * h
        ax.barh(y + off, [r.get(k, np.nan) for r in rows], h, color=cols[ci], label=lab,
                edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="#444", lw=1.0)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.5)
    ax.set_xlabel("protein-minus-shuffle profile-Pearson gap")
    if title:
        ax.set_title(title)
    ax.legend(frameon=True, loc="lower right"); despine(ax)


def profile_track(axes, pos, observed, predicted, shuffled=None, fine=None, motif=None, window_span=None,
                  pearson=None, rbp=""):
    """Stacked eCLIP profile tracks sharing x (nt position). axes: list of >=2 axes.
    Top: observed eCLIP (fill) + protein-predicted (line) + shuffled-predicted (line).
    Bottom: a fine/ISM/motif importance heat-strip."""
    a0 = axes[0]
    o = np.asarray(observed, float); o = o / (o.sum() + 1e-9)
    a0.fill_between(pos, o, color=PALETTE["eclip"], alpha=0.55, lw=0, label="observed eCLIP")
    a0.plot(pos, np.asarray(predicted, float), color=PALETTE["protein"], lw=1.2, label="protein-conditioned")
    if shuffled is not None:
        a0.plot(pos, np.asarray(shuffled, float), color=PALETTE["shuffle"], lw=0.8, alpha=0.8, label="shuffled protein")
    if window_span:
        a0.axvspan(window_span[0], window_span[1], color="#FFF3CD", alpha=0.4, zorder=0)
    a0.set_ylabel("profile"); a0.legend(frameon=False, ncol=3, loc="upper right")
    if pearson is not None:
        a0.text(0.99, 0.80, f"r={pearson:+.2f}", transform=a0.transAxes, ha="right", fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", lw=0.5))
    if rbp:
        a0.set_title(f"unseen RBP: {rbp}")
    despine(a0)
    if len(axes) > 1:
        a1 = axes[1]; strip = np.asarray(fine if fine is not None else predicted, float)[None, :]
        nrm = TwoSlopeNorm(vmin=strip.min(), vcenter=0.0, vmax=max(strip.max(), 1e-6)) if strip.min() < 0 else None
        a1.imshow(strip, aspect="auto", cmap="RdBu_r", norm=nrm,
                  extent=[pos[0], pos[-1], 0, 1])
        a1.set_yticks([]); a1.set_xlabel("RNA position (nt)")
        a1.set_ylabel("fine", rotation=0, ha="right", va="center")


def _glyph(ax, base, x, y0, w, h):
    if h <= 1e-6:
        return
    tp = TextPath((0, 0), base, size=1, prop=_LOGO_FP); bb = tp.get_extents()
    if bb.width <= 0 or bb.height <= 0:
        return
    tr = (Affine2D().translate(-bb.x0, -bb.y0).scale(w / bb.width, h / bb.height).translate(x, y0)) + ax.transData
    ax.add_patch(PathPatch(tp, transform=tr, fc=BASE_COL.get(base, "#888"), ec="none"))


def seqlogo(ax, pwm, order="ACGU", title="", ylabel="bits"):
    """Regulatory-genomics information-content sequence logo. pwm: (L,4) probabilities over `order`.
    Per-position letter stack scaled by information content IC = 2 + sum_b p_b log2 p_b (bits)."""
    P = np.asarray(pwm, float); P = P / P.sum(1, keepdims=True).clip(1e-9)
    ic = (2.0 + (P * np.log2(np.clip(P, 1e-9, 1))).sum(1)).clip(0)
    L = P.shape[0]
    for i in range(L):
        heights = P[i] * ic[i]; y = 0.0
        for j in np.argsort(heights):                 # stack smallest first
            if heights[j] > 1e-3:
                _glyph(ax, order[j], i + 0.05, y, 0.9, heights[j]); y += heights[j]
    ax.set_xlim(0, L); ax.set_ylim(0, 2.0); ax.set_xticks(np.arange(L) + 0.5)
    ax.set_xticklabels(range(1, L + 1), fontsize=6); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=8.5)
    despine(ax)


def logo_grid(pwms, names, ncol=3, order="ACGU", suptitle=""):
    """Grid of sequence logos (one generated PWM per RBP). pwms: list of (L,4); returns the figure."""
    n = len(pwms); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 1.3 * nrow), squeeze=False)
    for i, ax in enumerate([a for row in axes for a in row]):
        if i < n:
            seqlogo(ax, pwms[i], order=order, title=names[i], ylabel=("bits" if i % ncol == 0 else ""))
        else:
            ax.set_visible(False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.0)
    fig.tight_layout(); return fig


def attention_heatmap(ax, attn, domains=None, top_k=8, title="", xlabel="protein residue", ylabel="RNA position"):
    """attn: (RNA_pos, residue) or (residue,) 1D residue-attention. domains: list of (start,end,kind,label)
    where kind in {rrm,kh,linker}. Marks top-k attended residues."""
    attn = np.asarray(attn, float)
    if attn.ndim == 1:
        im = ax.imshow(attn[None, :], aspect="auto", cmap="rocket" if "rocket" in plt.colormaps() else "magma",
                       extent=[0, len(attn), 0, 1]); ax.set_yticks([])
        prof = attn
    else:
        im = ax.imshow(attn, aspect="auto", cmap="rocket" if "rocket" in plt.colormaps() else "magma",
                       extent=[0, attn.shape[1], 0, attn.shape[0]])
        prof = attn.mean(0)
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="attention")
    if domains:
        for (s, e, kind, lab) in domains:
            ax.add_patch(plt.Rectangle((s, -0.06 * (1 if attn.ndim == 1 else attn.shape[0]) ), e - s,
                         0.05 * (1 if attn.ndim == 1 else attn.shape[0]), color=PALETTE.get(kind, PALETTE["linker"]),
                         clip_on=False, transform=ax.transData))
            ax.text((s + e) / 2, -0.13 * (1 if attn.ndim == 1 else attn.shape[0]), lab, ha="center",
                    va="top", fontsize=6, color=PALETTE.get(kind, "#555"))
    tk = np.argsort(prof)[-top_k:]
    for r in tk:
        ax.plot([r], [1.04 if attn.ndim == 1 else attn.shape[0] * 1.02], marker="v", ms=3, color="#222", clip_on=False)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def eclip_track(axes, pos, observed, predicted, shuffled=None, rbp="", pearson=None):
    """RBPNet-style genome-browser stack: observed eCLIP crosslink counts (bars) over the protein-conditioned
    predicted per-nt profile + the shuffled-protein prediction. axes = list of 2 (shared x)."""
    a0, a1 = axes[0], axes[1]; obs = np.asarray(observed, float)
    a0.bar(pos, obs, width=1.0, color=TRACK["observed"], lw=0); a0.set_ylabel("crosslink\ncount", fontsize=7)
    a1.plot(pos, np.asarray(predicted, float), color=TRACK["target"], lw=1.0, label="protein-conditioned")
    if shuffled is not None:
        a1.plot(pos, np.asarray(shuffled, float), color="#9AA3AD", lw=0.8, alpha=0.85, label="shuffled protein")
    a1.set_ylabel("predicted prob.\n(per-nt, sum=1)", fontsize=7)
    a1.legend(frameon=False, ncol=2, loc="upper right"); a1.set_xlabel("RNA position (nt)")
    if rbp:
        ttl = f"unseen RBP: {rbp}" + (f"  (profile r={pearson:+.2f})" if pearson is not None else "")
        a0.set_title(ttl, fontsize=9)
    despine(a0); despine(a1)


def bakeoff_bars(ax, items, null_level=None, noise=None, ylabel="gap", title=""):
    """items: list of (name, value, lo, hi, kind) with kind in {winner,match,null,below}. Sorted bars + CI +
    optional within-noise band so null heads are visually obvious."""
    items = sorted(items, key=lambda r: -r[1]); x = np.arange(len(items))
    cc = {"winner": "#1F77B4", "match": "#2CA02C", "null": "#CCCCCC", "below": "#6BAED6"}
    cols = [cc.get(r[4], "#6BAED6") for r in items]
    yerr = [[r[1] - r[2] for r in items], [r[3] - r[1] for r in items]]
    ax.bar(x, [r[1] for r in items], yerr=yerr, capsize=3, color=cols, edgecolor="k", lw=0.5)
    if null_level is not None:
        ax.axhline(null_level, ls="--", color="#999999", lw=1)
        if noise:
            ax.axhspan(null_level - noise, null_level + noise, color="#F0F0F0", zorder=0)
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in items], rotation=30, ha="right", fontsize=6.5)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    despine(ax)


def scatter_identity(ax, x, y, family=None, xlabel="", ylabel="", title=""):
    """Square scatter vs y=x reference (cross-cell replication / pairwise head comparison), family-colored."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    lim = [float(min(x.min(), y.min())), float(max(x.max(), y.max()))]
    pad = 0.05 * (lim[1] - lim[0] + 1e-9); lim = [lim[0] - pad, lim[1] + pad]
    ax.plot(lim, lim, ls="--", color="#999999", lw=0.8, zorder=0)
    c = [DOMAIN.get(f, "#999999") for f in family] if family is not None else "#2C6FBB"
    ax.scatter(x, y, s=16, c=c, edgecolor="w", lw=0.3, zorder=3)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.text(0.05, 0.92, f"{int((y > x).sum())}/{len(x)} above", transform=ax.transAxes, fontsize=7)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    despine(ax)
