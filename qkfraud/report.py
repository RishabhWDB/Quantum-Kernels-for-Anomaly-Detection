import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import diagnostics, stats

COLORS = {"qsvm": "#1f77b4", "rbf_svm": "#e69f00", "isolation_forest": "#009e73"}
NAMES = {"qsvm": "QSVM", "rbf_svm": "RBF-SVM", "isolation_forest": "IsolationForest"}
MAIN_METRICS = ["pr_auc", "roc_auc", "recall_at_fpr_1", "recall_at_fpr_5", "precision_at_k", "best_f1", "mcc"]
PREV_METRICS = ["pr_auc", "roc_auc", "recall_at_fpr_1", "precision_at_recall_50", "precision_at_recall_80",
                "fp_per_1k_at_recall_50", "fp_per_1k_at_recall_80"]


def _load(results_dir, name):
    path = Path(results_dir) / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _frame(payload):
    return pd.DataFrame(payload["records"]) if payload else None


def _pm(values):
    return f"{np.mean(values):.3f} ± {np.std(values, ddof=1):.3f}" if len(values) > 1 else f"{np.mean(values):.3f}"


def _md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _pivot(df, keys, model_col="model", metric="pr_auc"):
    return df.pivot_table(index="seed", columns=[*keys, model_col], values=metric).sort_index()


def _compare(per_seed, pairs, n_resamples):
    out = []
    for label, a, b in pairs:
        if a in per_seed and b in per_seed:
            out.append({"comparison": label, **stats.paired_comparison(per_seed[a], per_seed[b], n_resamples)})
    return stats.family(out)


def _comparison_table(rows):
    return _md(pd.DataFrame([
        {**{k: v for k, v in r.items() if k not in ("delta", "ci_low", "ci_high", "p", "p_holm")},
         "ΔPR-AUC": f"{r['delta']:+.3f}", "95% CI": f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]",
         "p": f"{r['p']:.3f}", "p_Holm": f"{r['p_holm']:.3f}"} for r in rows
    ]))


def _summary_table(df, group_cols, metrics):
    rows = []
    for key, g in df.groupby(group_cols, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(group_cols, key)), **{m: _pm(g[m].dropna().to_numpy()) for m in metrics if m in g}})
    return _md(pd.DataFrame(rows))


def _seed_series(df, mask):
    return df[mask].sort_values("seed")["pr_auc"].to_numpy()


def table_main(main, prev, n_resamples, out):
    test = main[main.test == "test"].copy()
    test["model"] = test["model"].map(NAMES)
    out.append("## Table IV: test performance at 10% fraud\n")
    out.append(_summary_table(test, ["features", "model"], MAIN_METRICS))
    out.append("\n## Table V: paired Wilcoxon tests on PR-AUC (Holm within each fraud level)\n")
    rows = []
    for level, df in (("10%", main[main.test == "test"]), ("0.2%", prev[prev.test == "prevalence"] if prev is not None else None)):
        if df is None:
            continue
        family = []
        for d in sorted(df.d.dropna().unique()):
            s = {m: _seed_series(df, (df.d == d) & (df.model == m)) for m in NAMES}
            for b in ("rbf_svm", "isolation_forest"):
                family.append({"fraud": level, "d": int(d), "comparison": f"QSVM − {NAMES[b]}",
                               **stats.paired_comparison(s["qsvm"], s[b], n_resamples)})
        rows += stats.family(family)
    out.append(_comparison_table(rows))


def edge_summary(main, out):
    rows = []
    for model, g in main[main.test == "test"].groupby("model"):
        counts = {}
        for edges in g["edges"]:
            for param, e in edges.items():
                c = counts.setdefault(param, {"edge": 0, "plateau": 0, "still_improving": 0})
                c["edge"] += 1
                c[e["status"]] += 1
        rows += [{"model": NAMES[model], "parameter": p, **c, "fits": len(g)} for p, c in counts.items()]
    out.append("\n## Grid-edge diagnostic (main experiment)\n")
    out.append(_md(pd.DataFrame(rows)) if rows else "No edge choices.")


def table_qubits(df, n_resamples, out):
    out.append("\n## Table VI: test PR-AUC vs number of qubits\n")
    family, rows = [], []
    for d in sorted(df.d.unique()):
        s = {m: _seed_series(df, (df.d == d) & (df.model == m)) for m in NAMES}
        rows.append({"d": d, **{NAMES[m]: _pm(v) for m, v in s.items()}})
        for b in ("rbf_svm", "isolation_forest"):
            family.append({"d": d, "vs": b, **stats.paired_comparison(s["qsvm"], s[b], n_resamples)})
    family = stats.family(family)
    for r in rows:
        r["p_Holm (vs RBF)"] = f"{next(f['p_holm'] for f in family if f['d'] == r['d'] and f['vs'] == 'rbf_svm'):.3f}"
    out.append(_md(pd.DataFrame(rows)))


def table_prevalence(df, out):
    df = df[df.test == "prevalence"].copy()
    df["model"] = df["model"].map(NAMES)
    out.append("\n## Table VII: realistic prevalence (0.2% fraud)\n")
    out.append(_summary_table(df, ["d", "model"], PREV_METRICS))


def table_noise(df, out):
    out.append("\n## Table VIII: QSVM under finite-shot and depolarizing noise\n")
    rows = []
    for (kernel, d, kind, shots, eps), g in df.groupby(["kernel", "d", "noise", "shots", "eps"], sort=False, dropna=False):
        label = "exact" if kind == "exact" else (f"{int(shots):,} shots" if kind == "shots" else f"ε = {eps} (p = {g.survival.iloc[0]:.3f})")
        rows.append({"kernel": kernel, "d": d, "noise": label, "PR-AUC": _pm(g.pr_auc), "ROC-AUC": _pm(g.roc_auc),
                     "KTA": f"{g.kta.mean():.3f}", "off-diag mean": f"{g.offdiag_mean.mean():.3f}",
                     "off-diag var": f"{g.offdiag_var.mean():.2e}", "PSD clip": f"{int(g.psd_clipped.sum())}/{len(g)}"})
    out.append(_md(pd.DataFrame(rows)))


def table_geometry(df, out):
    out.append("\n## Table IX: training-kernel diagnostics (mean over seeds)\n")
    rows = []
    for d in sorted(df.d.unique()):
        g = df[df.d == d]
        rbf = g[g.kernel == "rbf"]
        rows.append({"d": d, "kernel": "RBF (tuned γ)", "KTA": f"{rbf.kta.mean():.3f}", "r": "–", "g": "–"})
        for lam, h in g[g.kernel == "zz"].groupby("bandwidth"):
            rows.append({"d": d, "kernel": f"ZZ, λ = {lam}", "KTA": f"{h.kta.mean():.3f}",
                         "r": f"{h.pearson_rbf.mean():.2f}", "g": f"{h.geometric_difference.mean():.1f}"})
        sel = g[(g.kernel == "zz") & (g.selected == True)]  # noqa: E712
        rows.append({"d": d, "kernel": "ZZ, selected λ", "KTA": f"{sel.kta.mean():.3f}",
                     "r": f"{sel.pearson_rbf.mean():.2f}", "g": f"{sel.geometric_difference.mean():.1f}"})
    out.append(_md(pd.DataFrame(rows)))


def table_feature_map(df, n_resamples, out):
    out.append("\n## Table X: Z vs ZZ feature map (C and λ tuned jointly)\n")
    family = []
    for (d, reps), g in df.groupby(["d", "reps"]):
        z = _seed_series(g, g.kind == "z")
        zz = _seed_series(g, g.kind == "zz")
        family.append({"d": d, "reps": reps, "Z map": _pm(z), "ZZ map": _pm(zz), **stats.paired_comparison(z, zz, n_resamples)})
    out.append(_comparison_table(stats.family(family)))


def table_learning_curve(df, n_resamples, out):
    out.append("\n## Learning curves: test PR-AUC vs training-set size\n")
    out.append(_summary_table(df.assign(model=df.model.map(NAMES)), ["d", "n_train", "model"], ["pr_auc"]))
    family = []
    for (d, n), g in df.groupby(["d", "n_train"]):
        s = {m: _seed_series(g, g.model == m) for m in NAMES}
        for b in ("rbf_svm", "isolation_forest"):
            family.append({"d": d, "n": n, "comparison": f"QSVM − {NAMES[b]}", **stats.paired_comparison(s["qsvm"], s[b], n_resamples)})
    out.append("\n" + _comparison_table(stats.family(family)))


def table_bandwidth(df, out):
    out.append("\n## Bandwidth sweep (C tuned per λ)\n")
    out.append(_summary_table(df, ["d", "bandwidth"], ["pr_auc", "kta", "offdiag_var"]))
    rows = []
    for d, g in df.groupby("d"):
        m = g.groupby("bandwidth")[["kta", "pr_auc"]].mean()
        rho, p = spearmanr(m.kta, m.pr_auc)
        rows.append({"d": d, "Spearman ρ (KTA vs PR-AUC)": f"{rho:.2f}", "p": f"{p:.3f}",
                     "median C per λ": ", ".join(f"{lam}: {c:g}" for lam, c in g.groupby("bandwidth").apply(lambda h: np.median([p['C'] for p in h.params])).items())})
    out.append("\n" + _md(pd.DataFrame(rows)))


def table_pca(df, out):
    out.append("\n## Table I: cumulative PCA variance retained (%)\n")
    out.append(_md(pd.DataFrame([{"d": d, "variance (%)": f"{100 * g.variance.mean():.1f} ± {100 * g.variance.std(ddof=1):.1f}"}
                                 for d, g in df.groupby("d")])))


def table_resources(payload, out):
    out.append("\n## Table II: circuit resources (reps = 2, linear coupling)\n")
    rows = [{"d": r["d"], **{f"{c} {k}": v for c in ("feature_map", "kernel") for k, v in r[c].items()}} for r in payload["records"]]
    out.append(_md(pd.DataFrame(rows)))
    out.append("\nHardware cost: " + ", ".join(f"{k} = {v:,}" for k, v in payload["hardware_cost"].items()))


def _errorbar(ax, x, groups, **kw):
    ax.errorbar(x, groups.mean().to_numpy(), yerr=groups.std(ddof=1).to_numpy(), capsize=3, marker="o", **kw)


def figures(results_dir, fig_dir, frames):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    main, bw, conc, lc, noise_df, prev = (frames.get(k) for k in ("main", "bandwidth", "concentration", "learning_curve", "noise", "prevalence"))

    if main is not None:
        t = main[main.test == "test"]
        feats = list(dict.fromkeys(t.features))
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, m in enumerate(NAMES):
            g = t[t.model == m].groupby("features").pr_auc
            xs = [j + (i - 1) * 0.27 for j, f in enumerate(feats) if f in g.groups]
            ys = [g.get_group(f) for f in feats if f in g.groups]
            ax.bar(xs, [y.mean() for y in ys], 0.27, yerr=[y.std(ddof=1) for y in ys], capsize=3, color=COLORS[m], label=NAMES[m])
        ax.set_xticks(range(len(feats)), feats)
        ax.set_ylabel("Test PR-AUC")
        ax.set_ylim(0, 1)
        ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.3))
        fig.savefig(fig_dir / "fig3_main.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    if bw is not None:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
        for d, g in bw.groupby("d"):
            grp = g.groupby("bandwidth")
            _errorbar(a1, grp.pr_auc.mean().index, grp.pr_auc, label=f"QSVM, d = {d}")
            _errorbar(a2, grp.offdiag_var.mean().index, grp.offdiag_var, label=f"d = {d}")
            if main is not None:
                ref = main[(main.test == "test") & (main.model == "rbf_svm") & (main.d == d)].pr_auc.mean()
                a1.axhline(ref, ls="--", lw=1, color="gray")
        a1.set(xscale="log", xlabel="Bandwidth λ", ylabel="Test PR-AUC")
        a2.set(xscale="log", yscale="log", xlabel="Bandwidth λ", ylabel="Off-diagonal variance of K")
        a1.legend()
        fig.savefig(fig_dir / "fig4_bandwidth.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        for d, g in bw.groupby("d"):
            ax.scatter(g.groupby("bandwidth").kta.mean(), g.groupby("bandwidth").pr_auc.mean(), label=f"QSVM, d = {d} (λ sweep)")
        ax.set(xlabel="Centred KTA of K_train", ylabel="Test PR-AUC")
        ax.legend()
        fig.savefig(fig_dir / "fig9_kta.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    if conc is not None:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
        for lam, g in conc.groupby("bandwidth"):
            grp = g.groupby("d")
            _errorbar(a1, grp.offdiag_var.mean().index, grp.offdiag_var, label=f"λ = {lam}")
            _errorbar(a2, grp.effective_rank.mean().index, grp.effective_rank, label=f"λ = {lam}")
        a1.set(yscale="log", xlabel="Number of qubits d", ylabel="Off-diagonal variance of K")
        a2.set(yscale="log", xlabel="Number of qubits d", ylabel="Effective rank of K")
        a2.legend()
        fig.savefig(fig_dir / "fig5_concentration.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    heat = Path(results_dir) / "geometry_heatmap.npz"
    geo = frames.get("geometry")
    if heat.exists():
        h = np.load(heat)
        best = 0.01
        if geo is not None and "selected" in geo:
            best = geo[(geo.kernel == "zz") & (geo.selected == True)].bandwidth.mode().iloc[0]  # noqa: E712
        panels = [("Quantum, λ = 1", h["zz_1.0"]), (f"Quantum, λ = {best} (best)", h[f"zz_{best}"]),
                  (f"RBF, γ = {float(h['rbf_gamma']):g}", h["rbf"])]
        n_normal = int((h["y"] == 0).sum())
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        norm = matplotlib.colors.SymLogNorm(linthresh=0.01, vmin=-1, vmax=1)
        for ax, (title, K) in zip(axes, panels):
            im = ax.imshow(diagnostics.normalized_centered(K), cmap="RdBu_r", norm=norm)
            ax.axhline(n_normal - 0.5, color="k", lw=0.8)
            ax.axvline(n_normal - 0.5, color="k", lw=0.8)
            ax.set(title=title, xticks=[], yticks=[], xlabel="train sample (normal | fraud)")
        fig.colorbar(im, ax=axes, label="HKH / max|HKH|")
        fig.savefig(fig_dir / "fig6_kernels.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    if lc is not None:
        ds = sorted(lc.d.unique())
        fig, axes = plt.subplots(1, len(ds), figsize=(5 * len(ds), 3.6), sharey=True)
        for ax, d in zip(np.atleast_1d(axes), ds):
            for m in NAMES:
                grp = lc[(lc.d == d) & (lc.model == m)].groupby("n_train").pr_auc
                _errorbar(ax, grp.mean().index, grp, color=COLORS[m], label=NAMES[m])
            ax.set(xscale="log", title=f"d = {d}", xlabel="Training-set size", ylim=(0, 1))
        np.atleast_1d(axes)[0].set_ylabel("Test PR-AUC")
        np.atleast_1d(axes)[0].legend()
        fig.savefig(fig_dir / "fig7_learning_curves.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    if noise_df is not None:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
        for (kernel, d), g in noise_df.groupby(["kernel", "d"]):
            label = f"d = {d}, " + ("tuned λ" if kernel == "tuned" else "λ = 1")
            grp = g[g.noise == "shots"].groupby("shots").pr_auc
            _errorbar(a1, grp.mean().index, grp, label=label)
            dep = g[g.noise == "depolarizing"]
            if len(dep):
                grp = dep.groupby("eps").pr_auc
                _errorbar(a2, grp.mean().index, grp, label=label)
        a1.set(xscale="log", xlabel="Shots per kernel entry", ylabel="Test PR-AUC (QSVM)", ylim=(0, 1))
        a2.set(xlabel="Two-qubit depolarizing error rate ε", ylabel="Test PR-AUC (QSVM)", ylim=(0, 1))
        a1.legend()
        a2.legend()
        fig.savefig(fig_dir / "fig8_noise.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    if prev is not None:
        p = prev[prev.test == "prevalence"]
        d = max(p.d)
        grid = np.linspace(0, 1, 101)
        fig, ax = plt.subplots(figsize=(5, 4))
        for m in NAMES:
            curves = np.array(p[(p.d == d) & (p.model == m)].pr_curve.tolist())
            mu, sd = curves.mean(0), curves.std(0, ddof=1)
            ax.plot(grid, mu, color=COLORS[m], label=NAMES[m])
            ax.fill_between(grid, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1), color=COLORS[m], alpha=0.2)
        ax.axhline(p.positive_rate.mean(), ls=":", color="k", label="prevalence (random)")
        ax.set(xlabel="Recall", ylabel="Precision (interpolated)", xlim=(0, 1), ylim=(0, 1), title=f"d = {d}")
        ax.legend()
        fig.savefig(fig_dir / "fig10_pr_curves.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def build(cfg):
    rd = cfg["results_dir"]
    n_res = cfg["bootstrap_resamples"]
    names = ["main", "qubits", "bandwidth", "concentration", "geometry", "feature_map",
             "learning_curve", "prevalence", "noise", "pca_variance"]
    payloads = {n: _load(rd, n) for n in names}
    frames = {n: _frame(p) for n, p in payloads.items() if p}
    out = ["# Results\n", "All numbers are means ± standard deviations over seeds.\n"]
    if "pca_variance" in frames:
        table_pca(frames["pca_variance"], out)
    resources = _load(rd, "resources")
    if resources:
        table_resources(resources, out)
    if "main" in frames:
        table_main(frames["main"], frames.get("prevalence"), n_res, out)
        edge_summary(frames["main"], out)
    if "qubits" in frames:
        table_qubits(frames["qubits"], n_res, out)
    if "bandwidth" in frames:
        table_bandwidth(frames["bandwidth"], out)
    if "prevalence" in frames:
        table_prevalence(frames["prevalence"], out)
    if "noise" in frames:
        table_noise(frames["noise"], out)
    if "geometry" in frames:
        table_geometry(frames["geometry"], out)
    if "feature_map" in frames:
        table_feature_map(frames["feature_map"], n_res, out)
    if "learning_curve" in frames:
        table_learning_curve(frames["learning_curve"], n_res, out)
    figures(rd, cfg["figures_dir"], frames)
    path = Path(rd) / "report.md"
    path.write_text("\n".join(out) + "\n")
    return path
