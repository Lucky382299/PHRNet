"""Re-draw figures 4 and 6 of the paper from saved predictions (no re-training needed).

Usage:
    python scripts/make_figures_4_6.py --predictions paper_outputs/predictions.npz \
        --robustness results/robustness.csv --out figures

predictions.npz is written by the experiment notebook (it is ~50 MB and therefore not stored in this repository).
"""
import os, math, argparse
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, brier_score_loss, log_loss)

ap = argparse.ArgumentParser()
ap.add_argument("--predictions", default="paper_outputs/predictions.npz")
ap.add_argument("--robustness", default="results/robustness.csv")
ap.add_argument("--out", default="figures")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

CFG = dict(ALPHA_STRICT=0.02, ALPHA_MAIN=0.05, EPI_Q=0.95, FIXED_CUTS=(0.2, 0.4, 0.6, 0.8))
GROUPS = None

# ---- decision-layer functions (identical to the experiment code) ----
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))

def bin_entropy(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))

def fit_temperature(L, y):
    """Temperature scaling (one number T) fitted on the validation split."""
    Ts = np.exp(np.linspace(np.log(0.05), np.log(20.0), 200))
    losses = [log_loss(y, np.clip(sigmoid(L / T).mean(1), EPS, 1 - EPS), labels=[0, 1]) for T in Ts]
    return float(Ts[int(np.argmin(losses))])

def uncertainty(P, L=None):
    """P: [n, k] calibrated member probabilities, L: [n, k] member logits.
       Returns mean probability, total uncertainty (entropy of the mean) and
       epistemic uncertainty = standard deviation of the member logits (member disagreement).
       The logit-space spread is used because, for rare events (p ~ 0.01), mutual information
       is numerically tiny even when members disagree substantially."""
    pbar = P.mean(1)
    total = bin_entropy(pbar)
    if L is None:
        L = logit(P)
    epi = L.std(1) if L.shape[1] > 1 else np.zeros(len(pbar))
    return pbar, total, epi

def ece_score(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)

def risk_coverage(err, unc):
    order = np.argsort(unc, kind="stable")
    e = err[order]
    n = np.arange(1, len(e) + 1)
    return n / len(e), np.cumsum(e) / n

def aurc(err, unc):
    return float(risk_coverage(err, unc)[1].mean())

def classification_metrics(y, p, unc):
    pred = (p >= 0.5).astype(int)
    err = (pred != y).astype(int)
    m = dict(
        Accuracy=accuracy_score(y, pred),
        Precision=precision_score(y, pred, zero_division=0),
        Recall=recall_score(y, pred, zero_division=0),
        F1=f1_score(y, pred, zero_division=0),
        AUC=roc_auc_score(y, p),
        AUPRC=average_precision_score(y, p),
        Brier=brier_score_loss(y, p),
        NLL=log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1]),
        ECE=ece_score(y, p),
        AURC=aurc(err, unc),
    )
    m["ErrAUROC"] = roc_auc_score(err, unc) if 0 < err.sum() < len(err) else np.nan
    return m

def best_f1_threshold(y, p):
    from sklearn.metrics import precision_recall_curve
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    return float(th[int(np.nanargmax(f1))]) if len(th) else 0.5

def spec_threshold(y, p, spec=0.90):
    neg = np.sort(p[y == 0])
    return float(neg[min(int(np.ceil(spec * len(neg))), len(neg) - 1)]) if len(neg) else 0.5

def threshold_metrics(yt, pt, pv, yv):
    t1 = best_f1_threshold(yv, pv)
    t2 = spec_threshold(yv, pv, 0.90)
    pred = (pt >= t1).astype(int)
    tp = ((pred == 1) & (yt == 1)).sum(); tn = ((pred == 0) & (yt == 0)).sum()
    sens = tp / max((yt == 1).sum(), 1); spec = tn / max((yt == 0).sum(), 1)
    pred2 = (pt >= t2).astype(int)
    return dict(Thr_F1=t1, F1_tuned=f1_score(yt, pred, zero_division=0),
                Prec_tuned=precision_score(yt, pred, zero_division=0), Sens_tuned=sens, Spec_tuned=spec,
                BalAcc_tuned=(sens + spec) / 2,
                Sens_at_Spec90=float(((pred2 == 1) & (yt == 1)).sum() / max((yt == 1).sum(), 1)),
                Spec_at_Spec90=float(((pred2 == 0) & (yt == 0)).sum() / max((yt == 0).sum(), 1)))

# ---------- Patient-level Conformal Risk Control (Angelopoulos et al., ICLR 2024) ----------
# Rows of one patient are correlated, so exchangeability holds between PATIENTS, not hours.
# We therefore define one loss per calibration patient:
#   miss loss        l_j(tau) = fraction of patient j's high-risk hours with p <  tau   (non-decreasing in tau)
#   false-alarm loss l_j(tau) = fraction of patient j's low-risk  hours with p >= tau   (non-increasing in tau)
# and pick tau with the CRC rule  n/(n+1) * mean_j l_j(tau) + 1/(n+1) <= alpha.
# Guarantee: E[ l_new(tau_hat) ] <= alpha for a new, exchangeable patient.
def _patient_ids(g):
    _, inv = np.unique(g, return_inverse=True)
    return inv

def crc_low_threshold(p, y, groups, alpha):
    """Largest tau whose patient-averaged miss risk satisfies the CRC bound."""
    m = y == 1
    if m.sum() == 0:
        return 0.0
    pg = _patient_ids(groups[m]); pp = p[m]
    n = pg.max() + 1
    w = 1.0 / np.bincount(pg)[pg] / n                # each patient has total weight 1/n
    order = np.argsort(pp)
    cum = np.cumsum(w[order])                         # risk if tau is just above pp[order[i]]
    srt = pp[order]
    ok = n / (n + 1) * cum + 1 / (n + 1) <= alpha
    ok &= np.r_[srt[:-1] < srt[1:], True]             # only cut between distinct values (ties)
    if not ok.any():
        return 0.0                                    # nobody may be placed this low
    i = int(np.where(ok)[0].max())
    return float(np.nextafter(srt[i], np.inf))        # rows with p <= srt[i] are below tau

def crc_high_threshold(p, y, groups, alpha):
    """Smallest tau whose patient-averaged false-alarm risk satisfies the CRC bound."""
    m = y == 0
    if m.sum() == 0:
        return 1.0 + 1e-9
    pg = _patient_ids(groups[m]); pn = p[m]
    n = pg.max() + 1
    w = 1.0 / np.bincount(pg)[pg] / n
    order = np.argsort(-pn)
    cum = np.cumsum(w[order])                         # risk if tau == pn[order[i]]
    srt = pn[order]
    ok = n / (n + 1) * cum + 1 / (n + 1) <= alpha
    ok &= np.r_[srt[:-1] > srt[1:], True]             # only cut between distinct values (ties)
    if not ok.any():
        return 1.0 + 1e-9                             # nobody may be placed this high
    i = int(np.where(ok)[0].max())
    return float(srt[i])                              # rows with p >= srt[i] are above tau

def fit_tiers(p_cal, y_cal, g_cal, epi_cal, cfg):
    a_s, a_m = cfg["ALPHA_STRICT"], cfg["ALPHA_MAIN"]
    th = dict(tL1=crc_low_threshold(p_cal, y_cal, g_cal, a_s), tL2=crc_low_threshold(p_cal, y_cal, g_cal, a_m),
              tH2=crc_high_threshold(p_cal, y_cal, g_cal, a_m), tH1=crc_high_threshold(p_cal, y_cal, g_cal, a_s))
    # overlap guard: when the two sides cross, the higher (safer) tier wins
    th["eL1"] = min(th["tL1"], th["tH2"]); th["eL2"] = max(th["eL1"], min(th["tL2"], th["tH2"]))
    th["eH2"] = th["tH2"]; th["eH1"] = max(th["tH1"], th["tH2"])
    th["epi_q"] = float(np.quantile(epi_cal, cfg["EPI_Q"])) if np.any(epi_cal > 0) else np.inf
    return th

def assign_tiers(p, th, epi=None, override=True):
    """0 Very Low, 1 Low, 2 Moderate, 3 High, 4 Critical.
       Escalation (member disagreement above its calibration 95th percentile) only raises
       Very Low/Low to Moderate, so neither the miss nor the false-alarm guarantee is affected."""
    t = np.full(len(p), 2)
    t[p < th["eL2"]] = 1
    t[p < th["eL1"]] = 0
    t[p >= th["eH2"]] = 3
    t[p >= th["eH1"]] = 4
    flag = np.zeros(len(p), bool)
    if override and epi is not None:
        flag = epi > th["epi_q"]
        t = np.where(flag & (t < 2), 2, t)
    return t, flag

def fixed_tiers(p, cuts=None):
    """Naive fixed cut-points (no CRC, no escalation) - used as the 'without CRC' comparison."""
    return np.digitize(p, list(cuts if cuts is not None else CFG["FIXED_CUTS"]))

def patient_risk(mask_rows, y_rows, g_rows, cls):
    """Patient-averaged fraction of class-`cls` hours for which mask is True (the CRC loss on test)."""
    m = y_rows == cls
    if m.sum() == 0:
        return np.nan
    pg = _patient_ids(g_rows[m])
    return float(np.mean(np.bincount(pg, weights=mask_rows[m].astype(float)) / np.bincount(pg)))

def tier_metrics(y, tiers, g, prefix=""):
    pos, neg = y == 1, y == 0
    d = dict(Miss_VeryLow=float(np.mean(tiers[pos] == 0)), Miss_LeLow=float(np.mean(tiers[pos] <= 1)),
             FA_Critical=float(np.mean(tiers[neg] == 4)), FA_GeHigh=float(np.mean(tiers[neg] >= 3)),
             PMiss_VeryLow=patient_risk(tiers == 0, y, g, 1), PMiss_LeLow=patient_risk(tiers <= 1, y, g, 1),
             PFA_Critical=patient_risk(tiers == 4, y, g, 0), PFA_GeHigh=patient_risk(tiers >= 3, y, g, 0))
    # per-tier share of hours, observed high-risk rate (PPV) and share of all high-risk hours captured
    for t, nm in enumerate(["VL", "L", "M", "H", "C"]):
        sel = tiers == t
        d[f"Share_{nm}"] = float(sel.mean())
        d[f"Rate_{nm}"] = float(y[sel].mean()) if sel.any() else np.nan
        d[f"Capture_{nm}"] = float(sel[pos].mean()) if pos.any() else np.nan
    # patient-level alerting: septic patients with >=1 High/Critical hour inside their positive window,
    # and never-septic patients with >=1 High/Critical hour (patient-level false alarm)
    ug, inv = np.unique(g, return_inverse=True)
    hi = np.bincount(inv, weights=(tiers >= 3).astype(float)) > 0
    hi_pos = np.bincount(inv, weights=((tiers >= 3) & pos).astype(float)) > 0
    septic = np.bincount(inv, weights=pos.astype(float)) > 0
    d["PatDetect_High"] = float(hi_pos[septic].mean()) if septic.any() else np.nan
    d["PatFalseAlarm_High"] = float(hi[~septic].mean()) if (~septic).any() else np.nan
    return {prefix + k: v for k, v in d.items()}


# ---- figures ----
plt.rcParams.update({"savefig.dpi": 300, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
TIER_NAMES = ["Very Low", "Low", "Moderate", "High", "Critical"]
z = np.load(args.predictions)
y_all, g_all = z["y_all"].astype(int), z["groups"]
k = "s0|PHRNet (Proposed)|"
ic, it = z["s0_idx_cal"], z["s0_idx_test"]
pc, pt = z[k + "p_cal"].astype(float), z[k + "p_test"].astype(float)
ec, et = z[k + "e_cal"].astype(float), z[k + "e_test"].astype(float)
yc, yt, gc, gt = y_all[ic], y_all[it], g_all[ic], g_all[it]
th = fit_tiers(pc, yc, gc, ec, CFG)
tiers, flag = assign_tiers(pt, th, epi=et)
print({kk: round(v, 4) for kk, v in th.items()})
print("shares", np.round(np.bincount(tiers, minlength=5) / len(tiers), 4), "esc", flag.mean().round(4))

def label_below(ax, s):
    ax.text(0.5, -0.30, s, transform=ax.transAxes, ha="center", va="top", fontsize=12, fontweight="bold")

# ---------------- Figure 4: tiers ----------------
fig, ax = plt.subplots(1, 3, figsize=(12, 3.9))
cnt = np.bincount(tiers, minlength=5); esc = np.bincount(tiers[flag], minlength=5)
ax[0].bar(range(5), cnt - esc, color="tab:blue", label="Confident")
ax[0].bar(range(5), esc, bottom=cnt - esc, color="orange", hatch="//", label="Escalated (disagreement)")
for i, c in enumerate(cnt):
    ax[0].text(i, c, f"{100 * c / len(tiers):.1f}%", ha="center", va="bottom", fontsize=8)
ax[0].set_xticks(range(5)); ax[0].set_xticklabels(TIER_NAMES, rotation=30)
ax[0].set_ylabel("Number of patient-hours"); ax[0].legend(fontsize=7)
rate = [yt[tiers == t].mean() if (tiers == t).any() else 0 for t in range(5)]
capt = [np.mean(tiers[yt == 1] == t) for t in range(5)]
ax[1].bar(np.arange(5) - 0.2, rate, 0.4, color="tab:blue", label="High-risk rate in tier")
ax[1].bar(np.arange(5) + 0.2, capt, 0.4, color="tab:orange", hatch="..", label="Share of all high-risk hours")
ax[1].axhline(yt.mean(), color="k", ls=":", lw=1, label=f"Prevalence ({100 * yt.mean():.1f}%)")
ax[1].set_xticks(range(5)); ax[1].set_xticklabels(TIER_NAMES, rotation=30); ax[1].set_ylabel("Fraction"); ax[1].legend(fontsize=7)
bins = np.logspace(np.log10(max(pt.min(), 1e-5)), 0, 60)
ax[2].hist(pt[yt == 0], bins=bins, alpha=0.6, label="Low-risk hours", density=True)
ax[2].hist(pt[yt == 1], bins=bins, alpha=0.6, label="High-risk hours", density=True, histtype="stepfilled", hatch="//")
ymax = ax[2].get_ylim()[1]
for key, ls, lab, f in [("eL1", ":", r"$\tau'_{L1}$", 0.92), ("eL2", "--", r"$\tau'_{L2}$", 0.72), ("eH2", "--", r"$\tau_{H2}$", 0.92), ("eH1", ":", r"$\tau_{H1}$", 0.72)]:
    ax[2].axvline(th[key], color="k", ls=ls, lw=1); ax[2].text(th[key] * 1.06, ymax * f, lab, fontsize=9)
ax[2].set_xscale("log"); ax[2].set_xlabel("Predicted risk probability (log scale)"); ax[2].set_ylabel("Density"); ax[2].legend(fontsize=7)
label_below(ax[0], "(a)"); label_below(ax[1], "(b)"); label_below(ax[2], "(c)")
plt.subplots_adjust(wspace=0.3, bottom=0.28)
plt.savefig(os.path.join(args.out, "fig_tiers_alerts.png"), bbox_inches="tight"); plt.close()

# ---------------- Figure 6: robustness ----------------
ROB = pd.read_csv(args.robustness)
rn = ROB[ROB.kind == "noise"]
models = ["PHRNet (Proposed)", "Deep Ensemble (5 MLPs)", "Attention-MLP + MC dropout", "FT-Transformer", "XGBoost"]
mk = ["o-", "s--", "^-.", "d:", "v-"]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.7))
for m, st in zip(models, mk):
    r = rn[rn.model == m]
    ax[0].plot(r.level, r.AUC, st, ms=4, label=m); ax[1].plot(r.level, r.AUPRC, st, ms=4, label=m)
r = rn[rn.model == "PHRNet (Proposed)"]
ax[2].plot(r.level, r.PMiss_LeLow, "o-", label="Miss risk (Very Low/Low)")
ax[2].plot(r.level, r.PFA_GeHigh, "s--", label="False-alarm risk (High/Critical)")
ax[2].plot(r.level, r.Override_Rate, "^-.", label="Escalation rate")
ax[2].axhline(0.05, color="gray", ls=":", label="Target (5%)")
for a_, yl in zip(ax, ["AUC", "AUPRC", "Rate"]):
    a_.set_xlabel("Gaussian sensor noise (std, z-units)"); a_.set_ylabel(yl)
ax[0].legend(fontsize=7); ax[2].legend(fontsize=7)
label_below(ax[0], "(a)"); label_below(ax[1], "(b)"); label_below(ax[2], "(c)")
plt.subplots_adjust(wspace=0.3, bottom=0.28)
plt.savefig(os.path.join(args.out, "fig_noise_robustness.png"), bbox_inches="tight"); plt.close()
print("done")
