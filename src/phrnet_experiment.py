"""PHRNet - full experiment as one script (same code as notebooks/PHRNet_kaggle.ipynb).

On Kaggle: paste into one code cell, or each "# %% CELL" block into its own cell.
"""

# %% ======================== CELL: c01_setup.py ========================
# ============================================================
# 1. SETUP AND SETTINGS  (you normally only change QUICK_RUN)
# ============================================================
import os, re, glob, json, math, time, copy, random, warnings, zipfile
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, brier_score_loss,
                             log_loss, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
warnings.filterwarnings("ignore")

# QUICK_RUN = True  -> fast test (2 seeds, few epochs, ~10-15 min). Use it once to check everything works.
# QUICK_RUN = False -> full experiment for the paper (10 seeds, ~3-4 hours on a Kaggle GPU).
QUICK_RUN = False

# Which dataset to use:
#   "sepsis" -> PhysioNet/CinC 2019 sepsis data (real ICU vital signs, Kaggle copy). MAIN dataset of the paper.
#   "vitals" -> Human Vital Signs Dataset 2024 (labels follow simple rules; secondary / sanity-check dataset).
DATASET = "sepsis"
# Leave None: the notebook finds the right file in /kaggle/input automatically.
DATA_PATH = None
MAX_PATIENTS = 20000   # sepsis only: random subset of patients (run time ~2.5 h on a T4); None = all 40k

OUT = "/kaggle/working/paper_outputs" if os.path.isdir("/kaggle/working") else "./paper_outputs"
os.makedirs(OUT, exist_ok=True)

# ---------------- OPTIONAL: copy results to Google Drive ----------------
# Kaggle cannot mount Google Drive, so files are uploaded with the Drive API.
# One-time setup (see README, section "Saving results to Google Drive"):
#   Kaggle notebook -> Add-ons -> Secrets -> add GDRIVE_CLIENT_ID, GDRIVE_CLIENT_SECRET,
#   GDRIVE_REFRESH_TOKEN and GDRIVE_FOLDER_ID (ID of your Drive folder), then set SAVE_TO_GDRIVE = True.
SAVE_TO_GDRIVE = False
_GDRIVE = None

def save_to_gdrive(path):
    """Upload (or overwrite) one file in the configured Google Drive folder. Never stops the experiment."""
    global _GDRIVE
    if not SAVE_TO_GDRIVE or not os.path.isfile(path):
        return
    try:
        if _GDRIVE is None:
            from kaggle_secrets import UserSecretsClient
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
            sec = UserSecretsClient()
            creds = Credentials(None, refresh_token=sec.get_secret("GDRIVE_REFRESH_TOKEN"),
                                client_id=sec.get_secret("GDRIVE_CLIENT_ID"),
                                client_secret=sec.get_secret("GDRIVE_CLIENT_SECRET"),
                                token_uri="https://oauth2.googleapis.com/token",
                                scopes=["https://www.googleapis.com/auth/drive.file"])
            _GDRIVE = (build("drive", "v3", credentials=creds, cache_discovery=False), sec.get_secret("GDRIVE_FOLDER_ID"))
        from googleapiclient.http import MediaFileUpload
        svc, folder = _GDRIVE
        name = os.path.basename(path)
        q = f"name = '{name}' and '{folder}' in parents and trashed = false"
        old = svc.files().list(q=q, fields="files(id)").execute().get("files", [])
        media = MediaFileUpload(path, resumable=True)
        if old:
            svc.files().update(fileId=old[0]["id"], media_body=media).execute()
        else:
            svc.files().create(body={"name": name, "parents": [folder]}, media_body=media, fields="id").execute()
        print("  saved to Google Drive:", name)
    except Exception as e:
        print("  (Google Drive upload skipped:", repr(e)[:200], ")")

CFG = dict(
    N_SEEDS     = 2 if QUICK_RUN else 10,   # independent runs (different splits + inits)
    N_SEEDS_ABL = 1 if QUICK_RUN else 3,    # runs for the ablation study
    EPOCHS      = 4 if QUICK_RUN else 40,
    PATIENCE    = 3 if QUICK_RUN else 6,    # early stopping on validation log-loss
    BATCH       = 1024,
    LR          = 1e-3,
    WD          = 1e-4,
    # --- PHRNet backbone ---
    D_TOKEN     = 64,     # size of each feature token
    N_LAYERS    = 3,      # transformer layers
    N_HEADS     = 8,      # attention heads
    HIDDEN      = 256,    # width of the ensemble head
    K_ENS       = 16,     # number of BatchEnsemble (TabM) members
    DROPOUT     = 0.1,
    N_FREQ      = 16,     # periodic embedding frequencies
    FREQ_SIGMA  = 0.3,
    # --- baselines ---
    MC_T        = 30,     # MC-dropout passes (old PHRNet baseline + ablation)
    DE_M        = 5,      # deep-ensemble size
    USE_TABPFN  = False,  # needs Internet ON + `pip install tabpfn`; optional
    TABPFN_N    = 10000,
    # --- conformal / risk control ---
    ALPHA_STRICT = 0.02,  # Very Low / Critical tiers: <= 2% expected error (patient-level CRC)
    ALPHA_MAIN   = 0.05,  # Low / High tiers: <= 5% expected error (patient-level CRC)
    EPI_Q        = 0.95,  # epistemic (member-disagreement) escalation quantile
    # naive fixed cut-points, used only as the comparison "without CRC"
    FIXED_CUTS   = (0.20, 0.40, 0.60, 0.80),
)

TIER_NAMES  = ["Very Low", "Low", "Moderate", "High", "Critical"]
ALERT_NAMES = ["No Alert", "Routine Monitoring", "Doctor Review", "Urgent Alert", "Emergency Alert"]
PROPOSED = "PHRNet (Proposed)"

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight"})
print("Outputs will be saved to:", OUT)
print("QUICK_RUN =", QUICK_RUN)


# %% ======================== CELL: c02_utils.py ========================
# ============================================================
# 2. METRICS, CALIBRATION, PATIENT-LEVEL CONFORMAL RISK CONTROL, RISK TIERS
# ============================================================
EPS = 1e-6

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

def evaluate_from_logits(L, y_all, sp, cfg, T=None, override=True, groups=None):
    """Full evaluation of one model given member logits for val/cal/test."""
    groups = GROUPS if groups is None else groups
    if groups is None:
        groups = np.arange(len(y_all))
    if T is None:
        T = fit_temperature(L["val"], y_all[sp["val"]])
    P = {s: sigmoid(L[s] / T) for s in L}
    yt, yc = y_all[sp["test"]], y_all[sp["cal"]]
    gt, gc = groups[sp["test"]], groups[sp["cal"]]
    pb, tot, ep = uncertainty(P["test"], L["test"] / T)
    m = classification_metrics(yt, pb, tot)
    pv, yv = P["val"].mean(1), y_all[sp["val"]]
    m.update(threshold_metrics(yt, pb, pv, yv))      # threshold chosen on validation (rare events)
    m["ECE_raw"] = ece_score(yt, sigmoid(L["test"]).mean(1))
    m["Temperature"] = T
    pcal, _, epcal = uncertainty(P["cal"], L["cal"] / T)
    th = fit_tiers(pcal, yc, gc, epcal, cfg)
    tiers, flag = assign_tiers(pb, th, epi=ep, override=override)
    m.update(Override_Rate=float(flag.mean()), MeanEpi=float(ep.mean()),
             **{k: v for k, v in th.items() if k != "epi_q"})
    # does escalation catch would-be misses? high-risk rate among escalated low-tier hours
    t0, _ = assign_tiers(pb, th, override=False)
    esc = flag & (t0 < 2)
    m["Esc_Rate_Pos"] = float(yt[esc].mean()) if esc.any() else np.nan
    m["NoEsc_LowTier_Rate_Pos"] = float(yt[(t0 < 2) & ~esc].mean()) if ((t0 < 2) & ~esc).any() else np.nan
    m.update(tier_metrics(yt, tiers, gt))
    m.update(tier_metrics(yt, fixed_tiers(pb), gt, prefix="Fixed_"))
    m["Prevalence"] = float(yt.mean())
    art = dict(T=T, P=P, pbar=pb, total=tot, epi=ep, tiers=tiers, flag=flag, th=th, y=yt, g=gt)
    return {k: float(v) for k, v in m.items()}, art

# ---------- LaTeX helpers ----------
def fmt_ms(vals, nd=4):
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return "--"
    if len(v) == 1:
        return f"{v[0]:.{nd}f}"
    return f"{v.mean():.{nd}f}$\\pm${v.std(ddof=1):.{nd}f}"

def latex_escape(s):
    return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")

def build_table(rows, metrics, higher, caption, label, star=True, nd=4, colnames=None):
    """rows: list of (name, {metric: list_of_values}); bold = best mean per column."""
    means = {mt: [np.nanmean(r[1][mt]) if len(r[1].get(mt, [])) else np.nan for r in rows] for mt in metrics}
    best = {}
    for mt in metrics:
        arr = np.array(means[mt], float)
        if np.all(np.isnan(arr)):
            best[mt] = -1
        else:
            best[mt] = int(np.nanargmax(arr) if higher.get(mt, True) else np.nanargmin(arr))
    env = "table*" if star else "table"
    cols = colnames or metrics
    s = [f"\\begin{{{env}}}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}", "\\renewcommand{\\arraystretch}{1.15}",
         ("" if star else "\\resizebox{\\columnwidth}{!}{") + "\\begin{tabular}{l" + "c" * len(metrics) + "}", "\\toprule",
         "\\textbf{Model} & " + " & ".join(f"\\textbf{{{c}}}" for c in cols) + " \\\\", "\\midrule"]
    for i, (name, d) in enumerate(rows):
        cells = []
        for mt in metrics:
            c = fmt_ms(d.get(mt, []), nd)
            cells.append(f"\\textbf{{{c}}}" if i == best[mt] and c != "--" else c)
        s.append(latex_escape(name) + " & " + " & ".join(cells) + " \\\\")
    s += ["\\bottomrule", "\\end{tabular}" + ("" if star else "}"), f"\\end{{{env}}}"]
    return "\n".join(s)

def write(name, text):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(text)
    print("saved", name)

def savefig(name):
    plt.savefig(os.path.join(OUT, name))
    plt.show()
    plt.close()
    print("saved", name)


# %% ======================== CELL: c03_data.py ========================
# ============================================================
# 3. LOAD DATA + DATA AUDIT
#    sepsis: PhysioNet/CinC 2019 (hourly ICU vital signs, label = SepsisLabel)
#    vitals: Human Vital Signs Dataset 2024 (one row per record)
# ============================================================
def _cols(f, sep=","):
    try:
        return [c.strip() for c in pd.read_csv(f, sep=sep, nrows=0).columns]
    except Exception:
        return []

def _find(col_list, *names):
    low = {c.lower(): c for c in col_list}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None

GROUPS = None   # patient id per row (None = every row is its own patient)

if DATASET == "sepsis":
    csvs = [DATA_PATH] if DATA_PATH else glob.glob("/kaggle/input/**/*.csv", recursive=True)
    csvs = [f for f in csvs if _find(_cols(f), "SepsisLabel")]
    if csvs:
        path = max(csvs, key=os.path.getsize)
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        pid = _find(df.columns, "Patient_ID", "patient_id", "PatientID", "pid")
        if pid is None:
            raise ValueError("Sepsis CSV has no patient-id column; please tell Claude the column names: " + str(list(df.columns)))
    else:   # raw PhysioNet .psv files (one file per patient)
        psvs = glob.glob("/kaggle/input/**/*.psv", recursive=True)
        if not psvs:
            raise FileNotFoundError("No sepsis data found. Add Input -> search 'Prediction of Sepsis' or 'PhysioNet 2019 sepsis'.")
        rs_ = np.random.RandomState(0)
        if MAX_PATIENTS and len(psvs) > MAX_PATIENTS:
            psvs = list(rs_.choice(psvs, MAX_PATIENTS, replace=False))
        parts = []
        for f in psvs:
            d = pd.read_csv(f, sep="|"); d["Patient_ID"] = os.path.basename(f).split(".")[0]; parts.append(d)
        df = pd.concat(parts, ignore_index=True)
        path, pid = os.path.dirname(psvs[0]), "Patient_ID"
    print("File:", path, "| shape:", df.shape)
    # random subset of patients (keeps the run time manageable)
    pids = df[pid].unique()
    if MAX_PATIENTS and len(pids) > MAX_PATIENTS:
        keep = np.random.RandomState(0).choice(pids, MAX_PATIENTS, replace=False)
        df = df[df[pid].isin(keep)]
    hour = _find(df.columns, "ICULOS", "Hour")
    df = df.sort_values([pid, hour]).reset_index(drop=True)
    target_col = _find(df.columns, "SepsisLabel")
    VITALS = [c for c in ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"] if c in df.columns]
    DEMO = [c for c in ["Age", "Gender", "HospAdmTime", "ICULOS"] if c in df.columns]
    g = df.groupby(pid)
    df[VITALS] = g[VITALS].ffill()                       # last observation carried forward (per patient)
    g = df.groupby(pid)
    feats = df[VITALS + DEMO].copy()
    for v in VITALS:                                     # simple temporal context from the last 6 hours
        feats[f"{v}_mean6h"] = g[v].rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)
        feats[f"{v}_delta6h"] = df[v] - g[v].shift(6)
    X_all = feats.reset_index(drop=True)
    y_all = df[target_col].astype(int).values
    GROUPS = df[pid].astype(str).values
    drop_cols = [c for c in df.columns if c not in X_all.columns]
    DATA_NAME = "PhysioNet/CinC 2019 sepsis"
else:
    def find_csv():
        files = glob.glob("/kaggle/input/**/*.csv", recursive=True)
        if not files:
            raise FileNotFoundError("No CSV found in /kaggle/input. Click 'Add Input' and add the dataset, or set DATA_PATH.")
        return max(files, key=os.path.getsize)
    path = DATA_PATH or find_csv()
    df = pd.read_csv(path)
    print("File:", path, "| shape:", df.shape)
    target_col = next(c for c in df.columns if "risk" in c.lower())
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    yraw = df[target_col]
    if not pd.api.types.is_numeric_dtype(yraw):
        y_all = yraw.astype(str).str.lower().str.contains("high").astype(int).values
    else:
        y_all = (yraw.values > 0).astype(int)
    def _is_id_col(c):
        toks = re.split(r"[\s_\-]+", c.lower())
        return any(t in ("id", "timestamp", "time", "date", "datetime") for t in toks)
    drop_cols = [target_col] + [c for c in df.columns if c != target_col and _is_id_col(c)]
    X_all = df.drop(columns=drop_cols)
    DATA_NAME = "Human Vital Signs 2024"

cat_cols = [c for c in X_all.columns if (not pd.api.types.is_numeric_dtype(X_all[c])) or X_all[c].nunique() <= 5]
num_cols = [c for c in X_all.columns if c not in cat_cols]
FEATURES = num_cols + cat_cols
n_dup = int(X_all.duplicated().sum())

print(f"\nDataset: {DATA_NAME} | rows: {len(X_all):,} | patients: {len(np.unique(GROUPS)) if GROUPS is not None else len(X_all):,}")
print(f"Positives: {y_all.mean():.4f}")
print("Dropped columns:", drop_cols)
print("Numerical:", num_cols)
print("Categorical:", cat_cols)
print("Missing fraction per column (top):\n", X_all.isna().mean().sort_values(ascending=False).head(10).round(3))

DATA_INFO = dict(name=DATA_NAME, file=os.path.basename(str(path)), n_rows=int(len(X_all)),
                 n_patients=int(len(np.unique(GROUPS))) if GROUPS is not None else int(len(X_all)),
                 n_num=len(num_cols), n_cat=len(cat_cols), prevalence=float(y_all.mean()),
                 n_duplicates=n_dup, features=FEATURES)


# %% ======================== CELL: c04_prep.py ========================
# ============================================================
# 4. PREPROCESSING (fitted on the training split only) + SPLITS
#    70% train / 10% validation / 10% calibration / 10% test
# ============================================================
class Prep:
    def __init__(self, num_cols, cat_cols):
        self.num_cols, self.cat_cols = num_cols, cat_cols

    def fit(self, X):
        Xn = X[self.num_cols]
        self.med = Xn.median().fillna(0.0)
        Xn = Xn.fillna(self.med)
        self.mu, self.sd = Xn.mean(), Xn.std().replace(0, 1.0).fillna(1.0)
        self.maps = {c: {v: i + 1 for i, v in enumerate(sorted(X[c].astype(str).unique()))} for c in self.cat_cols}
        self.cards = [len(self.maps[c]) + 1 for c in self.cat_cols]   # index 0 = unseen category
        return self

    def transform(self, X):
        xn = ((X[self.num_cols].fillna(self.med) - self.mu) / self.sd).clip(-5, 5).values.astype(np.float32)
        if self.cat_cols:
            xc = np.stack([X[c].astype(str).map(self.maps[c]).fillna(0).astype(int).values
                           for c in self.cat_cols], 1).astype(np.int64)
            oh = [np.eye(card, dtype=np.float32)[xc[:, j]][:, 1:] for j, card in enumerate(self.cards)]
            xo = np.concatenate([xn] + oh, 1).astype(np.float32)
        else:
            xc = np.zeros((len(X), 0), np.int64)
            xo = xn.copy()
        return xn, xc, xo

def make_split(y, seed, groups=None):
    """Stratified 70/10/10/10 split. With patient ids, all rows of a patient stay in ONE split (no leakage)."""
    groups = GROUPS if groups is None else groups
    if groups is None:
        idx = np.arange(len(y))
        tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
        va, rest = train_test_split(tmp, test_size=2 / 3, stratify=y[tmp], random_state=seed)
        ca, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)
        return dict(train=tr, val=va, cal=ca, test=te)
    ug, inv = np.unique(groups, return_inverse=True)
    gy = np.zeros(len(ug), int); np.maximum.at(gy, inv, y)       # patient ever positive?
    gi = np.arange(len(ug))
    tr, tmp = train_test_split(gi, test_size=0.30, stratify=gy, random_state=seed)
    va, rest = train_test_split(tmp, test_size=2 / 3, stratify=gy[tmp], random_state=seed)
    ca, te = train_test_split(rest, test_size=0.5, stratify=gy[rest], random_state=seed)
    rows = lambda gsel: np.where(np.isin(inv, gsel))[0]
    return dict(train=rows(tr), val=rows(va), cal=rows(ca), test=rows(te))

# ---- Data audit: how "easy" are the labels? (reported honestly in the paper) ----
_sp = make_split(y_all, 0)
_pp = Prep(num_cols, cat_cols).fit(X_all.iloc[_sp["train"]])
_, _, _xo = _pp.transform(X_all)
_oh_names = num_cols + [f"{c}={v}" for c in cat_cols for v in sorted(_pp.maps[c])]
AUDIT = {}
for depth in (1, 2, 3):
    dt = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(_xo[_sp["train"]], y_all[_sp["train"]])
    AUDIT[f"tree_depth{depth}_acc"] = float(accuracy_score(y_all[_sp["test"]], dt.predict(_xo[_sp["test"]])))
for depth in (1, 2, 3):
    dt = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(_xo[_sp["train"]], y_all[_sp["train"]])
    AUDIT[f"tree_depth{depth}_auc"] = float(roc_auc_score(y_all[_sp["test"]], dt.predict_proba(_xo[_sp["test"]])[:, 1]))
print("Data audit - tiny decision trees:", AUDIT)
dt3 = DecisionTreeClassifier(max_depth=3, random_state=0).fit(_xo[_sp["train"]], y_all[_sp["train"]])
print(export_text(dt3, feature_names=_oh_names))
corr = pd.Series({f: abs(np.nan_to_num(np.corrcoef(_xo[:, j], y_all)[0, 1])) for j, f in enumerate(_oh_names)}).sort_values(ascending=False)
print("Absolute correlation with label:\n", corr.round(3))
DATA_INFO.update(AUDIT)


# %% ======================== CELL: c05_models.py ========================
# ============================================================
# 5. MODELS (PyTorch)
#    PHRNet = periodic feature tokens -> Transformer (feature-to-feature attention)
#                -> TabM-style BatchEnsemble head with K members
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda"
print("Device:", DEVICE, "| torch", torch.__version__)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

class LinearTokenizer(nn.Module):          # FT-Transformer style: e_j = x_j * w_j + b_j
    def __init__(self, n_num, d):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_num, d) * d ** -0.5)
        self.b = nn.Parameter(torch.randn(n_num, d) * d ** -0.5)
    def forward(self, x):
        return x[..., None] * self.W + self.b

class PeriodicTokenizer(nn.Module):        # PLR embeddings (Gorishniy et al., 2022)
    def __init__(self, n_num, d, n_freq, sigma):
        super().__init__()
        self.c = nn.Parameter(torch.randn(n_num, n_freq) * sigma)
        self.W = nn.Parameter(torch.randn(n_num, 2 * n_freq, d) * (2 * n_freq) ** -0.5)
        self.b = nn.Parameter(torch.zeros(n_num, d))
    def forward(self, x):
        v = 2 * math.pi * self.c[None] * x[..., None]
        v = torch.cat([torch.sin(v), torch.cos(v)], -1)
        return F.relu(torch.einsum("bnf,nfd->bnd", v, self.W) + self.b)

class CatTokenizer(nn.Module):
    def __init__(self, cards, d):
        super().__init__()
        self.register_buffer("off", torch.tensor([0] + list(np.cumsum(cards)[:-1]), dtype=torch.long))
        self.emb = nn.Embedding(int(sum(cards)), d)
    def forward(self, xc):
        return self.emb(xc + self.off[None])

class Block(nn.Module):                    # pre-norm transformer block that can return attention
    def __init__(self, d, h, drop):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(drop)
    def forward(self, x, need_w=False):
        z = self.n1(x)
        a, w = self.att(z, z, z, need_weights=need_w, average_attn_weights=True)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.n2(x)))
        return x, w

class BELinear(nn.Module):                 # BatchEnsemble layer: W shared, r/s/b per member
    def __init__(self, k, din, dout, random_sign=False):
        super().__init__()
        self.lin = nn.Linear(din, dout, bias=False)
        r = (torch.randint(0, 2, (k, din)).float() * 2 - 1) if random_sign else torch.ones(k, din)
        self.r = nn.Parameter(r)
        self.s = nn.Parameter(torch.ones(k, dout))
        self.b = nn.Parameter(torch.zeros(k, dout))
    def forward(self, z):                  # z: [B, k, din]
        return self.lin(z * self.r) * self.s + self.b

class TabMHead(nn.Module):
    def __init__(self, k, din, hidden, drop, n_layers=2):
        super().__init__()
        self.k = k
        dims = [din] + [hidden] * n_layers
        # random-sign adapters in every layer keep the members diverse (better epistemic uncertainty)
        self.layers = nn.ModuleList([BELinear(k, dims[i], dims[i + 1], random_sign=True) for i in range(n_layers)])
        self.drop = nn.Dropout(drop)
        self.out_w = nn.Parameter(torch.randn(k, hidden) * hidden ** -0.5)
        self.out_b = nn.Parameter(torch.zeros(k))
    def forward(self, h):                  # h: [B, din] -> logits [B, k]
        z = h[:, None, :].expand(-1, self.k, -1)
        for l in self.layers:
            z = self.drop(F.relu(l(z)))
        return (z * self.out_w[None]).sum(-1) + self.out_b

class PHRNet2(nn.Module):
    def __init__(self, n_num, cards, cfg, k=None, use_attn=True, tokenizer="periodic", head="tabm"):
        super().__init__()
        d, drop = cfg["D_TOKEN"], cfg["DROPOUT"]
        self.num_tok = (PeriodicTokenizer(n_num, d, cfg["N_FREQ"], cfg["FREQ_SIGMA"])
                        if tokenizer == "periodic" else LinearTokenizer(n_num, d))
        self.cat_tok = CatTokenizer(cards, d) if len(cards) else None
        n_tok = n_num + len(cards)
        self.use_attn = use_attn
        if use_attn:
            self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            self.blocks = nn.ModuleList([Block(d, cfg["N_HEADS"], drop) for _ in range(cfg["N_LAYERS"])])
            self.norm = nn.LayerNorm(d)
        else:                              # ablation: no feature-to-feature attention
            self.flat = nn.Sequential(nn.Flatten(), nn.Linear(n_tok * d, d), nn.ReLU(), nn.Dropout(drop))
        k = cfg["K_ENS"] if k is None else k
        self.head = TabMHead(k, d, cfg["HIDDEN"], drop) if head == "tabm" else nn.Linear(d, 1)

    def encode(self, b, return_attn=False):
        t = self.num_tok(b["xn"])
        if self.cat_tok is not None:
            t = torch.cat([t, self.cat_tok(b["xc"])], 1)
        if not self.use_attn:
            return self.flat(t), None
        x = torch.cat([self.cls.expand(len(t), -1, -1), t], 1)
        w = None
        for i, blk in enumerate(self.blocks):
            x, w = blk(x, need_w=(return_attn and i == len(self.blocks) - 1))
        return self.norm(x[:, 0]), (w[:, 0, 1:] if return_attn else None)

    def forward(self, b):
        return self.head(self.encode(b)[0])

    def cls_attention(self, b):
        return self.encode(b, return_attn=True)[1]

class MLPNet(nn.Module):
    def __init__(self, din, hidden=256, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hidden), nn.ReLU(), nn.Dropout(drop),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden, 1))
    def forward(self, b):
        return self.net(b["xo"])

class PHRNetV1(nn.Module):                 # the ORIGINAL PHRNet (Eqs. 3-6 of the old manuscript)
    def __init__(self, din, hidden=128, drop=0.2):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(din, 256), nn.ReLU(), nn.Dropout(drop),
                               nn.Linear(256, hidden), nn.ReLU(), nn.Dropout(drop))
        self.Wa, self.Wo = nn.Linear(hidden, hidden), nn.Linear(hidden, 1)
    def forward(self, b):
        h = self.f(b["xo"])
        return self.Wo(torch.softmax(self.Wa(h), -1) * h)


# %% ======================== CELL: c06_train.py ========================
# ============================================================
# 6. TRAINING AND PREDICTION HELPERS
# ============================================================
def to_device(xn, xc, xo, y):
    return dict(xn=torch.tensor(xn, device=DEVICE), xc=torch.tensor(xc, device=DEVICE),
                xo=torch.tensor(xo, device=DEVICE), y=torch.tensor(y, dtype=torch.float32, device=DEVICE))

def take(D, idx):
    return {k: v[idx] for k, v in D.items()}

def _batches(n, bs, shuffle):
    order = torch.randperm(n, device=DEVICE) if shuffle else torch.arange(n, device=DEVICE)
    for i in range(0, n, bs):
        yield order[i:i + bs]

@torch.no_grad()
def predict_torch(model, B, mc_T=0, bs=4096):
    """Returns member logits [n, k]. With mc_T > 0, dropout stays ON (MC dropout) and passes are stacked."""
    model.eval()
    if mc_T:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
    n = len(B["xn"])
    out = []
    for idx in _batches(n, bs, False):
        b = take(B, idx)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=AMP):
            if mc_T:
                l = torch.cat([model(b) for _ in range(mc_T)], 1)
            else:
                l = model(b)
        out.append(l.float().cpu())
    model.eval()
    return torch.cat(out).numpy()

def train_torch(model, D, sp, cfg, seed, verbose=False):
    set_seed(seed)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["LR"], weight_decay=cfg["WD"])
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    tr = torch.tensor(sp["train"], device=DEVICE)
    Bva = take(D, torch.tensor(sp["val"], device=DEVICE))
    yva = Bva["y"].cpu().numpy()
    best, best_state, bad = np.inf, None, 0
    for ep in range(cfg["EPOCHS"]):
        model.train()
        for idx in _batches(len(tr), cfg["BATCH"], True):
            b = take(D, tr[idx])
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=AMP):
                logits = model(b)
            loss = F.binary_cross_entropy_with_logits(logits.float(), b["y"][:, None].expand_as(logits))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        pv = np.clip(sigmoid(predict_torch(model, Bva)).mean(1), EPS, 1 - EPS)
        vl = log_loss(yva, pv, labels=[0, 1])
        if verbose:
            print(f"   epoch {ep + 1:3d}  val log-loss {vl:.5f}")
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg["PATIENCE"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


# %% ======================== CELL: c07_pipeline.py ========================
# ============================================================
# 7. ALL MODELS: FIT + PREDICT, AND ONE FULL RUN PER SEED
# ============================================================
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

MODELS = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost",
          "MLP", "Attention-MLP + MC dropout", "Deep Ensemble (5 MLPs)", "FT-Transformer", PROPOSED]
if CFG["USE_TABPFN"]:
    MODELS.insert(-1, "TabPFN-v2")
TORCH_MODELS = {"MLP", "Attention-MLP + MC dropout", "Deep Ensemble (5 MLPs)", "FT-Transformer", PROPOSED}

def build_torch(name, n_num, cards, d_oh, cfg, **kw):
    if name == "MLP":
        return MLPNet(d_oh)
    if name == "Attention-MLP + MC dropout":
        return PHRNetV1(d_oh)
    if name == "FT-Transformer":
        return PHRNet2(n_num, cards, cfg, tokenizer="linear", head="linear")
    if name == PROPOSED:
        return PHRNet2(n_num, cards, cfg, **kw)
    raise ValueError(name)

def mc_passes(name, cfg):
    return cfg["MC_T"] if name == "Attention-MLP + MC dropout" else 0

def fit_model(name, seed, D, Xo, y, sp, meta, cfg):
    tr, va = sp["train"], sp["val"]
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=3000).fit(Xo[tr], y[tr])
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed).fit(Xo[tr], y[tr])
    if name == "XGBoost":
        m = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8,
                              colsample_bytree=0.8, tree_method="hist",
                              device="cuda" if DEVICE.type == "cuda" else "cpu",
                              early_stopping_rounds=100, eval_metric="logloss", random_state=seed)
        return m.fit(Xo[tr], y[tr], eval_set=[(Xo[va], y[va])], verbose=False)
    if name == "LightGBM":
        m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=63, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
        return m.fit(Xo[tr], y[tr], eval_set=[(Xo[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    if name == "CatBoost":
        m = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, random_seed=seed, verbose=0, thread_count=-1)
        return m.fit(Xo[tr], y[tr], eval_set=(Xo[va], y[va]), early_stopping_rounds=100)
    if name == "TabPFN-v2":
        from tabpfn import TabPFNClassifier
        sub = np.random.RandomState(seed).choice(tr, min(cfg["TABPFN_N"], len(tr)), replace=False)
        return TabPFNClassifier(device=DEVICE.type).fit(Xo[sub], y[sub])
    if name == "Deep Ensemble (5 MLPs)":
        return [train_torch(MLPNet(meta["d_oh"]), D, sp, cfg, seed * 100 + j) for j in range(cfg["DE_M"])]
    model = build_torch(name, meta["n_num"], meta["cards"], meta["d_oh"], cfg, **meta.get("kw", {}))
    return train_torch(model, D, sp, cfg, seed)

def predict_any(name, model, B, Xo_np, cfg):
    """B: dict of torch tensors (xn, xc, xo); Xo_np: same rows as numpy. Returns member logits [n, k]."""
    if name in TORCH_MODELS:
        if name == "Deep Ensemble (5 MLPs)":
            return np.concatenate([predict_torch(m, B) for m in model], 1)
        return predict_torch(model, B, mc_T=mc_passes(name, cfg))
    if name == "TabPFN-v2":
        p = np.concatenate([model.predict_proba(Xo_np[i:i + 5000])[:, 1] for i in range(0, len(Xo_np), 5000)])
    else:
        p = model.predict_proba(Xo_np)[:, 1]
    return logit(p)[:, None]

def prepare_seed(seed):
    sp = make_split(y_all, seed)
    prep = Prep(num_cols, cat_cols).fit(X_all.iloc[sp["train"]])
    xn, xc, xo = prep.transform(X_all)
    D = to_device(xn, xc, xo, y_all)
    meta = dict(n_num=xn.shape[1], cards=prep.cards, d_oh=xo.shape[1])
    return sp, prep, D, xn, xc, xo, meta

def split_logits(name, model, D, xo, sp, cfg):
    return {s: predict_any(name, model, take(D, torch.tensor(sp[s], device=DEVICE)), xo[sp[s]], cfg)
            for s in ("val", "cal", "test")}

PRED_STORE = {}

def run_seed(seed, keep=False):
    set_seed(seed)
    sp, prep, D, xn, xc, xo, meta = prepare_seed(seed)
    res, art, logits_store = {}, {}, {}
    for name in MODELS:
        t0 = time.time()
        try:
            model = fit_model(name, seed, D, xo, y_all, sp, meta, CFG)
            fit_t = time.time() - t0
            t1 = time.time()
            L = split_logits(name, model, D, xo, sp, CFG)
            inf_t = (time.time() - t1) / sum(len(sp[s]) for s in ("val", "cal", "test")) * 1e6
        except Exception as e:
            print(f"  !! {name} failed: {e}")
            continue
        m, a = evaluate_from_logits(L, y_all, sp, CFG)
        m["FitTime_s"], m["Infer_us_per_sample"] = fit_t, inf_t
        if name in TORCH_MODELS:
            mods = model if isinstance(model, list) else [model]
            m["Params"] = float(sum(p.numel() for mm in mods for p in mm.parameters()))
        res[name] = m
        if name == PROPOSED:
            logits_store = L
        if keep:
            a["model"] = model
            art[name] = a
        # compact per-sample predictions (lets the decision layer be re-analysed without re-training)
        PRED_STORE.setdefault(seed, {})[name] = {
            **{f"p_{s}": a["P"][s].mean(1).astype(np.float16) for s in ("val", "cal", "test")},
            **({f"e_{s}": (L[s] / a["T"]).std(1).astype(np.float16) for s in ("cal", "test")} if L["test"].shape[1] > 1 else {})}
        print(f"  {name:28s} AUC={m['AUC']:.4f} AUPRC={m['AUPRC']:.4f} F1*={m['F1_tuned']:.4f} ECE={m['ECE']:.4f} "
              f"Miss={m['PMiss_LeLow']:.3f} FA={m['PFA_GeHigh']:.3f} HighCapt={m['Capture_H'] + m['Capture_C']:.3f} "
              f"Esc={m['Override_Rate']:.3f}  ({fit_t:.0f}s)")
    ctx = dict(sp=sp, prep=prep, D=D, xn=xn, xc=xc, xo=xo, meta=meta) if keep else None
    if not keep:
        del D
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return res, art, ctx, logits_store


# %% ======================== CELL: c08_main.py ========================
# ============================================================
# 8. MAIN EXPERIMENT: all models x N_SEEDS runs
# ============================================================
ALL = {}            # model -> list of metric dicts (one per seed)
PROP_LOGITS = {}    # seed -> proposed model logits (reused by the ablation)
T_START = time.time()
for seed in range(CFG["N_SEEDS"]):
    print(f"\n===== SEED {seed} =====")
    res, art, ctx, Lp = run_seed(seed, keep=(seed == 0))
    if seed == 0:
        ART, CTX = art, ctx
    if seed < CFG["N_SEEDS_ABL"]:
        PROP_LOGITS[seed] = Lp
    for k, v in res.items():
        ALL.setdefault(k, []).append(v)
    with open(os.path.join(OUT, "results_per_seed.json"), "w") as f:
        json.dump(dict(cfg=CFG, data=DATA_INFO, results=ALL), f, indent=1)
    save_to_gdrive(os.path.join(OUT, "results_per_seed.json"))      # progress is kept even if the session stops
    print(f"elapsed: {(time.time() - T_START) / 60:.1f} min")

# save per-sample predictions (float16) + split indices + labels/patient ids (for offline re-analysis)
_npz = {"y_all": y_all.astype(np.int8), "groups": np.unique(GROUPS, return_inverse=True)[1].astype(np.int32)
        if GROUPS is not None else np.arange(len(y_all), dtype=np.int32)}
for sd in range(CFG["N_SEEDS"]):
    _sp = make_split(y_all, sd)
    for s_ in ("val", "cal", "test"):
        _npz[f"s{sd}_idx_{s_}"] = _sp[s_].astype(np.int32)
    for mname, dct in PRED_STORE.get(sd, {}).items():
        for k, v in dct.items():
            _npz[f"s{sd}|{mname}|{k}"] = v
np.savez_compressed(os.path.join(OUT, "predictions.npz"), **_npz)
print("saved predictions.npz", round(os.path.getsize(os.path.join(OUT, "predictions.npz")) / 1e6, 1), "MB")

summary = pd.DataFrame({k: pd.DataFrame(v).mean() for k, v in ALL.items()}).T
summary.to_csv(os.path.join(OUT, "summary_mean.csv"))
pd.DataFrame({k: pd.DataFrame(v).std() for k, v in ALL.items()}).T.to_csv(os.path.join(OUT, "summary_std.csv"))
summary[["AUC", "AUPRC", "F1_tuned", "Sens_at_Spec90", "ECE", "Brier", "PMiss_LeLow", "PFA_GeHigh",
         "Capture_H", "Capture_C", "PatDetect_High", "PatFalseAlarm_High", "Override_Rate"]].round(4)


# %% ======================== CELL: c09_ablation.py ========================
# ============================================================
# 9. ABLATION STUDY (each part of PHRNet removed one at a time)
# ============================================================
ABL_VARIANTS = {
    "w/o feature-token attention": dict(use_attn=False),
    "w/o periodic embeddings": dict(tokenizer="linear"),
    "w/o BatchEnsemble head (MC dropout)": dict(k=1),
}
ABL = {"PHRNet (full)": [], "w/o temperature scaling": [], "w/o CRC (fixed cut-points)": [],
       "w/o disagreement escalation": []}
for v in ABL_VARIANTS:
    ABL[v] = []

for seed in range(CFG["N_SEEDS_ABL"]):
    print(f"\n===== ABLATION SEED {seed} =====")
    sp, prep, D, xn, xc, xo, meta = prepare_seed(seed)
    L = PROP_LOGITS[seed]
    m_full, _ = evaluate_from_logits(L, y_all, sp, CFG)
    ABL["PHRNet (full)"].append(m_full)
    m_noT, _ = evaluate_from_logits(L, y_all, sp, CFG, T=1.0)
    ABL["w/o temperature scaling"].append(m_noT)
    m_fix = dict(m_full)
    for k in list(m_full):
        if k.startswith("Fixed_"):
            m_fix[k[len("Fixed_"):]] = m_full[k]
    m_fix["Override_Rate"] = 0.0
    ABL["w/o CRC (fixed cut-points)"].append(m_fix)
    m_noov, _ = evaluate_from_logits(L, y_all, sp, CFG, override=False)
    ABL["w/o disagreement escalation"].append(m_noov)
    for vname, kw in ABL_VARIANTS.items():
        model = train_torch(PHRNet2(meta["n_num"], meta["cards"], CFG, **kw), D, sp, CFG, seed)
        mc = CFG["MC_T"] if kw.get("k") == 1 else 0
        Lv = {s: predict_torch(model, take(D, torch.tensor(sp[s], device=DEVICE)), mc_T=mc) for s in ("val", "cal", "test")}
        mv, _ = evaluate_from_logits(Lv, y_all, sp, CFG)
        ABL[vname].append(mv)
        print(f"  {vname:40s} AUC={mv['AUC']:.4f} AUPRC={mv['AUPRC']:.4f} ECE={mv['ECE']:.4f} Miss={mv['PMiss_LeLow']:.3f}")
    del D
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

pd.DataFrame({k: pd.DataFrame(v).mean() for k, v in ABL.items()}).T[
    ["AUC", "AUPRC", "ECE", "PMiss_LeLow", "PFA_GeHigh", "Capture_H", "Override_Rate"]].round(4)


# %% ======================== CELL: c10_robust.py ========================
# ============================================================
# 10. ROBUSTNESS TO SENSOR NOISE AND MISSING SENSOR VALUES (seed 0 models)
# ============================================================
ROB_MODELS = [m for m in [PROPOSED, "Deep Ensemble (5 MLPs)", "Attention-MLP + MC dropout", "FT-Transformer", "XGBoost"] if m in ART]
sp, D = CTX["sp"], CTX["D"]
te = sp["test"]
n_num = CTX["meta"]["n_num"]
rng = np.random.RandomState(123)
ROB = []

def corrupted_test(kind, level):
    xn = CTX["xn"][te].copy()
    if kind == "noise" and level > 0:
        xn = xn + rng.normal(0, level, xn.shape).astype(np.float32)
    if kind == "missing" and level > 0:
        xn[rng.rand(*xn.shape) < level] = 0.0          # missing -> mean imputation (0 after scaling)
    xo = CTX["xo"][te].copy()
    xo[:, :n_num] = xn
    B = dict(xn=torch.tensor(xn, device=DEVICE), xc=take(D, torch.tensor(te, device=DEVICE))["xc"],
             xo=torch.tensor(xo, device=DEVICE))
    return B, xo

for kind, levels in [("noise", [0.0, 0.1, 0.2, 0.3, 0.5]), ("missing", [0.1, 0.2, 0.3])]:
    for lev in levels:
        B, xo_np = corrupted_test(kind, lev)
        for name in ROB_MODELS:
            a = ART[name]
            Lr = predict_any(name, a["model"], B, xo_np, CFG) / a["T"]
            pb, tot, ep = uncertainty(sigmoid(Lr), Lr)
            yt = y_all[te]
            tiers, flag = assign_tiers(pb, a["th"], epi=ep)
            g = tier_metrics(yt, tiers, a["g"])
            ROB.append(dict(kind=kind, level=lev, model=name, AUC=roc_auc_score(yt, pb),
                            AUPRC=average_precision_score(yt, pb), ECE=ece_score(yt, pb),
                            MeanEpi=ep.mean(), PMiss_LeLow=g["PMiss_LeLow"], PFA_GeHigh=g["PFA_GeHigh"],
                            Capture_HC=g["Capture_H"] + g["Capture_C"], Override_Rate=flag.mean()))
ROB = pd.DataFrame(ROB)
ROB.to_csv(os.path.join(OUT, "robustness.csv"), index=False)
ROB.pivot_table(index=["kind", "level"], columns="model", values="AUC").round(4)


# %% ======================== CELL: c11_explain.py ========================
# ============================================================
# 11. EXPLAINABILITY: SHAP + attention + permutation importance (proposed model, seed 0)
# ============================================================
A = ART[PROPOSED]
model, T = A["model"], A["T"]
sp = CTX["sp"]
te = sp["test"]
Z_test = np.concatenate([CTX["xn"][te], CTX["xc"][te].astype(np.float32)], 1)
n_num = CTX["meta"]["n_num"]

def f_prob(Z):
    Z = np.asarray(Z, np.float32)
    B = dict(xn=torch.tensor(Z[:, :n_num], device=DEVICE),
             xc=torch.tensor(np.rint(Z[:, n_num:]).astype(np.int64), device=DEVICE),
             xo=torch.zeros((len(Z), 1), device=DEVICE))
    return sigmoid(predict_torch(model, B) / T).mean(1)

rs = np.random.RandomState(0)
SHAP_OK = False
try:
    import shap
    bg = Z_test[rs.choice(len(Z_test), 100, replace=False)]
    ex_idx = rs.choice(len(Z_test), 400, replace=False)
    explainer = shap.PermutationExplainer(f_prob, shap.maskers.Independent(bg, max_samples=100))
    sv = explainer(Z_test[ex_idx], max_evals=2 * Z_test.shape[1] + 1 + 200)
    sv.feature_names = FEATURES
    plt.figure()
    shap.plots.beeswarm(sv, max_display=len(FEATURES), show=False)
    plt.title("Global SHAP attributions (PHRNet)")
    savefig("fig_shap_global.png")
    SHAP_MEAN = pd.Series(np.abs(sv.values).mean(0), index=FEATURES).sort_values(ascending=False)
    SHAP_OK = True
except Exception as e:
    print("SHAP skipped:", e)
    SHAP_MEAN = None

# permutation importance (AUC drop) - always computed
yt = y_all[te]
base_auc = roc_auc_score(yt, f_prob(Z_test))
PERM = {}
for j, f in enumerate(FEATURES):
    Zp = Z_test.copy()
    Zp[:, j] = rs.permutation(Zp[:, j])
    PERM[f] = base_auc - roc_auc_score(yt, f_prob(Zp))
PERM = pd.Series(PERM).sort_values(ascending=False)
print("Permutation importance (AUC drop):\n", PERM.round(5))
if not SHAP_OK:
    plt.figure(figsize=(5, 4))
    PERM.iloc[::-1].plot.barh()
    plt.xlabel("AUC drop when feature is shuffled")
    savefig("fig_shap_global.png")

# attention from [CLS] token to each feature token, averaged per risk tier
with torch.no_grad():
    Bte = take(CTX["D"], torch.tensor(te, device=DEVICE))
    att = torch.cat([model.cls_attention(take(Bte, torch.arange(i, min(i + 4096, len(te)), device=DEVICE))).float().cpu()
                     for i in range(0, len(te), 4096)]).numpy()
ATT_TIER = pd.DataFrame([att[A["tiers"] == t].mean(0) if (A["tiers"] == t).any() else np.full(att.shape[1], np.nan)
                         for t in range(5)], index=TIER_NAMES, columns=FEATURES)
ATT_TIER.round(3)


# %% ======================== CELL: c12_figures.py ========================
# ============================================================
# 12. FIGURES FOR THE PAPER
# ============================================================
A = ART[PROPOSED]
te = CTX["sp"]["test"]
yt, pb, tiers = A["y"], A["pbar"], A["tiers"]
th = A["th"]

# (a) confusion matrix
thr_cm = best_f1_threshold(y_all[CTX["sp"]["val"]], A["P"]["val"].mean(1))
cm = confusion_matrix(yt, (pb >= thr_cm).astype(int))
plt.figure(figsize=(3.6, 3.2))
plt.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.xticks([0, 1], ["Low risk", "High risk"]); plt.yticks([0, 1], ["Low risk", "High risk"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.grid(False)
savefig("fig_confusion_matrix.png")

# (b) reliability diagram (equal-count bins; axis zoomed to the range of predicted risk)
def rel_curve(y, p, bins=12):
    q = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.digitize(p, q[1:-1]), 0, len(q) - 2)
    xs, ys = [], []
    for b in range(len(q) - 1):
        m = idx == b
        if m.sum() >= 20:
            xs.append(p[m].mean()); ys.append(y[m].mean())
    return xs, ys
plt.figure(figsize=(4, 3.8))
for name, style in [(PROPOSED, "o-"), ("Attention-MLP + MC dropout", "s-"), ("Deep Ensemble (5 MLPs)", "^-"),
                    ("CatBoost", "d-"), ("XGBoost", "v-")]:
    if name in ART:
        xs, ys = rel_curve(yt, ART[name]["pbar"]); plt.plot(xs, ys, style, ms=4, label=name)
p_raw = sigmoid(logit(ART[PROPOSED]["P"]["test"]) * ART[PROPOSED]["T"]).mean(1)
xs, ys = rel_curve(yt, p_raw)
plt.plot(xs, ys, "x:", ms=4, label="PHRNet before temperature scaling")
lim = float(np.quantile(pb, 0.999)) * 1.1
plt.plot([0, lim], [0, lim], "k--", lw=1, label="Perfect calibration")
plt.xlim(0, lim); plt.ylim(0, lim)
plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency"); plt.legend(fontsize=6)
savefig("fig_reliability.png")

# (c) probability vs member disagreement, coloured by tier
rs = np.random.RandomState(0)
idx = rs.choice(len(pb), min(8000, len(pb)), replace=False)
cols_t = ["#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#d73027"]
plt.figure(figsize=(4.4, 3.6))
for t in range(5):
    m = tiers[idx] == t
    if m.any():
        plt.scatter(pb[idx][m], A["epi"][idx][m], s=5, alpha=0.5, c=cols_t[t], label=TIER_NAMES[t])
if np.isfinite(th["epi_q"]):
    plt.axhline(th["epi_q"], color="gray", ls=":", lw=1, label="Escalation level")
plt.xscale("log"); plt.xlabel("Predicted risk probability (log scale)")
plt.ylabel("Member disagreement (logit std)"); plt.legend(fontsize=6, markerscale=3)
savefig("fig_uncertainty_vs_risk.png")

# (d) uncertainty per tier
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3))
for k, (key, lab) in enumerate([("total", "Total uncertainty (entropy)"), ("epi", "Member disagreement (logit std)")]):
    data = [A[key][tiers == t] for t in range(5)]
    ax[k].boxplot([d if len(d) else [0.0] for d in data], showfliers=False)
    ax[k].set_xticks(range(1, 6)); ax[k].set_xticklabels(TIER_NAMES, rotation=30)
    ax[k].set_ylabel(lab)
savefig("fig_uncertainty_by_tier.png")

# (e) tiers (figure 4 of the paper): panel labels (a)(b)(c) BELOW each panel
def label_below(ax, txt):
    ax.text(0.5, -0.30, txt, transform=ax.transAxes, ha="center", va="top", fontsize=12, fontweight="bold")
fig, ax = plt.subplots(1, 3, figsize=(12, 3.9))
cnt = np.bincount(tiers, minlength=5); esc = np.bincount(tiers[A["flag"]], minlength=5)
ax[0].bar(range(5), cnt - esc, color="tab:blue", label="Confident")
ax[0].bar(range(5), esc, bottom=cnt - esc, color="orange", hatch="//", label="Escalated (disagreement)")
for i_, c in enumerate(cnt):
    ax[0].text(i_, c, f"{100 * c / len(tiers):.1f}%", ha="center", va="bottom", fontsize=8)
ax[0].set_xticks(range(5)); ax[0].set_xticklabels(TIER_NAMES, rotation=30)
ax[0].set_ylabel("Number of patient-hours"); ax[0].legend(fontsize=7)
rate = [yt[tiers == t].mean() if (tiers == t).any() else 0 for t in range(5)]
capt = [np.mean(tiers[yt == 1] == t) for t in range(5)]
ax[1].bar(np.arange(5) - 0.2, rate, 0.4, color="tab:blue", label="High-risk rate in tier")
ax[1].bar(np.arange(5) + 0.2, capt, 0.4, color="tab:orange", hatch="..", label="Share of all high-risk hours")
ax[1].axhline(yt.mean(), color="k", ls=":", lw=1, label=f"Prevalence ({100 * yt.mean():.1f}%)")
ax[1].set_xticks(range(5)); ax[1].set_xticklabels(TIER_NAMES, rotation=30); ax[1].set_ylabel("Fraction"); ax[1].legend(fontsize=7)
bins = np.logspace(np.log10(max(pb.min(), 1e-5)), 0, 60)
ax[2].hist(pb[yt == 0], bins=bins, alpha=0.6, label="Low-risk hours", density=True)
ax[2].hist(pb[yt == 1], bins=bins, alpha=0.6, label="High-risk hours", density=True, histtype="stepfilled", hatch="//")
ymax = ax[2].get_ylim()[1]
for key, ls, lab, f_ in [("eL1", ":", r"$\tau'_{L1}$", 0.92), ("eL2", "--", r"$\tau'_{L2}$", 0.72),
                         ("eH2", "--", r"$\tau_{H2}$", 0.92), ("eH1", ":", r"$\tau_{H1}$", 0.72)]:
    if 0 < th[key] <= 1:
        ax[2].axvline(th[key], color="k", ls=ls, lw=1); ax[2].text(th[key] * 1.06, ymax * f_, lab, fontsize=9)
ax[2].set_xscale("log"); ax[2].set_xlabel("Predicted risk probability (log scale)"); ax[2].set_ylabel("Density"); ax[2].legend(fontsize=7)
for a_, t_ in zip(ax, ["(a)", "(b)", "(c)"]):
    label_below(a_, t_)
plt.subplots_adjust(wspace=0.3, bottom=0.28)
savefig("fig_tiers_alerts.png")

# (f) precision-recall curves (the informative curve for rare events)
from sklearn.metrics import precision_recall_curve
plt.figure(figsize=(4.2, 3.4))
for name in [PROPOSED, "CatBoost", "XGBoost", "FT-Transformer", "Deep Ensemble (5 MLPs)", "Attention-MLP + MC dropout"]:
    if name in ART:
        pr, rc, _ = precision_recall_curve(yt, ART[name]["pbar"])
        plt.plot(rc, pr, lw=1.1, label=f"{name} (AP={average_precision_score(yt, ART[name]['pbar']):.3f})")
plt.axhline(yt.mean(), color="k", ls=":", lw=1, label="Prevalence")
plt.xlabel("Recall (sensitivity)"); plt.ylabel("Precision (PPV)"); plt.legend(fontsize=5.5); plt.ylim(0, 0.6)
savefig("fig_pr_curves.png")

# (g) robustness (figure 6 of the paper): panel labels (a)(b)(c) BELOW each panel
rn = ROB[ROB.kind == "noise"]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.7))
for name, st in zip(ROB_MODELS, ["o-", "s--", "^-.", "d:", "v-"]):
    r = rn[rn.model == name]
    ax[0].plot(r.level, r.AUC, st, ms=4, label=name); ax[1].plot(r.level, r.AUPRC, st, ms=4, label=name)
r = rn[rn.model == PROPOSED]
ax[2].plot(r.level, r.PMiss_LeLow, "o-", label="Miss risk (Very Low/Low)")
ax[2].plot(r.level, r.PFA_GeHigh, "s--", label="False-alarm risk (High/Critical)")
ax[2].plot(r.level, r.Override_Rate, "^-.", label="Escalation rate")
ax[2].axhline(CFG["ALPHA_MAIN"], color="gray", ls=":", label="Target (5%)")
for a_, yl in zip(ax, ["AUC", "AUPRC", "Rate"]):
    a_.set_xlabel("Gaussian sensor noise (std, z-units)"); a_.set_ylabel(yl)
ax[0].legend(fontsize=7); ax[2].legend(fontsize=7)
for a_, t_ in zip(ax, ["(a)", "(b)", "(c)"]):
    label_below(a_, t_)
plt.subplots_adjust(wspace=0.3, bottom=0.28)
savefig("fig_noise_robustness.png")

# (h) attention by tier
plt.figure(figsize=(7, 2.8))
plt.imshow(ATT_TIER.values, aspect="auto", cmap="viridis")
plt.colorbar(label="[CLS] attention")
plt.yticks(range(5), TIER_NAMES); plt.xticks(range(len(FEATURES)), FEATURES, rotation=60, ha="right", fontsize=7); plt.grid(False)
savefig("fig_attention_tiers.png")

# (i) feature profile per tier (z-scores, raw means written in cells)
raw = X_all.iloc[te][num_cols].reset_index(drop=True)
zt = (raw - raw.mean()) / raw.std()
Zm = pd.DataFrame([zt[tiers == t].mean() for t in range(5)], index=TIER_NAMES)
Rm = pd.DataFrame([raw[tiers == t].mean() for t in range(5)], index=TIER_NAMES)
plt.figure(figsize=(8, 3))
plt.imshow(Zm.values, aspect="auto", cmap="coolwarm", vmin=-1.5, vmax=1.5)
plt.colorbar(label="Mean z-score")
for i in range(Zm.shape[0]):
    for j in range(Zm.shape[1]):
        if not np.isnan(Rm.values[i, j]):
            plt.text(j, i, f"{Rm.values[i, j]:.1f}", ha="center", va="center", fontsize=6)
plt.yticks(range(5), TIER_NAMES); plt.xticks(range(len(num_cols)), num_cols, rotation=60, ha="right", fontsize=7); plt.grid(False)
savefig("fig_feature_by_tier.png")
Rm.round(2).to_csv(os.path.join(OUT, "feature_means_by_tier.csv"))


# %% ======================== CELL: c13_tables.py ========================
# ============================================================
# 13. LATEX TABLES + NUMBERS (MACROS) FOR THE PAPER + STATISTICAL TESTS
# ============================================================
def col(d_list, key):
    return [d.get(key, np.nan) for d in d_list]

order = [m for m in MODELS if m in ALL]
rows = [(m, {k: col(ALL[m], k) for k in ALL[m][0]}) for m in order]
for _, d in rows:
    d["Capture_HC"] = list(np.add(d["Capture_H"], d["Capture_C"]))


write("tab_performance.tex", build_table(
    rows, ["AUC", "AUPRC", "F1_tuned", "Sens_tuned", "Spec_tuned", "Sens_at_Spec90"], {},
    f"Discrimination on the held-out test split (mean$\\pm$std over {CFG['N_SEEDS']} independent runs). F1, sensitivity and "
    "specificity use the threshold that maximizes F1 on the validation split; Sens@Spec90 uses the validation threshold giving 90\\% specificity. Best mean in bold.",
    "tab:performance", colnames=["AUC", "AUPRC", "F1", "Sensitivity", "Specificity", "Sens@Spec90"]))

write("tab_reliability.tex", build_table(
    rows, ["Brier", "NLL", "ECE_raw", "ECE"], {"Brier": False, "NLL": False, "ECE_raw": False, "ECE": False},
    "Probability quality on the test split (lower is better). ECE is computed with 15 equal-width bins before (raw) "
    "and after temperature scaling fitted on the validation split.",
    "tab:reliability", star=False, nd=4, colnames=["Brier", "NLL", "ECE (raw)", "ECE"]))

# ---- decision layer of PHRNet: guarantees + what each tier contains ----
P = {k: col(ALL[PROPOSED], k) for k in ALL[PROPOSED][0]}
a_s, a_m = CFG["ALPHA_STRICT"], CFG["ALPHA_MAIN"]
cuts = "/".join(f"{c:g}" for c in CFG["FIXED_CUTS"])
def pct(v): return fmt_ms(100 * np.asarray(v, float), 2)
s = ["\\begin{table}[t]", "\\centering",
     f"\\caption{{Decision layer of PHRNet on the test patients (mean$\\pm$std over {CFG['N_SEEDS']} runs, \\%). "
     "Patient-level risks are the CRC losses: the average, over patients, of the fraction of their high-risk (or low-risk) "
     "hours that land in the stated tiers. Targets hold in expectation over calibration draws.}",
     "\\label{tab:tiers_guarantee}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}", "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{lccc}", "\\toprule",
     "\\textbf{Quantity} & \\textbf{Target} & \\textbf{CRC tiers} & \\textbf{Fixed " + cuts + "} \\\\", "\\midrule"]
for lab, tgt, key in [("High-risk hours in Very Low", f"$\\leq {100*a_s:g}$", "PMiss_VeryLow"),
                      ("High-risk hours in Very Low/Low", f"$\\leq {100*a_m:g}$", "PMiss_LeLow"),
                      ("Low-risk hours in Critical", f"$\\leq {100*a_s:g}$", "PFA_Critical"),
                      ("Low-risk hours in High/Critical", f"$\\leq {100*a_m:g}$", "PFA_GeHigh")]:
    s.append(f"{lab} & {tgt} & {pct(P[key])} & {pct(P['Fixed_' + key])} \\\\")
s += ["\\midrule",
      f"High-risk hours in High/Critical & -- & {pct(np.add(P['Capture_H'], P['Capture_C']))} & {pct(np.add(P['Fixed_Capture_H'], P['Fixed_Capture_C']))} \\\\",
      f"Septic patients with a High/Critical alert & -- & {pct(P['PatDetect_High'])} & {pct(P['Fixed_PatDetect_High'])} \\\\",
      f"Non-septic patients with a High/Critical alert & -- & {pct(P['PatFalseAlarm_High'])} & {pct(P['Fixed_PatFalseAlarm_High'])} \\\\",
      f"Hours escalated by member disagreement & -- & {pct(P['Override_Rate'])} & -- \\\\",
      "\\bottomrule", "\\end{tabular}}", "\\end{table}"]
write("tab_tiers_guarantee.tex", "\n".join(s))

s = ["\\begin{table}[t]", "\\centering",
     f"\\caption{{Content of each PHRNet risk tier on the test split (mean over {CFG['N_SEEDS']} runs, \\%). "
     "Rate: observed share of high-risk hours in the tier (positive predictive value); Capture: share of all high-risk hours "
     "that fall in the tier; Lift: rate divided by prevalence.}", "\\label{tab:tiers_content}", "\\footnotesize",
     "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{llcccc}", "\\toprule",
     "\\textbf{Tier} & \\textbf{Alert} & \\textbf{Hours} & \\textbf{Rate} & \\textbf{Capture} & \\textbf{Lift} \\\\", "\\midrule"]
prev = np.nanmean(P["Prevalence"])
for nm, tname, aname in zip(["VL", "L", "M", "H", "C"], TIER_NAMES, ALERT_NAMES):
    sh, rt, cp = np.nanmean(P[f"Share_{nm}"]), np.nanmean(P[f"Rate_{nm}"]), np.nanmean(P[f"Capture_{nm}"])
    rt_s = "--" if np.isnan(rt) else f"{100*rt:.2f}"
    lift = "--" if np.isnan(rt) else f"{rt/prev:.2f}"
    s.append(f"{tname} & {aname} & {100*sh:.2f} & {rt_s} & {100*cp:.2f} & {lift} \\\\")
s += ["\\midrule", f"All & -- & 100 & {100*prev:.2f} & 100 & 1.00 \\\\", "\\bottomrule", "\\end{tabular}", "\\end{table}"]
write("tab_tiers_content.tex", "\n".join(s))

# decision layer applied to every model (the layer is model-agnostic)
write("tab_decision_all.tex", build_table(
    rows, ["PMiss_LeLow", "PFA_GeHigh", "Capture_HC", "PatDetect_High", "PatFalseAlarm_High"],
    {"PMiss_LeLow": False, "PFA_GeHigh": False, "Capture_HC": True, "PatDetect_High": True, "PatFalseAlarm_High": False},
    "The same patient-level CRC decision layer applied to every model (fractions, mean$\\pm$std). Miss and FA must stay below 0.05; "
    "with these satisfied, a better model places more high-risk hours in High/Critical (Capture) and alerts on more septic patients.",
    "tab:decision_all", nd=3, colnames=["Miss", "FA", "Capture H/C", "Septic pts alerted", "Non-septic pts alerted"]))

# statistical tests: proposed vs every baseline (paired over seeds, Holm-corrected Wilcoxon)
def holm(pvals):
    p = np.asarray(pvals, float); n = len(p); order_ = np.argsort(p); adj = np.empty(n); run = 0
    for r, i in enumerate(order_):
        run = max(run, min(1.0, (n - r) * p[i])); adj[i] = run
    return adj
STAT = []
STAT_METRICS = ["AUC", "AUPRC", "Sens_at_Spec90", "ECE"]
for mt in STAT_METRICS:
    ps, rows_mt = [], []
    for b in order:
        if b == PROPOSED:
            continue
        x, z = np.array(col(ALL[PROPOSED], mt)), np.array(col(ALL[b], mt))
        n = min(len(x), len(z)); x, z = x[:n], z[:n]
        try:
            pw = stats.wilcoxon(x, z).pvalue
        except Exception:
            pw = 1.0
        try:
            pt = stats.ttest_rel(x, z).pvalue
        except Exception:
            pt = 1.0
        ps.append(pw); rows_mt.append(dict(metric=mt, baseline=b, delta=float(np.mean(x - z)), p_wilcoxon=pw, p_ttest=pt))
    for r, pa in zip(rows_mt, holm(ps)):
        r["p_holm"] = pa
    STAT += rows_mt
STAT = pd.DataFrame(STAT)
STAT.to_csv(os.path.join(OUT, "stat_tests.csv"), index=False)

def pfmt(p):
    return "$<$0.001" if p < 1e-3 else f"{p:.3f}"
s = ["\\begin{table*}[t]", "\\centering",
     f"\\caption{{Paired comparison of PHRNet against each baseline over {CFG['N_SEEDS']} runs. "
     "$\\Delta$ = PHRNet minus baseline (positive is better for AUC, AUPRC and Sens@Spec90; negative is better for ECE); "
     "$p$ = two-sided Wilcoxon signed-rank test, Holm-corrected.}", "\\label{tab:stats}", "\\footnotesize",
     "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{lcccccccc}", "\\toprule",
     "\\textbf{Baseline} & $\\Delta$AUC & $p$ & $\\Delta$AUPRC & $p$ & $\\Delta$Sens@Sp90 & $p$ & $\\Delta$ECE & $p$ \\\\", "\\midrule"]
for b in [o for o in order if o != PROPOSED]:
    cells = []
    for mt in STAT_METRICS:
        r = STAT[(STAT.metric == mt) & (STAT.baseline == b)].iloc[0]
        cells += [f"{r.delta:+.4f}", pfmt(r.p_holm)]
    s.append(latex_escape(b) + " & " + " & ".join(cells) + " \\\\")
s += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
write("tab_stats.tex", "\n".join(s))

# ablation
arows = [(k, {kk: col(v, kk) for kk in v[0]}) for k, v in ABL.items() if len(v)]
for _, d in arows:
    d["Capture_HC"] = list(np.add(d["Capture_H"], d["Capture_C"]))
write("tab_ablation.tex", build_table(
    arows, ["AUC", "AUPRC", "ECE", "PMiss_LeLow", "PFA_GeHigh", "Capture_HC"],
    {"ECE": False, "PMiss_LeLow": False, "PFA_GeHigh": False},
    f"Ablation of PHRNet (mean$\\pm$std over {CFG['N_SEEDS_ABL']} runs). Miss/FA: patient-level CRC risks (targets 0.05); "
    "Capture: share of high-risk hours placed in High/Critical. The first four rows share one trained network.",
    "tab:ablation", star=True, colnames=["AUC", "AUPRC", "ECE", "Miss", "FA", "Capture H/C"]))

# robustness
short = {"PHRNet (Proposed)": "PHRNet", "Attention-MLP + MC dropout": "MC-Drop", "Deep Ensemble (5 MLPs)": "DeepEns",
         "FT-Transformer": "FT-T"}
rp = ROB[ROB.kind == "noise"].pivot_table(index="level", columns="model", values="AUC")
rq = ROB[(ROB.kind == "noise") & (ROB.model == PROPOSED)].set_index("level")
s = ["\\begin{table}[t]", "\\centering", "\\caption{Test AUC under additive Gaussian sensor noise (standardized units), and the "
     "patient-level miss risk and escalation rate of PHRNet's decision layer (first run).}", "\\label{tab:noise}", "\\footnotesize",
     "\\setlength{\\tabcolsep}{2.5pt}", "\\begin{tabular}{c" + "c" * len(ROB_MODELS) + "cc}", "\\toprule",
     "$\\sigma$ & " + " & ".join(latex_escape(short.get(m, m)) for m in ROB_MODELS) + " & Miss & Esc. \\\\", "\\midrule"]
for lev in rp.index:
    s.append(f"{lev:.1f} & " + " & ".join(f"{rp.loc[lev, m]:.4f}" for m in ROB_MODELS)
             + f" & {rq.loc[lev, 'PMiss_LeLow']:.3f} & {rq.loc[lev, 'Override_Rate']:.3f} \\\\")
s += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
write("tab_noise.tex", "\n".join(s))

# numbers used in the text (\newcommand macros)
def mean(m, k):
    return float(np.nanmean(col(ALL[m], k)))
def std(m, k):
    return float(np.nanstd(col(ALL[m], k), ddof=1))
base_names = [o for o in order if o != PROPOSED]
best = lambda k, hi=True: (max if hi else min)(base_names, key=lambda b: mean(b, k))
th0 = ART[PROPOSED]["th"]
MC = "Attention-MLP + MC dropout"
def pv(base, mt):
    r = STAT[(STAT.metric == mt) & (STAT.baseline == base)]
    return "--" if r.empty else ("$<$0.001" if r.p_holm.values[0] < 1e-3 else f"{r.p_holm.values[0]:.3f}")
M = {
    "DataName": DATA_INFO["name"], "NRows": f"{DATA_INFO['n_rows']:,}", "NPatients": f"{DATA_INFO['n_patients']:,}",
    "NFeat": str(len(FEATURES)), "Prev": f"{100 * DATA_INFO['prevalence']:.2f}",
    "NSeeds": str(CFG["N_SEEDS"]), "NSeedsAbl": str(CFG["N_SEEDS_ABL"]), "NTest": f"{len(CTX['sp']['test']):,}",
    "TreeThreeAUC": f"{DATA_INFO['tree_depth3_auc']:.3f}",
    "PropAUC": f"{mean(PROPOSED, 'AUC'):.3f}", "PropAUCsd": f"{std(PROPOSED, 'AUC'):.3f}",
    "PropAUPRC": f"{mean(PROPOSED, 'AUPRC'):.3f}", "PropFone": f"{mean(PROPOSED, 'F1_tuned'):.3f}",
    "PropSens": f"{100 * mean(PROPOSED, 'Sens_tuned'):.1f}", "PropSpec": f"{100 * mean(PROPOSED, 'Spec_tuned'):.1f}",
    "PropSensNinety": f"{100 * mean(PROPOSED, 'Sens_at_Spec90'):.1f}",
    "PropECE": f"{mean(PROPOSED, 'ECE'):.4f}", "PropECEraw": f"{mean(PROPOSED, 'ECE_raw'):.4f}",
    "PropBrier": f"{mean(PROPOSED, 'Brier'):.4f}", "PropT": f"{mean(PROPOSED, 'Temperature'):.2f}",
    "PropParams": f"{mean(PROPOSED, 'Params') / 1e3:.0f}k", "PropInfer": f"{mean(PROPOSED, 'Infer_us_per_sample'):.1f}",
    "BestAUCBase": latex_escape(best("AUC")), "BestAUCBaseVal": f"{mean(best('AUC'), 'AUC'):.3f}",
    "BestAUPRCBase": latex_escape(best("AUPRC")), "BestAUPRCBaseVal": f"{mean(best('AUPRC'), 'AUPRC'):.3f}",
    "PvalAUCBest": pv(best("AUC"), "AUC"), "PvalAUPRCBest": pv(best("AUPRC"), "AUPRC"),
    "VoneAUC": f"{mean(MC, 'AUC'):.3f}", "VoneAUPRC": f"{mean(MC, 'AUPRC'):.3f}", "PvalAUCVone": pv(MC, "AUC"),
    "FTAUC": f"{mean('FT-Transformer', 'AUC'):.3f}", "DEAUC": f"{mean('Deep Ensemble (5 MLPs)', 'AUC'):.3f}",
    "PropMissVL": f"{100 * mean(PROPOSED, 'PMiss_VeryLow'):.2f}", "PropMiss": f"{100 * mean(PROPOSED, 'PMiss_LeLow'):.2f}",
    "PropFAC": f"{100 * mean(PROPOSED, 'PFA_Critical'):.2f}", "PropFA": f"{100 * mean(PROPOSED, 'PFA_GeHigh'):.2f}",
    "PropMissRow": f"{100 * mean(PROPOSED, 'Miss_LeLow'):.2f}",
    "PropCaptHC": f"{100 * (mean(PROPOSED, 'Capture_H') + mean(PROPOSED, 'Capture_C')):.1f}",
    "PropShareHC": f"{100 * (mean(PROPOSED, 'Share_H') + mean(PROPOSED, 'Share_C')):.2f}",
    "PropShareLow": f"{100 * (mean(PROPOSED, 'Share_VL') + mean(PROPOSED, 'Share_L')):.1f}",
    "PropShareMod": f"{100 * mean(PROPOSED, 'Share_M'):.1f}",
    "PropRateC": f"{100 * mean(PROPOSED, 'Rate_C'):.1f}", "PropRateH": f"{100 * mean(PROPOSED, 'Rate_H'):.1f}",
    "PropRateLow": f"{100 * np.nanmean(np.r_[col(ALL[PROPOSED], 'Rate_VL'), col(ALL[PROPOSED], 'Rate_L')]):.2f}",
    "PropPatDetect": f"{100 * mean(PROPOSED, 'PatDetect_High'):.1f}", "PropPatFA": f"{100 * mean(PROPOSED, 'PatFalseAlarm_High'):.1f}",
    "PropOverride": f"{100 * mean(PROPOSED, 'Override_Rate'):.1f}",
    "PropEscRate": f"{100 * mean(PROPOSED, 'Esc_Rate_Pos'):.2f}", "PropNoEscRate": f"{100 * mean(PROPOSED, 'NoEsc_LowTier_Rate_Pos'):.2f}",
    "FixedMiss": f"{100 * mean(PROPOSED, 'Fixed_PMiss_LeLow'):.1f}", "FixedCaptHC": f"{100 * (mean(PROPOSED, 'Fixed_Capture_H') + mean(PROPOSED, 'Fixed_Capture_C')):.1f}",
    "FixedCuts": cuts, "AlphaS": f"{100 * a_s:g}", "AlphaM": f"{100 * a_m:g}",
    "TauLone": f"{th0['eL1']:.4f}", "TauLtwo": f"{th0['eL2']:.4f}", "TauHtwo": f"{th0['eH2']:.4f}", "TauHone": f"{th0['eH1']:.4f}",
}
r_n = ROB[(ROB.kind == "noise") & (ROB.model == PROPOSED)].set_index("level")
M["NoiseAUCHigh"] = f"{r_n.loc[0.5, 'AUC']:.3f}"; M["NoiseMissHigh"] = f"{100 * r_n.loc[0.5, 'PMiss_LeLow']:.1f}"
M["NoiseFAHigh"] = f"{100 * r_n.loc[0.5, 'PFA_GeHigh']:.1f}"; M["NoiseOverrideHigh"] = f"{100 * r_n.loc[0.5, 'Override_Rate']:.1f}"
M["TopFeature"] = latex_escape(SHAP_MEAN.index[0] if SHAP_MEAN is not None else PERM.index[0])
M["SecondFeature"] = latex_escape(SHAP_MEAN.index[1] if SHAP_MEAN is not None else PERM.index[1])
write("results_macros.tex", "\n".join(f"\\newcommand{{\\Res{k}}}{{{v}}}" for k, v in M.items()))

# plain-language summary to send back to Claude
lines = ["# Results summary", f"Data: {DATA_INFO}", "", "## Mean over seeds", summary.round(5).to_string(), "",
         "## Ablation (mean)", pd.DataFrame({k: pd.DataFrame(v).mean() for k, v in ABL.items()}).T.round(5).to_string(), "",
         "## Stat tests", STAT.round(5).to_string(), "", "## Robustness", ROB.round(4).to_string(), "",
         "## Permutation importance", PERM.round(5).to_string(), "", "## Macros", json.dumps(M, indent=1)]
write("results_summary.md", "\n".join(lines))


# %% ======================== CELL: c14_zip.py ========================
# ============================================================
# 14. ZIP EVERYTHING -> download paper_outputs.zip from the Output tab
# ============================================================
zpath = os.path.join(os.path.dirname(OUT), "paper_outputs.zip")
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir(OUT)):
        z.write(os.path.join(OUT, f), f)
print("Created", zpath)
save_to_gdrive(zpath)                                               # final results to Google Drive
print(sorted(os.listdir(OUT)))
print(f"Total time: {(time.time() - T_START) / 60:.1f} min")


