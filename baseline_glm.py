# ==============================================================================
# baseline_glm.py  —  Phase 3 Rewrite
# Legacy Poisson × Gamma GLM baseline for ResiScore™ demo
#
# Architecture (per Specs S3, S5, S12 + Ambiguity Resolution D):
#   • StatsmodelsGLMWrapper — sklearn-compatible, exposes actuarial diagnostics
#     (deviance, AIC, BIC, p-values, coefficient CIs)
#   • Separate sklearn ColumnTransformer preprocessor (saved independently)
#   • 16-feature input: 12 main effects + 4 engineered actuary interactions
#   • Exposure offset: AOI / 100_000  (1 unit = $100K coverage-year)
#   • Frequency model: Poisson(log link), fitted on full dataset
#   • Severity model: Gamma(log link), fitted only on claim-bearing policies,
#     weighted by Claim_Count
#   • Strict 80/20 train/test split; OOS metrics reported
#   • Returns metrics dict for setup.py pipeline summary
#
# Output files:
#   models/glm_preprocessor.pkl       — sklearn ColumnTransformer
#   models/freq_glm.pkl               — StatsmodelsGLMWrapper (Poisson)
#   models/sev_glm.pkl                — StatsmodelsGLMWrapper (Gamma)
#   data/synthetic_homeowners_data_with_baseline.csv  — enriched with GLM cols
# ==============================================================================

import os
import sys
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    RAW_DATA_PATH,
    BASELINE_DATA_PATH,
    FREQ_MODEL_PATH,
    SEV_MODEL_PATH,
    PREPROCESSOR_PATH,
    METADATA_PATH,
    MODEL_DIR,
    PREMIUM_FLOOR,
    TEST_SIZE,
    RANDOM_STATE,
    GLM_ALL_FEATURES,
    GLM_CAT_COLS,
    GLM_NUM_COLS,
    GLM_INTERACTIONS,
)


# ══════════════════════════════════════════════════════════════════════════════
# STATSMODELS GLM WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class StatsmodelsGLMWrapper(BaseEstimator, RegressorMixin):
    """
    Wraps a statsmodels GLM to provide:
      1. sklearn-compatible .fit(X, y) / .predict(X) interface
         (X is a pre-processed dense numeric array from the ColumnTransformer)
      2. Actuarial diagnostics: deviance, AIC, BIC, dispersion, p-values,
         coefficient confidence intervals — unavailable in sklearn GLMs
      3. Named coefficient access via .coefficients (pd.Series) for the
         GLM Breakdown waterfall in app.py

    The wrapper prepends a constant column via sm.add_constant().
    Feature names must be provided at fit time to label .coefficients.
    """

    def __init__(self, family, feature_names=None):
        self.family       = family
        self.feature_names = feature_names   # list of names for preprocessed columns
        self.model_        = None
        self.results_      = None

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X, y, exposure=None, freq_weights=None):
        """
        Parameters
        ----------
        X            : ndarray (n, p) — preprocessed features (ColumnTransformer output)
        y            : array-like     — target (claim count or severity)
        exposure     : array-like     — exposure vector for Poisson offset; passed to
                       sm.GLM as `exposure` (log-link: adds log(exposure) to linear predictor)
        freq_weights : array-like     — observation weights (claim count for Gamma severity)
        """
        X_const = sm.add_constant(X, has_constant="add")
        self.model_ = sm.GLM(
            endog          = y,
            exog           = X_const,
            family         = self.family,
            exposure       = exposure,
            freq_weights   = freq_weights,
        )
        self.results_ = self.model_.fit(
            method   = "irls",
            maxiter  = 200,
            disp     = False,
        )
        return self

    # ── predict ──────────────────────────────────────────────────────────────

    def predict(self, X, exposure=None):
        """
        Parameters
        ----------
        X        : ndarray (n, p) — preprocessed features (same scale as fit)
        exposure : array-like or None — exposure for Poisson prediction.
                   If None, uses exposure = 1.0 (rate per unit exposure).
        """
        X_const = sm.add_constant(X, has_constant="add")
        return self.results_.predict(X_const, exposure=exposure)

    # ── named coefficients ────────────────────────────────────────────────────

    @property
    def coefficients(self):
        """
        Returns model coefficients as a pd.Series with named index.
        Includes the intercept as 'const'.
        Used by app.py GLM Breakdown waterfall to attribute log-scale contributions.
        """
        if self.results_ is None:
            raise RuntimeError("Model not fitted yet.")
        params = self.results_.params
        if self.feature_names is not None:
            names = ["const"] + list(self.feature_names)
        else:
            names = [f"x{i}" for i in range(len(params))]
        return pd.Series(params, index=names)

    @property
    def pvalues(self):
        """p-values for all coefficients (including const)."""
        if self.results_ is None:
            raise RuntimeError("Model not fitted yet.")
        params = self.results_.pvalues
        if self.feature_names is not None:
            names = ["const"] + list(self.feature_names)
        else:
            names = [f"x{i}" for i in range(len(params))]
        return pd.Series(params, index=names)

    @property
    def conf_int(self):
        """95% confidence intervals as a DataFrame."""
        if self.results_ is None:
            raise RuntimeError("Model not fitted yet.")
        ci = self.results_.conf_int()
        if self.feature_names is not None:
            ci.index = ["const"] + list(self.feature_names)
        return ci

    @property
    def aic(self):
        return self.results_.aic if self.results_ else None

    @property
    def bic(self):
        return self.results_.bic if self.results_ else None

    @property
    def deviance(self):
        return self.results_.deviance if self.results_ else None

    @property
    def deviance_explained(self):
        """Percentage of null deviance explained (analogous to R² for GLMs)."""
        if self.results_ is None:
            return None
        return (1.0 - self.results_.deviance / self.results_.null_deviance) * 100

    def summary(self):
        return self.results_.summary() if self.results_ else "Not fitted."


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _engineer_glm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Dwelling_Age and the 4 engineered interaction columns.
    Called on the full dataset before splitting.
    """
    df = df.copy()

    # Dwelling_Age — more interpretable than raw Year_Built (GLM_MAIN_EFFECTS uses it)
    if "Dwelling_Age" not in df.columns:
        df["Dwelling_Age"] = (2026 - df["Year_Built"]).astype(int)

    # ── 4 actuary-grade GLM interactions (ISO-grounded) ──────────────────────
    # I1: Frame construction × Poor fire response (classic ISO named interaction)
    df["Frame_HighPC"] = (
        (df["Construction_Type"] == "Frame") &
        (df["Protection_Class"] > 6)
    ).astype(int)

    # I2: Frequent claimants + low deductible = adverse selection signal
    df["FreqClaims_LowDed"] = (
        (df["CLUE_Loss_Count"] >= 2) &
        (df["Deductible"].astype(int) <= 500)
    ).astype(int)

    # I3: Urban territory + poor fire protection class
    df["Urban_HighPC"] = (
        (df["Territory"] == "Urban") &
        (df["Protection_Class"] > 6)
    ).astype(int)

    # I4: Old roof in high-hail zone
    df["OldRoof_HighHail"] = (
        (df["Roof_Age_Applicant"] > 20) &
        (df["Hail_Frequency"] >= 3)
    ).astype(int)

    # Cast interaction columns to str for OneHotEncoder consistency
    for col in GLM_INTERACTIONS:
        df[col] = df[col].astype(str)

    return df


def _build_preprocessor() -> ColumnTransformer:
    """
    Build the sklearn ColumnTransformer for GLM inputs.
    Uses GLM_NUM_COLS / GLM_CAT_COLS from config.
    StandardScaler on numeric; OneHotEncoder(drop='first') on categorical.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), GLM_NUM_COLS),
            ("cat", OneHotEncoder(drop="first", sparse_output=False,
                                  handle_unknown="ignore"), GLM_CAT_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def _get_preprocessor_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Return the flat list of feature names output by the fitted ColumnTransformer.
    Used to label StatsmodelsGLMWrapper coefficients.
    """
    return list(preprocessor.get_feature_names_out())


# ══════════════════════════════════════════════════════════════════════════════
# PRINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _print_coefficient_table(model: StatsmodelsGLMWrapper, title: str) -> None:
    """Print a compact coefficient table with significance stars."""
    print(f"\n  ── {title} ──")
    print(f"  {'Feature':<50} {'Coef':>10} {'p-value':>10} {'Sig'}")
    print(f"  {'-'*50} {'-'*10} {'-'*10} {'-'*4}")

    coefs   = model.coefficients
    pvals   = model.pvalues

    for name in coefs.index:
        c = coefs[name]
        p = pvals[name]
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "** "
        elif p < 0.05:
            sig = "*  "
        elif p < 0.10:
            sig = ".  "
        else:
            sig = "   "
        print(f"  {name:<50} {c:>10.4f} {p:>10.4f} {sig}")


def _print_metrics_table(split_name: str, r2: float, rmse: float,
                         mean_pred: float, mean_true: float) -> None:
    print(f"  [{split_name}]  R² = {r2:.4f}   "
          f"RMSE = ${rmse:,.0f}   "
          f"Mean predicted = ${mean_pred:,.0f}   "
          f"Mean true = ${mean_true:,.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_baseline_glm(data_path: str = RAW_DATA_PATH) -> dict:
    """
    Train the Poisson × Gamma GLM baseline on a synthetic homeowners dataset.

    Parameters
    ----------
    data_path : path to the raw CSV produced by data_simulation.py

    Returns
    -------
    dict  — metrics for the setup.py pipeline summary
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 1. LOAD DATA
    # ─────────────────────────────────────────────────────────────────────────
    print("\nLoading synthetic data…")
    df = pd.read_csv(data_path)
    n  = len(df)
    print(f"  {n:,} policies loaded from '{data_path}'")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. FEATURE ENGINEERING  (Dwelling_Age + 4 GLM interactions)
    # ─────────────────────────────────────────────────────────────────────────
    print("Engineering GLM features…")
    df = _engineer_glm_features(df)

    # Ensure categorical columns are strings for the encoder
    for col in GLM_CAT_COLS:
        df[col] = df[col].astype(str)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. TRAIN / TEST SPLIT (80 / 20, random_state=42, stratified by Risk_Tier)
    # The Split column is saved into the enriched CSV so residual_model.py
    # can align on the exact same train / test indices.
    # ─────────────────────────────────────────────────────────────────────────
    print(f"Splitting data: {int((1-TEST_SIZE)*100)}% train / "
          f"{int(TEST_SIZE*100)}% test (random_state={RANDOM_STATE})…")

    train_idx, test_idx = train_test_split(
        df.index,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        # Stratify on Risk_Tier to ensure all tiers appear in both sets
        stratify     = df["Risk_Tier"] if "Risk_Tier" in df.columns else None,
    )
    df["Split"] = "train"
    df.loc[test_idx, "Split"] = "test"
    print(f"  Train: {len(train_idx):,}   Test: {len(test_idx):,}")

    df_train = df.loc[train_idx].copy()
    df_test  = df.loc[test_idx].copy()

    # ─────────────────────────────────────────────────────────────────────────
    # 4. FEATURE MATRIX  X  and TARGETS
    # ─────────────────────────────────────────────────────────────────────────
    X_train_raw = df_train[GLM_ALL_FEATURES].copy()
    X_test_raw  = df_test[GLM_ALL_FEATURES].copy()

    y_freq_train = df_train["Claim_Count"].values.astype(float)
    y_freq_test  = df_test["Claim_Count"].values.astype(float)

    # Severity: only policies with at least one claim; weighted by claim count
    sev_mask_train = y_freq_train > 0
    sev_mask_test  = y_freq_test  > 0

    y_sev_train  = (df_train.loc[sev_mask_train, "Claim_Amount"].values /
                    df_train.loc[sev_mask_train, "Claim_Count"].values)
    w_sev_train  = df_train.loc[sev_mask_train, "Claim_Count"].values.astype(float)

    # Exposure: AOI / 100_000  (standard in HO rate filings — 1 unit = $100K insured)
    # Using a consistent exposure prevents the frequency model from attributing
    # AOI variation to the intercept rather than to AOI as a rating factor.
    exposure_train = (df_train["AOI"].values / 100_000).astype(float)
    exposure_test  = (df_test["AOI"].values  / 100_000).astype(float)

    exposure_sev_train = exposure_train[sev_mask_train]

    # ─────────────────────────────────────────────────────────────────────────
    # 5. PREPROCESSOR
    # ─────────────────────────────────────────────────────────────────────────
    print("Building and fitting ColumnTransformer preprocessor…")
    preprocessor = _build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train_raw).astype(float)
    X_test_proc  = preprocessor.transform(X_test_raw).astype(float)

    # Severity subsets (rows where Claim_Count > 0)
    X_sev_train = X_train_proc[sev_mask_train]
    X_sev_test  = X_test_proc[sev_mask_test]

    feat_names = _get_preprocessor_feature_names(preprocessor)
    print(f"  Preprocessed feature count: {len(feat_names)}")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. FREQUENCY MODEL  —  Poisson(log), exposure-offset
    # ─────────────────────────────────────────────────────────────────────────
    print("Training Frequency GLM (Poisson, log link, AOI exposure offset)…")
    freq_glm = StatsmodelsGLMWrapper(
        family         = sm.families.Poisson(link=sm.families.links.Log()),
        feature_names  = feat_names,
    )
    freq_glm.fit(X_train_proc, y_freq_train, exposure=exposure_train)

    print(f"  Frequency GLM — Deviance explained: "
          f"{freq_glm.deviance_explained:.2f}%  "
          f"AIC: {freq_glm.aic:,.1f}  "
          f"Deviance: {freq_glm.deviance:,.1f}")

    # ─────────────────────────────────────────────────────────────────────────
    # 7. SEVERITY MODEL  —  Gamma(log), claim-count weighted
    # ─────────────────────────────────────────────────────────────────────────
    print("Training Severity GLM (Gamma, log link, claim-count weights)…")
    sev_glm = StatsmodelsGLMWrapper(
        family        = sm.families.Gamma(link=sm.families.links.Log()),
        feature_names = feat_names,
    )
    # Severity model: exposure = 1 (not AOI-offset, since severity is per-claim)
    sev_glm.fit(X_sev_train, y_sev_train, freq_weights=w_sev_train)

    print(f"  Severity GLM — Deviance explained: "
          f"{sev_glm.deviance_explained:.2f}%  "
          f"AIC: {sev_glm.aic:,.1f}  "
          f"Deviance: {sev_glm.deviance:,.1f}")

    # ─────────────────────────────────────────────────────────────────────────
    # 8. GLM PREDICTIONS  (frequency × severity = pure premium)
    # ─────────────────────────────────────────────────────────────────────────
    print("Generating GLM predictions…")

    def _glm_pure_premium(X_proc, exposure):
        """
        Compute GLM pure premium:
          freq_pred:  predicted claim frequency  (Poisson rate per exposure unit)
          sev_pred:   predicted claim severity   (Gamma mean)
          pure_prem:  freq_pred × exposure × sev_pred  → expected loss per policy

        Note: exposure is included so that freq_pred × exposure = expected claim count,
        and pure_premium = expected_claim_count × severity_mean.
        """
        freq_pred = freq_glm.predict(X_proc, exposure=exposure)
        sev_pred  = sev_glm.predict(X_proc)
        freq_pred = np.clip(freq_pred, 0.001, 0.50)
        sev_pred  = np.clip(sev_pred,  500,   250_000)
        # freq_pred here is the expected RATE per exposure unit;
        # multiplying by exposure gives expected claim count;
        # multiplying by severity gives expected pure premium.
        pure_prem = freq_pred * sev_pred
        return np.clip(pure_prem, PREMIUM_FLOOR, None), freq_pred, sev_pred

    glm_pp_train, freq_pred_train, sev_pred_train = _glm_pure_premium(
        X_train_proc, exposure_train)
    glm_pp_test,  freq_pred_test,  sev_pred_test  = _glm_pure_premium(
        X_test_proc,  exposure_test)

    # ─────────────────────────────────────────────────────────────────────────
    # 9. PERFORMANCE METRICS
    # ─────────────────────────────────────────────────────────────────────────
    true_pp_train = df_train["Expected_Pure_Premium"].values
    true_pp_test  = df_test["Expected_Pure_Premium"].values

    r2_train   = r2_score(true_pp_train, glm_pp_train)
    r2_test    = r2_score(true_pp_test,  glm_pp_test)
    rmse_train = np.sqrt(mean_squared_error(true_pp_train, glm_pp_train))
    rmse_test  = np.sqrt(mean_squared_error(true_pp_test,  glm_pp_test))

    # ─────────────────────────────────────────────────────────────────────────
    # 10. PRINT SUMMARIES
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  BASELINE GLM EVALUATION")
    print("=" * 64)
    _print_metrics_table("TRAIN", r2_train, rmse_train,
                         glm_pp_train.mean(), true_pp_train.mean())
    _print_metrics_table("TEST ", r2_test,  rmse_test,
                         glm_pp_test.mean(),  true_pp_test.mean())

    print("\n  FREQUENCY MODEL DIAGNOSTICS")
    print(f"    Deviance explained : {freq_glm.deviance_explained:.2f}%")
    print(f"    AIC                : {freq_glm.aic:,.1f}")
    print(f"    BIC                : {freq_glm.bic:,.1f}")
    print(f"    Residual deviance  : {freq_glm.deviance:,.1f}")

    print("\n  SEVERITY MODEL DIAGNOSTICS")
    print(f"    Deviance explained : {sev_glm.deviance_explained:.2f}%")
    print(f"    AIC                : {sev_glm.aic:,.1f}")
    print(f"    BIC                : {sev_glm.bic:,.1f}")
    print(f"    Residual deviance  : {sev_glm.deviance:,.1f}")

    # Coefficient tables (compact)
    _print_coefficient_table(freq_glm, "FREQUENCY GLM COEFFICIENTS (log-scale)")
    _print_coefficient_table(sev_glm,  "SEVERITY GLM COEFFICIENTS (log-scale)")

    # Full statsmodels summary (long-form, for diagnostics)
    print("\n  ── FREQUENCY GLM STATSMODELS SUMMARY ──")
    print(freq_glm.summary())
    print("\n  ── SEVERITY GLM STATSMODELS SUMMARY ──")
    print(sev_glm.summary())
    print("=" * 64)

    # ─────────────────────────────────────────────────────────────────────────
    # 11. WRITE GLM PREDICTIONS INTO FULL DATAFRAME (train + test combined)
    # ─────────────────────────────────────────────────────────────────────────
    print("\nWriting GLM predictions to enriched dataset…")

    # Predict on full dataset (needed by residual_model.py for training target)
    X_all_raw  = df[GLM_ALL_FEATURES].copy()
    for col in GLM_CAT_COLS:
        X_all_raw[col] = X_all_raw[col].astype(str)
    X_all_proc = preprocessor.transform(X_all_raw).astype(float)
    exposure_all = (df["AOI"].values / 100_000).astype(float)

    glm_pp_all, freq_pred_all, sev_pred_all = _glm_pure_premium(
        X_all_proc, exposure_all)

    df["GLM_Freq_Pred"]    = freq_pred_all
    df["GLM_Sev_Pred"]     = sev_pred_all
    df["GLM_Pure_Premium"] = glm_pp_all
    df["GLM_Log_Offset"]   = np.log(glm_pp_all)

    # ─────────────────────────────────────────────────────────────────────────
    # 12. SAVE ARTIFACTS
    # ─────────────────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(BASELINE_DATA_PATH), exist_ok=True)

    # Enriched dataset (with GLM predictions + Split column)
    df.to_csv(BASELINE_DATA_PATH, index=False)
    print(f"  Saved enriched dataset to '{BASELINE_DATA_PATH}'")

    # Models
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"  Saved preprocessor   to '{PREPROCESSOR_PATH}'")

    joblib.dump(freq_glm, FREQ_MODEL_PATH)
    print(f"  Saved freq GLM       to '{FREQ_MODEL_PATH}'")

    joblib.dump(sev_glm, SEV_MODEL_PATH)
    print(f"  Saved sev GLM        to '{SEV_MODEL_PATH}'")

    # Model metadata JSON
    metadata = {
        "training_date":          datetime.now().isoformat(timespec="seconds"),
        "data_path":              data_path,
        "n_total":                n,
        "n_train":                len(train_idx),
        "n_test":                 len(test_idx),
        "glm_features":           GLM_ALL_FEATURES,
        "preprocessed_features":  feat_names,
        "glm_r2_train":           round(float(r2_train),  4),
        "glm_r2_test":            round(float(r2_test),   4),
        "glm_rmse_train":         round(float(rmse_train), 2),
        "glm_rmse_test":          round(float(rmse_test),  2),
        "glm_freq_aic":           round(float(freq_glm.aic or 0), 1),
        "glm_freq_deviance_explained": round(float(freq_glm.deviance_explained or 0), 2),
        "glm_sev_aic":            round(float(sev_glm.aic or 0), 1),
        "glm_sev_deviance_explained":  round(float(sev_glm.deviance_explained or 0), 2),
        "premium_floor":          PREMIUM_FLOOR,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved model metadata to '{METADATA_PATH}'")

    # ─────────────────────────────────────────────────────────────────────────
    # 13. RETURN METRICS DICT (for setup.py pipeline summary)
    # ─────────────────────────────────────────────────────────────────────────
    return {
        "train_r2":   f"{r2_train:.4f}",
        "test_r2":    f"{r2_test:.4f}",
        "test_rmse":  f"${rmse_test:,.0f}",
        "freq_aic":   f"{freq_glm.aic:,.1f}",
        "freq_deviance_explained": f"{freq_glm.deviance_explained:.2f}%",
    }


# ── Script entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    metrics = run_baseline_glm(RAW_DATA_PATH)
    print(f"\nGLM pipeline complete.")
    print(f"  Train R²: {metrics['train_r2']}")
    print(f"  Test  R²: {metrics['test_r2']}  (out-of-sample)")
    print(f"  Test RMSE: {metrics['test_rmse']}")
    print("Run 'python residual_model.py' next to train the EBM residual layer.\n")
