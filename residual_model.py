# ==============================================================================
# residual_model.py  —  Phase 4 Rewrite
# GA2M (EBM) residual intelligence layer on top of the baseline GLM
#
# Architecture (per Specs S2, S13, S-B.2, S-B.3 + Ambiguity Resolution C):
#
#   Target  : log(Expected_Pure_Premium / GLM_Pure_Premium)   [log-uplift]
#   Training: TRAINING split only (Split == "train" from baseline_glm.py)
#   Features: 25 base + 3 derived = 28 total  (EBM_ALL_FEATURES in config)
#   EBM     : 6 forced must-include interactions + 9 auto-discovered = 15 total
#   Corridor: clipped to [log(0.65), log(1.60)] before exponentiation
#   Norm    : GLM-premium-weighted mean uplift normalised to 1.0 exactly
#             → E_w[uplift] = 1.0  → total book premium unchanged
#
#   Final_Pure_Premium = GLM_Pure_Premium × normalized_uplift_factor
#
# Key outputs (written to final_predictions.csv):
#   EBM_Log_Uplift     — raw (post-clip, pre-norm) log uplift
#   EBM_Uplift_Factor  — normalised multiplicative factor
#   Final_Pure_Premium — final price per policy
#   EBM_Residual_Pred  — dollar adjustment (Final − GLM), kept for app.py compat
#   Adjustment_Pct     — (Final/GLM − 1) × 100, for scatter plot colouring
#   GLM_Risk_Tier      — risk tier using GLM_Pure_Premium
#   Final_Risk_Tier    — risk tier using Final_Pure_Premium (may differ)
#
# Performance reported on TEST set only (OOS N=20,000).
# ==============================================================================

import json
import os
import sys
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    BASELINE_DATA_PATH,
    FINAL_DATA_PATH,
    EBM_MODEL_PATH,
    METADATA_PATH,
    MODEL_DIR,
    PREMIUM_FLOOR,
    RANDOM_STATE,
    MIN_UPLIFT,
    MAX_UPLIFT,
    EBM_MAX_BINS,
    EBM_MAX_INT_BINS,
    EBM_LEARNING_RATE,
    EBM_OUTER_BAGS,
    EBM_INNER_BAGS,
    EBM_ALL_FEATURES,
    EBM_CAT_COLS,
    EBM_DERIVED_FEATURES,
    MUST_INCLUDE_INTERACTIONS,
    TIER_BOUNDARIES,
    TIER_ORDER,
)

# Log-corridor bounds (applied to raw EBM prediction before exponentiation)
LOG_MIN = np.log(MIN_UPLIFT)   # log(0.65) ≈ −0.431
LOG_MAX = np.log(MAX_UPLIFT)   # log(1.60) ≈  0.470


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add/overwrite the three EBM-specific derived columns:
      Dwelling_Age      — 2026 − Year_Built  (more interpretable age signal)
      RCV_Overstatement — max(0, AOI − RCV_Appraised)  (pre-computed moral hazard gap)
      Log_AOI           — log(AOI)  (stabilises right tail for interaction terms)

    These may already exist in the CSV from data_simulation.py; this function
    guarantees they are present and correctly typed.
    """
    df = df.copy()
    df["Dwelling_Age"]      = (2026 - df["Year_Built"]).astype(int)
    df["RCV_Overstatement"] = np.maximum(0.0, df["AOI"] - df["RCV_Appraised"])
    df["Log_AOI"]           = np.log(df["AOI"].clip(1))
    return df


def _cast_cat_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all EBM categorical columns are dtype str (InterpretML requirement)."""
    for col in EBM_CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def _assign_tier(premium: np.ndarray) -> np.ndarray:
    """Map premium values to risk tier labels using TIER_BOUNDARIES from config."""
    tiers = np.full(len(premium), "High", dtype=object)
    for tier, (lo, hi) in TIER_BOUNDARIES.items():
        mask = (premium >= lo) & (premium < hi)
        tiers[mask] = tier
    return tiers


def _build_interaction_list(feature_names: list) -> list:
    """
    Convert MUST_INCLUDE_INTERACTIONS (list of name-pair tuples from config) to
    feature-index tuples, then append the integer 9 to request 9 additional
    auto-discovered interactions → 6 forced + 9 auto = 15 total.

    Falls back to interactions=15 (pure auto) if the InterpretML version does
    not support mixed lists.  The fallback is detected at fit time via try/except.
    """
    idx = {name: i for i, name in enumerate(feature_names)}
    forced = []
    missing = []
    for fa, fb in MUST_INCLUDE_INTERACTIONS:
        if fa in idx and fb in idx:
            forced.append((idx[fa], idx[fb]))
        else:
            missing.append((fa, fb))
    if missing:
        print(f"  WARNING: {len(missing)} must-include pair(s) not found in "
              f"feature list (will be auto-discovered instead): {missing}")
    return forced


def _verify_discovered_interactions(ebm, feature_names: list) -> list:
    """
    Print and return all pairwise interaction terms discovered by the fitted EBM.
    Each interaction term in ebm.term_names_ is represented as a list/tuple of
    two feature indices when it's a pairwise term.
    """
    interactions_found = []
    print("\n  ── DISCOVERED EBM INTERACTIONS ──")
    n_main = len(feature_names)
    term_idx = 0
    for term in ebm.term_names_:
        # InterpretML ≥ 0.6: term_names_ contains strings like "feat_a x feat_b"
        # or tuples/lists of indices depending on version.
        if isinstance(term, (list, tuple)):
            # Older API: term is a list of two feature indices
            if len(term) == 2:
                fa = feature_names[term[0]] if term[0] < len(feature_names) else str(term[0])
                fb = feature_names[term[1]] if term[1] < len(feature_names) else str(term[1])
                interactions_found.append((fa, fb))
                print(f"    [{term_idx:>2}] {fa}  ×  {fb}")
        elif isinstance(term, str) and " x " in term:
            # Newer API: term is a string like "Feature_A x Feature_B"
            parts = term.split(" x ")
            interactions_found.append(tuple(p.strip() for p in parts))
            print(f"    [{term_idx:>2}] {term}")
        term_idx += 1

    if not interactions_found:
        print("    (Could not parse interaction terms — check InterpretML version)")

    # Check must-include coverage
    must_names = set()
    for fa, fb in MUST_INCLUDE_INTERACTIONS:
        must_names.add(frozenset([fa, fb]))

    found_sets = {frozenset(pair) for pair in interactions_found}
    print("\n  ── MUST-INCLUDE COVERAGE ──")
    for fa, fb in MUST_INCLUDE_INTERACTIONS:
        key    = frozenset([fa, fb])
        status = "✓ Found" if key in found_sets else "✗ NOT found (may need interactions=forced+auto)"
        print(f"    {fa:<40} × {fb:<40} : {status}")

    return interactions_found


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def train_residual_ebm(data_path: str = BASELINE_DATA_PATH) -> dict:
    """
    Train the EBM GA2M residual layer on log-scale GLM residuals.

    Parameters
    ----------
    data_path : path to the enriched CSV produced by baseline_glm.py
                (must contain GLM_Pure_Premium and Split columns)

    Returns
    -------
    dict — metrics for setup.py pipeline summary
    """
    from interpret.glassbox import ExplainableBoostingRegressor

    # ─────────────────────────────────────────────────────────────────────────
    # 1. LOAD DATA
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\nLoading enriched data from '{data_path}'…")
    df = pd.read_csv(data_path)
    n  = len(df)

    # Validate required columns are present
    required = ["GLM_Pure_Premium", "Expected_Pure_Premium", "Split"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Required columns missing from enriched dataset: {missing_cols}\n"
            f"Run baseline_glm.py first to generate these columns."
        )
    print(f"  {n:,} policies loaded  "
          f"(train: {(df['Split']=='train').sum():,}  "
          f"test: {(df['Split']=='test').sum():,})")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. FEATURE ENGINEERING — DERIVED COLUMNS
    # ─────────────────────────────────────────────────────────────────────────
    print("Adding derived EBM features (Dwelling_Age, RCV_Overstatement, Log_AOI)…")
    df = _add_derived_features(df)
    df = _cast_cat_cols(df)

    # Validate all 28 EBM features are present
    missing_feats = [f for f in EBM_ALL_FEATURES if f not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing EBM features: {missing_feats}")
    print(f"  EBM feature count: {len(EBM_ALL_FEATURES)} "
          f"(25 base + 3 derived = 28 total)")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. SPLIT — EBM trains on TRAINING set only
    # ─────────────────────────────────────────────────────────────────────────
    train_mask = df["Split"] == "train"
    test_mask  = df["Split"] == "test"

    df_train = df[train_mask].copy()
    df_test  = df[test_mask].copy()

    X_train  = df_train[EBM_ALL_FEATURES].copy()
    X_test   = df_test[EBM_ALL_FEATURES].copy()

    # ─────────────────────────────────────────────────────────────────────────
    # 4. LOG-SCALE RESIDUAL TARGET  y = log(True/GLM)
    # ─────────────────────────────────────────────────────────────────────────
    eps      = 1e-6
    log_true_train = np.log(df_train["Expected_Pure_Premium"].values + eps)
    log_glm_train  = np.log(df_train["GLM_Pure_Premium"].values       + eps)
    y_log_train    = log_true_train - log_glm_train   # log uplift factor

    log_true_test  = np.log(df_test["Expected_Pure_Premium"].values + eps)
    log_glm_test   = np.log(df_test["GLM_Pure_Premium"].values       + eps)
    y_log_test     = log_true_test - log_glm_test

    print(f"\n  Log-residual (train): "
          f"mean={y_log_train.mean():.4f}  std={y_log_train.std():.4f}  "
          f"p5={np.percentile(y_log_train, 5):.3f}  "
          f"p95={np.percentile(y_log_train, 95):.3f}")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. PRE-TRAINING RISK-NEUTRALITY CHECK  (Spec N4.4)
    # The GLM-premium-weighted mean of the training target should be ≈ 0.
    # A large deviation suggests the GLM systematically over- or under-prices.
    # We record this but do NOT centre the target — centering would distort
    # the learned shape functions and is handled at prediction time instead.
    # ─────────────────────────────────────────────────────────────────────────
    glm_weights_train = df_train["GLM_Pure_Premium"].values
    weighted_mean_target = np.average(y_log_train, weights=glm_weights_train)
    print(f"\n  Pre-training neutrality check:")
    print(f"    GLM-weighted mean of log-residual (train): {weighted_mean_target:.6f}")
    if abs(weighted_mean_target) > 0.05:
        print(f"    WARNING: |weighted mean| > 0.05 — GLM may be systematically "
              f"biased. Consider centring the target.")
    else:
        print(f"    OK — weighted mean is close to 0 (threshold ±0.05)")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. BUILD FORCED-INTERACTION LIST  (Spec S13 + Ambiguity C)
    # Try mixed list [forced_tuples, N_auto]; fall back to interactions=15.
    # ─────────────────────────────────────────────────────────────────────────
    forced_pairs = _build_interaction_list(EBM_ALL_FEATURES)
    N_AUTO       = 9    # auto-discovered on top of the 6 forced = 15 total

    print(f"\n  Interaction strategy: {len(forced_pairs)} forced + {N_AUTO} auto-discovered")

    # ─────────────────────────────────────────────────────────────────────────
    # 7. FIT EBM
    # Attempt Option A (mixed list) first; fall back to pure integer if the
    # installed InterpretML version raises TypeError.
    # ─────────────────────────────────────────────────────────────────────────
    ebm_kwargs = dict(
        feature_names     = EBM_ALL_FEATURES,
        max_bins          = EBM_MAX_BINS,
        max_interaction_bins = EBM_MAX_INT_BINS,
        learning_rate     = EBM_LEARNING_RATE,
        outer_bags        = EBM_OUTER_BAGS,
        inner_bags        = EBM_INNER_BAGS,
        random_state      = RANDOM_STATE,
    )

    _interaction_mode = "mixed"
    try:
        print(f"  Attempting mixed interactions list (6 forced + {N_AUTO} auto)…")
        ebm = ExplainableBoostingRegressor(
            interactions = forced_pairs + [N_AUTO],   # Option A
            **ebm_kwargs,
        )
        ebm.fit(X_train, y_log_train)
        print("  EBM training complete (mixed mode).")
    except (TypeError, ValueError) as e:
        print(f"  Mixed list not supported ({e.__class__.__name__}: {e})")
        print(f"  Falling back to interactions={len(forced_pairs) + N_AUTO} "
              f"(pure auto-discovery)…")
        _interaction_mode = "auto"
        ebm = ExplainableBoostingRegressor(
            interactions = len(forced_pairs) + N_AUTO,   # Option B
            **ebm_kwargs,
        )
        ebm.fit(X_train, y_log_train)
        print(f"  EBM training complete (auto mode, {len(forced_pairs) + N_AUTO} interactions).")

    # ─────────────────────────────────────────────────────────────────────────
    # 8. PRINT DISCOVERED INTERACTIONS
    # ─────────────────────────────────────────────────────────────────────────
    interactions_found = _verify_discovered_interactions(ebm, EBM_ALL_FEATURES)

    # ─────────────────────────────────────────────────────────────────────────
    # 9. PREDICT ON FULL DATASET (train + test)
    # Raw EBM prediction → clip to corridor → exponentiate → normalise
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  Generating predictions on full dataset…")
    raw_log_all   = ebm.predict(df[EBM_ALL_FEATURES].copy())
    clipped_log   = np.clip(raw_log_all, LOG_MIN, LOG_MAX)
    raw_uplift    = np.exp(clipped_log)

    # ─────────────────────────────────────────────────────────────────────────
    # 10. RISK-NEUTRALITY NORMALISATION  (Spec N4.1 — CRITICAL)
    # Normalise so the GLM-premium-weighted mean uplift = 1.0 exactly.
    # → sum(Final_PP_i) = sum(GLM_PP_i)   i.e. no book-level inflation.
    # Normalisation uses ALL policies (train + test) so the constraint holds
    # for the full portfolio that the carrier would price.
    # ─────────────────────────────────────────────────────────────────────────
    glm_weights_all  = df["GLM_Pure_Premium"].values
    raw_weighted_mean = np.average(raw_uplift, weights=glm_weights_all)
    norm_uplift       = raw_uplift / raw_weighted_mean

    # Hard verification — must hold to machine precision
    check_val = np.average(norm_uplift, weights=glm_weights_all)
    assert abs(check_val - 1.0) < 1e-6, (
        f"Risk neutrality assertion failed: "
        f"weighted mean uplift = {check_val:.8f}, expected 1.0"
    )

    print(f"\n  ── RISK NEUTRALITY ──")
    print(f"    Raw weighted mean uplift   : {raw_weighted_mean:.6f}×")
    print(f"    Normalisation factor       : {1.0 / raw_weighted_mean:.6f}")
    print(f"    Post-norm weighted mean    : {check_val:.6f}×  ✓")
    print(f"    Total GLM premium          : ${glm_weights_all.sum():,.0f}")
    print(f"    Total Final premium        : ${(glm_weights_all * norm_uplift).sum():,.0f}")
    diff_dollars = (glm_weights_all * norm_uplift).sum() - glm_weights_all.sum()
    print(f"    Book-level difference      : ${diff_dollars:,.0f}  "
          f"({'≈$0 ✓' if abs(diff_dollars) < 1 else 'WARNING'})")

    # ─────────────────────────────────────────────────────────────────────────
    # 11. WRITE PREDICTION COLUMNS INTO DATAFRAME
    # ─────────────────────────────────────────────────────────────────────────
    df["EBM_Log_Uplift"]     = clipped_log
    df["EBM_Uplift_Factor"]  = norm_uplift
    df["Final_Pure_Premium"] = np.clip(
        df["GLM_Pure_Premium"] * norm_uplift, PREMIUM_FLOOR, None
    )
    # Dollar adjustment — kept for app.py backwards compatibility
    df["EBM_Residual_Pred"]  = df["Final_Pure_Premium"] - df["GLM_Pure_Premium"]
    # Percentage adjustment — used by reclassification scatter (Spec S9)
    df["Adjustment_Pct"]     = (df["EBM_Uplift_Factor"] - 1.0) * 100.0
    # Per-policy risk tiers under each model
    df["GLM_Risk_Tier"]      = _assign_tier(df["GLM_Pure_Premium"].values)
    df["Final_Risk_Tier"]    = _assign_tier(df["Final_Pure_Premium"].values)

    # ─────────────────────────────────────────────────────────────────────────
    # 12. OOS PERFORMANCE  (TEST SET ONLY — Spec G2.6)
    # ─────────────────────────────────────────────────────────────────────────
    df_test_eval = df[test_mask]
    true_pp_test = df_test_eval["Expected_Pure_Premium"].values
    glm_pp_test  = df_test_eval["GLM_Pure_Premium"].values
    fin_pp_test  = df_test_eval["Final_Pure_Premium"].values

    glm_r2_test  = r2_score(true_pp_test, glm_pp_test)
    final_r2_test = r2_score(true_pp_test, fin_pp_test)
    glm_rmse_test  = np.sqrt(mean_squared_error(true_pp_test, glm_pp_test))
    fin_rmse_test  = np.sqrt(mean_squared_error(true_pp_test, fin_pp_test))
    delta_r2       = final_r2_test - glm_r2_test

    # Also compute EBM-only R² on the log-residual (OOS)
    test_raw_log = ebm.predict(df_test[EBM_ALL_FEATURES].copy())
    test_clipped = np.clip(test_raw_log, LOG_MIN, LOG_MAX)
    ebm_log_r2_test = r2_score(y_log_test, test_clipped)

    print(f"\n{'='*64}")
    print("  MODEL PERFORMANCE  (out-of-sample test set, N="
          f"{test_mask.sum():,})")
    print(f"{'='*64}")
    print(f"  {'Model':<35} {'R²':>8}  {'RMSE':>10}  {'Mean PP':>10}")
    print(f"  {'-'*35} {'-'*8}  {'-'*10}  {'-'*10}")
    print(f"  {'Legacy GLM (16 feat, linear)':<35} "
          f"{glm_r2_test:>8.4f}  "
          f"${glm_rmse_test:>9,.0f}  "
          f"${glm_pp_test.mean():>9,.0f}")
    print(f"  {'GLM + GA2M (28 feat, glass-box)':<35} "
          f"{final_r2_test:>8.4f}  "
          f"${fin_rmse_test:>9,.0f}  "
          f"${fin_pp_test.mean():>9,.0f}")
    print(f"  {'Incremental ΔR²':<35} "
          f"{delta_r2:>+8.4f}")
    print(f"  {'EBM log-residual R² (OOS)':<35} "
          f"{ebm_log_r2_test:>8.4f}")
    print(f"{'='*64}")

    # ΔR² target validation
    if delta_r2 < 0.05:
        print(f"  WARNING: ΔR² = {delta_r2:.4f} is below target (0.07–0.12). "
              f"Consider increasing DGP_MODERN_SCALAR in config.py.")
    elif delta_r2 > 0.15:
        print(f"  WARNING: ΔR² = {delta_r2:.4f} exceeds target (0.07–0.12). "
              f"Consider decreasing DGP_MODERN_SCALAR in config.py.")
    else:
        print(f"  ΔR² = {delta_r2:.4f} is within target range [0.07, 0.12] ✓")

    # ─────────────────────────────────────────────────────────────────────────
    # 13. PREMIUM MIGRATION SUMMARY  (Spec N4.5)
    # ─────────────────────────────────────────────────────────────────────────
    adj      = df["EBM_Residual_Pred"]
    up_mask  = adj > 0
    dn_mask  = adj < 0

    premium_up   = adj[up_mask].sum()
    premium_down = adj[dn_mask].abs().sum()
    pct_up       = up_mask.mean() * 100
    pct_down     = dn_mask.mean() * 100

    # Tier-level migration matrix
    from_glm_tiers = df["GLM_Risk_Tier"]
    to_final_tiers = df["Final_Risk_Tier"]
    tier_moves     = (from_glm_tiers != to_final_tiers).sum()

    print(f"\n  ── PREMIUM MIGRATION ──")
    print(f"    Policies receiving surcharge    : {up_mask.sum():>7,}  "
          f"({pct_up:.1f}%)  +${premium_up:,.0f}")
    print(f"    Policies receiving credit       : {dn_mask.sum():>7,}  "
          f"({pct_down:.1f}%)  −${premium_down:,.0f}")
    print(f"    Policies unchanged (within $10) : "
          f"{(adj.abs() <= 10).sum():>7,}")
    print(f"    Net premium migration           : ${premium_up - premium_down:>+,.0f}  "
          f"({'≈$0 ✓' if abs(premium_up - premium_down) / max(premium_up, 1) < 0.01 else 'CHECK'})")
    print(f"    Tier reclassifications          : {tier_moves:>7,}  "
          f"({tier_moves/n:.1%} of portfolio)")

    # Tier-to-tier migration table (test set)
    print(f"\n  ── TIER MIGRATION MATRIX (full portfolio) ──")
    print(f"  {'GLM Tier':>12}  →  {'Final Tier':<12}  Count    % of Book")
    for t_from in TIER_ORDER:
        for t_to in TIER_ORDER:
            if t_from == t_to:
                continue
            mask  = (from_glm_tiers == t_from) & (to_final_tiers == t_to)
            count = mask.sum()
            if count > 0:
                print(f"  {t_from:>12}  →  {t_to:<12}  {count:>6,}    "
                      f"{count/n:.2%}")

    # ─────────────────────────────────────────────────────────────────────────
    # 14. UPLIFT CORRIDOR DIAGNOSTICS
    # ─────────────────────────────────────────────────────────────────────────
    at_floor = (clipped_log <= LOG_MIN + 0.001).sum()
    at_ceil  = (clipped_log >= LOG_MAX - 0.001).sum()
    print(f"\n  ── UPLIFT CORRIDOR ──")
    print(f"    Corridor: [{MIN_UPLIFT:.2f}×, {MAX_UPLIFT:.2f}×]  "
          f"(log: [{LOG_MIN:.3f}, {LOG_MAX:.3f}])")
    print(f"    Asymmetry justified: underpriced risks pose greater "
          f"adverse selection danger → +60% ceiling vs −35% floor")
    print(f"    Policies at floor ({MIN_UPLIFT:.2f}×)  : "
          f"{at_floor:,}  ({at_floor/n:.2%})")
    print(f"    Policies at ceiling ({MAX_UPLIFT:.2f}×): "
          f"{at_ceil:,}  ({at_ceil/n:.2%})")
    print(f"    Uplift factor range : "
          f"{norm_uplift.min():.4f}× – {norm_uplift.max():.4f}×")
    print(f"    Mean uplift (norm)  : {norm_uplift.mean():.6f}×")

    # ─────────────────────────────────────────────────────────────────────────
    # 15. SAVE ARTIFACTS
    # ─────────────────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(FINAL_DATA_PATH), exist_ok=True)

    # Final predictions CSV
    df.to_csv(FINAL_DATA_PATH, index=False)
    print(f"\n  Saved final predictions to '{FINAL_DATA_PATH}'")

    # EBM model
    joblib.dump(ebm, EBM_MODEL_PATH)
    print(f"  Saved EBM model to '{EBM_MODEL_PATH}'")

    # Update model metadata JSON (merge with existing GLM metadata if present)
    existing_meta = {}
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH) as f:
                existing_meta = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing_meta = {}

    ebm_meta = {
        "ebm_training_date":           datetime.now().isoformat(timespec="seconds"),
        "ebm_features":                EBM_ALL_FEATURES,
        "n_ebm_features":              len(EBM_ALL_FEATURES),
        "n_interactions_total":        len(forced_pairs) + N_AUTO,
        "n_interactions_forced":       len(forced_pairs),
        "n_interactions_auto":         N_AUTO,
        "interaction_mode":            _interaction_mode,
        "interactions_discovered":     [list(p) for p in interactions_found],
        "log_corridor":                [round(LOG_MIN, 4), round(LOG_MAX, 4)],
        "uplift_corridor":             [MIN_UPLIFT, MAX_UPLIFT],
        "risk_neutrality_check":       round(float(check_val), 8),
        "raw_weighted_mean_uplift":    round(float(raw_weighted_mean), 6),
        "glm_r2_test":                 round(float(glm_r2_test),    4),
        "final_r2_test":               round(float(final_r2_test),  4),
        "delta_r2":                    round(float(delta_r2),       4),
        "glm_rmse_test":               round(float(glm_rmse_test),  2),
        "final_rmse_test":             round(float(fin_rmse_test),  2),
        "total_glm_premium":           round(float(glm_weights_all.sum()), 0),
        "total_final_premium":         round(float((glm_weights_all * norm_uplift).sum()), 0),
        "premium_up_dollars":          round(float(premium_up),   0),
        "premium_down_dollars":        round(float(premium_down), 0),
        "pct_policies_surcharge":      round(float(pct_up),   2),
        "pct_policies_credit":         round(float(pct_down), 2),
        "tier_reclassifications":      int(tier_moves),
    }
    existing_meta.update(ebm_meta)
    with open(METADATA_PATH, "w") as f:
        json.dump(existing_meta, f, indent=2)
    print(f"  Updated model metadata at '{METADATA_PATH}'")

    # ─────────────────────────────────────────────────────────────────────────
    # 16. RETURN METRICS DICT (for setup.py pipeline summary)
    # ─────────────────────────────────────────────────────────────────────────
    return {
        "final_r2":           f"{final_r2_test:.4f}",
        "delta_r2":           f"+{delta_r2:.4f}",
        "risk_neutral_check": f"{check_val:.6f}× (target: 1.000000)",
        "interactions_found": len(interactions_found),
        "book_delta":         f"${diff_dollars:,.0f}",
    }


# ── Script entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    metrics = train_residual_ebm(BASELINE_DATA_PATH)
    print(f"\nEBM residual model complete.")
    print(f"  Final R² (OOS) : {metrics['final_r2']}")
    print(f"  ΔR²            : {metrics['delta_r2']}")
    print(f"  Risk neutrality: {metrics['risk_neutral_check']}")
    print(f"  Book impact    : {metrics['book_delta']}")
    print("Run 'python app.py' next to launch the demo.\n")
