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
#             Three-strategy cascade (interaction spec fix):
#               Strategy A: mixed list [forced_tuples + integer] (InterpretML ≥ 0.6.2)
#               Strategy B: scout pass → extract top 9 pairs → explicit tuple list
#               Strategy C: pure integer fallback (interactions=15)
#   Corridor: clipped to [log(0.65), log(1.60)] before exponentiation
#   Norm    : GLM-premium-weighted mean uplift normalised to 1.0 exactly
#
#   Final_Pure_Premium = GLM_Pure_Premium × normalized_uplift_factor
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

LOG_MIN = np.log(MIN_UPLIFT)
LOG_MAX = np.log(MAX_UPLIFT)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add/overwrite the three EBM-specific derived columns."""
    df = df.copy()
    df["Dwelling_Age"]      = (2026 - df["Year_Built"]).astype(int)
    df["RCV_Overstatement"] = np.maximum(0.0, df["AOI"] - df["RCV_Appraised"])
    df["Log_AOI"]           = np.log(df["AOI"].clip(1))
    return df


def _cast_cat_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all EBM categorical columns are dtype str."""
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
    Convert MUST_INCLUDE_INTERACTIONS name-pairs to feature-index tuples.
    Returns the forced list only; the caller appends [N_AUTO] for Strategy A.
    """
    idx     = {name: i for i, name in enumerate(feature_names)}
    forced  = []
    missing = []
    for fa, fb in MUST_INCLUDE_INTERACTIONS:
        if fa in idx and fb in idx:
            forced.append((idx[fa], idx[fb]))
        else:
            missing.append((fa, fb))
    if missing:
        print(f"  WARNING: {len(missing)} must-include pair(s) not found: {missing}")
    return forced


def _verify_discovered_interactions(ebm, feature_names: list) -> list:
    """
    Detect and print all pairwise interaction terms in the fitted EBM.
    Uses a three-method cascade to handle different InterpretML versions:
      Method A: String delimiter parsing (" x ", " & ", " × ")
      Method B: Tuple of integer indices → map to feature names
                Tuple of string names → use directly
      Method C: Structural — explain_global().data(i)["names"] has 2D shape
                with each axis being an array of numeric bin edges (not a string)
    """
    interactions_found = []
    print("\n  ── DISCOVERED EBM INTERACTIONS ──")

    # Pre-compute global explanation for structural detection (Method C)
    try:
        _global = ebm.explain_global()
    except Exception:
        _global = None

    for term_idx, term in enumerate(ebm.term_names_):
        fa, fb   = None, None
        term_str = str(term)

        # Method A: string delimiter
        for delim in (" x ", " & ", " × "):
            if delim in term_str:
                parts = [p.strip() for p in term_str.split(delim)]
                if len(parts) == 2:
                    fa, fb = parts[0], parts[1]
                break

        # Method B: tuple/list of two items
        if fa is None and isinstance(term, (list, tuple)) and len(term) == 2:
            t0, t1 = term[0], term[1]
            if isinstance(t0, (int, np.integer)) and isinstance(t1, (int, np.integer)):
                fa = feature_names[t0] if t0 < len(feature_names) else str(t0)
                fb = feature_names[t1] if t1 < len(feature_names) else str(t1)
            elif isinstance(t0, str) and isinstance(t1, str):
                fa, fb = t0, t1

        # Method C: structural — 2D names array in explain_global data
        if fa is None and _global is not None:
            try:
                _td = _global.data(term_idx)
                if _td and "names" in _td:
                    _tn = _td["names"]
                    if (isinstance(_tn, (list, tuple)) and len(_tn) == 2
                            and hasattr(_tn[0], "__len__")
                            and hasattr(_tn[1], "__len__")
                            and not isinstance(_tn[0], str)
                            and not isinstance(_tn[1], str)):
                        # It's an interaction — identify features via term_features_
                        if hasattr(ebm, "term_features_") and term_idx < len(ebm.term_features_):
                            tf = ebm.term_features_[term_idx]
                            if isinstance(tf, (list, tuple)) and len(tf) == 2:
                                fa = feature_names[tf[0]] if tf[0] < len(feature_names) else str(tf[0])
                                fb = feature_names[tf[1]] if tf[1] < len(feature_names) else str(tf[1])
                        # Fallback: parse from term_str
                        if fa is None:
                            for delim in (" x ", " & ", " × ", ", "):
                                if delim in term_str:
                                    parts = [p.strip() for p in term_str.split(delim)]
                                    if len(parts) == 2:
                                        fa, fb = parts[0], parts[1]
                                    break
                        # Last resort: placeholder names
                        if fa is None:
                            fa = f"Feature_{term_idx}a"
                            fb = f"Feature_{term_idx}b"
            except Exception:
                pass

        if fa is not None and fb is not None:
            interactions_found.append((fa, fb))
            print(f"    [{term_idx:>2}] {fa}  ×  {fb}")

    if not interactions_found:
        print("    (No interaction terms detected — check InterpretML version)")

    # Must-include coverage check
    must_names = set()
    for f_a, f_b in MUST_INCLUDE_INTERACTIONS:
        must_names.add(frozenset([f_a, f_b]))
    found_sets = {frozenset(pair) for pair in interactions_found}

    print(f"\n  ── MUST-INCLUDE COVERAGE ({len(found_sets & must_names)}/{len(must_names)}) ──")
    for f_a, f_b in MUST_INCLUDE_INTERACTIONS:
        key    = frozenset([f_a, f_b])
        status = "✓ Found" if key in found_sets else "✗ NOT found"
        print(f"    {f_a:<40} × {f_b:<40} : {status}")

    return interactions_found


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def train_residual_ebm(data_path: str = BASELINE_DATA_PATH) -> dict:
    from interpret.glassbox import ExplainableBoostingRegressor

    # ── 1. LOAD DATA ─────────────────────────────────────────────────────────
    print(f"\nLoading enriched data from '{data_path}'…")
    df = pd.read_csv(data_path)
    n  = len(df)
    required = ["GLM_Pure_Premium", "Expected_Pure_Premium", "Split"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing: {missing_cols}")
    print(f"  {n:,} policies  (train: {(df['Split']=='train').sum():,}  test: {(df['Split']=='test').sum():,})")

    # ── 2. FEATURE ENGINEERING ───────────────────────────────────────────────
    print("Adding derived EBM features (Dwelling_Age, RCV_Overstatement, Log_AOI)…")
    df = _add_derived_features(df)
    df = _cast_cat_cols(df)
    missing_feats = [f for f in EBM_ALL_FEATURES if f not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing EBM features: {missing_feats}")
    print(f"  EBM feature count: {len(EBM_ALL_FEATURES)} (25 base + 3 derived = 28 total)")

    # ── 3. SPLIT ─────────────────────────────────────────────────────────────
    train_mask = df["Split"] == "train"
    test_mask  = df["Split"] == "test"
    df_train   = df[train_mask].copy()
    df_test    = df[test_mask].copy()
    X_train    = df_train[EBM_ALL_FEATURES].copy()
    X_test     = df_test[EBM_ALL_FEATURES].copy()

    # ── 4. LOG-SCALE RESIDUAL TARGET ─────────────────────────────────────────
    eps            = 1e-6
    log_true_train = np.log(df_train["Expected_Pure_Premium"].values + eps)
    log_glm_train  = np.log(df_train["GLM_Pure_Premium"].values       + eps)
    y_log_train    = log_true_train - log_glm_train
    log_true_test  = np.log(df_test["Expected_Pure_Premium"].values + eps)
    log_glm_test   = np.log(df_test["GLM_Pure_Premium"].values       + eps)
    y_log_test     = log_true_test - log_glm_test

    print(f"\n  Log-residual (train): mean={y_log_train.mean():.4f}  std={y_log_train.std():.4f}  "
          f"p5={np.percentile(y_log_train,5):.3f}  p95={np.percentile(y_log_train,95):.3f}")

    # ── 5. PRE-TRAINING RISK-NEUTRALITY CHECK ────────────────────────────────
    glm_weights_train    = df_train["GLM_Pure_Premium"].values
    weighted_mean_target = np.average(y_log_train, weights=glm_weights_train)
    print(f"\n  Pre-training neutrality check:")
    print(f"    GLM-weighted mean log-residual (train): {weighted_mean_target:.6f}")
    flag = "OK" if abs(weighted_mean_target) <= 0.05 else "WARNING: |mean| > 0.05"
    print(f"    {flag}")

    # ── 6. BUILD FORCED-INTERACTION LIST ─────────────────────────────────────
    forced_pairs = _build_interaction_list(EBM_ALL_FEATURES)
    N_AUTO       = 9
    print(f"\n  Interaction strategy: {len(forced_pairs)} forced + {N_AUTO} auto-discovered")

    # ── 7. FIT EBM — THREE-STRATEGY CASCADE (interaction spec fix) ───────────
    ebm_kwargs = dict(
        feature_names        = EBM_ALL_FEATURES,
        max_bins             = EBM_MAX_BINS,
        max_interaction_bins = EBM_MAX_INT_BINS,
        learning_rate        = EBM_LEARNING_RATE,
        outer_bags           = EBM_OUTER_BAGS,
        inner_bags           = EBM_INNER_BAGS,
        random_state         = RANDOM_STATE,
    )

    ebm               = None
    _interaction_mode = None

    # Strategy A: mixed list [forced_tuples + integer N_AUTO] — InterpretML ≥ 0.6.2
    try:
        print(f"  Strategy A: mixed interactions list (6 forced + {N_AUTO} auto)…")
        ebm = ExplainableBoostingRegressor(
            interactions=forced_pairs + [N_AUTO], **ebm_kwargs)
        ebm.fit(X_train, y_log_train)
        _interaction_mode = "mixed_list"
        print("  ✓ Strategy A succeeded (mixed mode)")
    except (TypeError, ValueError) as e:
        print(f"  ✗ Strategy A failed: {e}")
        ebm = None

    # Strategy B: scout pass → extract top 9 auto pairs → explicit tuple list
    if ebm is None:
        try:
            print("  Strategy B: scout EBM → extract top pairs → explicit tuple list…")
            ebm_scout = ExplainableBoostingRegressor(
                interactions=15,
                max_bins=EBM_MAX_BINS,
                learning_rate=0.05,    # faster for scouting
                outer_bags=2,          # less bagging for speed
                inner_bags=0,
                random_state=RANDOM_STATE,
            )
            ebm_scout.fit(X_train, y_log_train)

            # Extract auto-discovered interaction index pairs from scout
            auto_interactions = []
            for term in ebm_scout.term_names_:
                if isinstance(term, (list, tuple)) and len(term) == 2:
                    auto_interactions.append(tuple(term))
                elif isinstance(term, str) and " x " in term:
                    parts = [t.strip() for t in term.split(" x ")]
                    if len(parts) == 2:
                        try:
                            auto_interactions.append(
                                (EBM_ALL_FEATURES.index(parts[0]),
                                 EBM_ALL_FEATURES.index(parts[1])))
                        except ValueError:
                            pass

            # Merge: must-include first, then auto-discovered (deduplicated)
            must_set      = {tuple(sorted(p)) for p in forced_pairs}
            auto_filtered = [p for p in auto_interactions
                             if tuple(sorted(p)) not in must_set]
            all_interactions = forced_pairs + auto_filtered[:N_AUTO]

            ebm = ExplainableBoostingRegressor(
                interactions=all_interactions[:15], **ebm_kwargs)
            ebm.fit(X_train, y_log_train)
            _interaction_mode = "explicit_tuples"
            print(f"  ✓ Strategy B succeeded ({len(all_interactions[:15])} explicit interactions)")
        except Exception as e:
            print(f"  ✗ Strategy B failed: {e}")
            ebm = None

    # Strategy C: pure auto-discovery fallback
    if ebm is None:
        print(f"  Strategy C: pure auto-discovery (interactions={len(forced_pairs) + N_AUTO})…")
        ebm = ExplainableBoostingRegressor(
            interactions=len(forced_pairs) + N_AUTO, **ebm_kwargs)
        ebm.fit(X_train, y_log_train)
        _interaction_mode = "auto_discovery"
        print(f"  ⚠ Strategy C used — narrative pairs NOT guaranteed")

    print(f"  Interaction mode: {_interaction_mode}")

    # ── 8. PRINT DISCOVERED INTERACTIONS ─────────────────────────────────────
    interactions_found = _verify_discovered_interactions(ebm, EBM_ALL_FEATURES)

    # ── 9. PREDICT ON FULL DATASET ────────────────────────────────────────────
    print("\n  Generating predictions on full dataset…")
    raw_log_all = ebm.predict(df[EBM_ALL_FEATURES].copy())
    clipped_log = np.clip(raw_log_all, LOG_MIN, LOG_MAX)
    raw_uplift  = np.exp(clipped_log)

    # ── 10. RISK-NEUTRALITY NORMALISATION ────────────────────────────────────
    glm_weights_all   = df["GLM_Pure_Premium"].values
    raw_weighted_mean = np.average(raw_uplift, weights=glm_weights_all)
    norm_uplift       = raw_uplift / raw_weighted_mean

    check_val = np.average(norm_uplift, weights=glm_weights_all)
    assert abs(check_val - 1.0) < 1e-6, (
        f"Risk neutrality assertion failed: weighted mean = {check_val:.8f}")

    print(f"\n  ── RISK NEUTRALITY ──")
    print(f"    Raw weighted mean uplift   : {raw_weighted_mean:.6f}×")
    print(f"    Post-norm weighted mean    : {check_val:.6f}×  ✓")
    diff_dollars = (glm_weights_all * norm_uplift).sum() - glm_weights_all.sum()
    print(f"    Book-level difference      : ${diff_dollars:,.0f}  "
          f"({'≈$0 ✓' if abs(diff_dollars) < 1 else 'WARNING'})")

    # ── 11. WRITE PREDICTION COLUMNS ─────────────────────────────────────────
    df["EBM_Log_Uplift"]     = clipped_log
    df["EBM_Uplift_Factor"]  = norm_uplift
    df["Final_Pure_Premium"] = np.clip(df["GLM_Pure_Premium"] * norm_uplift, PREMIUM_FLOOR, None)
    df["EBM_Residual_Pred"]  = df["Final_Pure_Premium"] - df["GLM_Pure_Premium"]
    df["Adjustment_Pct"]     = (df["EBM_Uplift_Factor"] - 1.0) * 100.0
    df["GLM_Risk_Tier"]      = _assign_tier(df["GLM_Pure_Premium"].values)
    df["Final_Risk_Tier"]    = _assign_tier(df["Final_Pure_Premium"].values)

    # ── 12. OOS PERFORMANCE (TEST SET ONLY) ──────────────────────────────────
    df_test_eval  = df[test_mask]
    true_pp_test  = df_test_eval["Expected_Pure_Premium"].values
    glm_pp_test   = df_test_eval["GLM_Pure_Premium"].values
    fin_pp_test   = df_test_eval["Final_Pure_Premium"].values
    glm_r2_test   = r2_score(true_pp_test, glm_pp_test)
    final_r2_test = r2_score(true_pp_test, fin_pp_test)
    glm_rmse_test  = np.sqrt(mean_squared_error(true_pp_test, glm_pp_test))
    fin_rmse_test  = np.sqrt(mean_squared_error(true_pp_test, fin_pp_test))
    delta_r2       = final_r2_test - glm_r2_test

    test_raw_log    = ebm.predict(df_test[EBM_ALL_FEATURES].copy())
    test_clipped    = np.clip(test_raw_log, LOG_MIN, LOG_MAX)
    ebm_log_r2_test = r2_score(y_log_test, test_clipped)

    print(f"\n{'='*64}")
    print(f"  MODEL PERFORMANCE  (OOS test set, N={test_mask.sum():,})")
    print(f"{'='*64}")
    print(f"  {'Legacy GLM (16 feat, linear)':<35} R²={glm_r2_test:.4f}  "
          f"RMSE=${glm_rmse_test:>9,.0f}  Mean=${glm_pp_test.mean():>9,.0f}")
    print(f"  {'GLM + GA2M (28 feat, glass-box)':<35} R²={final_r2_test:.4f}  "
          f"RMSE=${fin_rmse_test:>9,.0f}  Mean=${fin_pp_test.mean():>9,.0f}")
    print(f"  {'Incremental ΔR²':<35} {delta_r2:>+8.4f}")
    print(f"  {'EBM log-residual R² (OOS)':<35} {ebm_log_r2_test:>8.4f}")
    print(f"{'='*64}")

    if delta_r2 < 0.05:
        print(f"  WARNING: ΔR²={delta_r2:.4f} below target [0.07, 0.12]")
    elif delta_r2 > 0.15:
        print(f"  WARNING: ΔR²={delta_r2:.4f} above target [0.07, 0.12]")
    else:
        print(f"  ΔR²={delta_r2:.4f} within target [0.07, 0.12] ✓")

    # ── 13. PREMIUM MIGRATION SUMMARY ────────────────────────────────────────
    adj      = df["EBM_Residual_Pred"]
    up_mask  = adj > 0
    dn_mask  = adj < 0
    premium_up   = adj[up_mask].sum()
    premium_down = adj[dn_mask].abs().sum()
    pct_up       = up_mask.mean() * 100
    pct_down     = dn_mask.mean() * 100
    from_glm_tiers = df["GLM_Risk_Tier"]
    to_final_tiers = df["Final_Risk_Tier"]
    tier_moves     = (from_glm_tiers != to_final_tiers).sum()

    print(f"\n  ── PREMIUM MIGRATION ──")
    print(f"    Policies receiving surcharge    : {up_mask.sum():>7,}  ({pct_up:.1f}%)  +${premium_up:,.0f}")
    print(f"    Policies receiving credit       : {dn_mask.sum():>7,}  ({pct_down:.1f}%)  −${premium_down:,.0f}")
    print(f"    Net premium migration           : ${premium_up - premium_down:>+,.0f}")
    print(f"    Tier reclassifications          : {tier_moves:>7,}  ({tier_moves/n:.1%})")

    print(f"\n  ── TIER MIGRATION MATRIX (full portfolio) ──")
    for t_from in TIER_ORDER:
        for t_to in TIER_ORDER:
            if t_from == t_to: continue
            mask  = (from_glm_tiers == t_from) & (to_final_tiers == t_to)
            count = mask.sum()
            if count > 0:
                print(f"  {t_from:>12}  →  {t_to:<12}  {count:>6,}    {count/n:.2%}")

    # ── 14. UPLIFT CORRIDOR DIAGNOSTICS ──────────────────────────────────────
    at_floor = (clipped_log <= LOG_MIN + 0.001).sum()
    at_ceil  = (clipped_log >= LOG_MAX - 0.001).sum()
    print(f"\n  ── UPLIFT CORRIDOR ──")
    print(f"    Corridor: [{MIN_UPLIFT:.2f}×, {MAX_UPLIFT:.2f}×]")
    print(f"    At floor ({MIN_UPLIFT:.2f}×)   : {at_floor:,}  ({at_floor/n:.2%})")
    print(f"    At ceiling ({MAX_UPLIFT:.2f}×) : {at_ceil:,}  ({at_ceil/n:.2%})")
    print(f"    Uplift factor range : {norm_uplift.min():.4f}× – {norm_uplift.max():.4f}×")

    # ── 15. SAVE ARTIFACTS ───────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(FINAL_DATA_PATH), exist_ok=True)
    df.to_csv(FINAL_DATA_PATH, index=False)
    print(f"\n  Saved final predictions to '{FINAL_DATA_PATH}'")
    joblib.dump(ebm, EBM_MODEL_PATH)
    print(f"  Saved EBM model to '{EBM_MODEL_PATH}'")

    existing_meta = {}
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH) as f:
                existing_meta = json.load(f)
        except Exception:
            pass

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

    # ── 16. RETURN METRICS ────────────────────────────────────────────────────
    return {
        "final_r2":           f"{final_r2_test:.4f}",
        "delta_r2":           f"+{delta_r2:.4f}",
        "risk_neutral_check": f"{check_val:.6f}× (target: 1.000000)",
        "interactions_found": len(interactions_found),
        "book_delta":         f"${diff_dollars:,.0f}",
    }


if __name__ == "__main__":
    metrics = train_residual_ebm(BASELINE_DATA_PATH)
    print(f"\nEBM residual model complete.")
    print(f"  Final R² (OOS) : {metrics['final_r2']}")
    print(f"  ΔR²            : {metrics['delta_r2']}")
    print(f"  Risk neutrality: {metrics['risk_neutral_check']}")
    print(f"  Book impact    : {metrics['book_delta']}")
    print("Run 'python app.py' next to launch the demo.\n")