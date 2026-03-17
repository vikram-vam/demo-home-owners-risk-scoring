# ==============================================================================
# data_simulation.py  —  Phase 2 Rewrite
# Realistic synthetic homeowners data for ResiScore™ GLM + GA2M demo
#
# Architecture (per Spec S4-Revised + Supplement S-A.2):
#   1.  State assignment (10 US states + Other, conditional peril distributions)
#   2.  Feature generation from marginal distributions
#   3.  Iman-Conover rank-correlation permutation (Gaussian copula surrogate)
#   4.  State-conditional quantile mapping for Wildfire / Hail / Flood
#   5.  Categorical features and alarms (independent)
#   6.  CLUE-conditional Water Loss Recency; Credit Score suppression (CA, MA)
#   7.  Derived features: Dwelling_Age, RCV_Overstatement
#   8.  Legacy log-signal  (linear; standardised × 0.32)  ← GLM territory
#   9.  Modern log-signal  split into two separate budgets:
#         modern_main_signal  (non-linear main effects; standardised × 0.16)
#         interaction_signal  (pairwise interactions;   standardised × 0.12)
#       Total modern variance: 0.16² + 0.12² = 0.04 = old 0.20² (unchanged)
#       Interactions now get 36% of modern budget instead of competing at ~20%.
#  10.  Noise (σ = 0.25)
#  11.  log(λ_i) = base_rate + legacy_scaled + modern_main_scaled
#                            + interaction_scaled + noise
#  12.  Severity mean (AOI-anchored, log-linear, mostly GLM-recoverable)
#  13.  Expected_Pure_Premium = λ_i × severity_i  (oracle target)
#  14.  Claim simulation: Poisson(λ_calibrated) × Gamma(2.5, sev/2.5)
#  15.  Validation printout with spot-checks
# ==============================================================================

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    N_SAMPLES,
    RANDOM_STATE,
    PREMIUM_FLOOR,
    DGP_LEGACY_SCALAR,
    DGP_MODERN_MAIN_SCALAR,
    DGP_INTERACTION_SCALAR,
    DGP_NOISE_SIGMA,
    BASE_CLAIM_RATE,
    BASE_SEVERITY,
    BASE_LOG_FREQ,
    STATE_CONFIG,
    CORRELATION_PAIRS,
    COPULA_CONTINUOUS_FEATURES,
    CREDIT_SUPPRESSED_STATES,
    TIER_BOUNDARIES,
    TIER_ORDER,
)

_COPULA_FEATS = COPULA_CONTINUOUS_FEATURES + ["Deductible"]
_N_COPULA     = len(_COPULA_FEATS)
_COPULA_IDX   = {f: i for i, f in enumerate(_COPULA_FEATS)}


def _build_psd_corr(feature_names, pairs):
    n   = len(feature_names)
    idx = {f: i for i, f in enumerate(feature_names)}
    C   = np.eye(n, dtype=float)
    for fa, fb, rho in pairs:
        if fa in idx and fb in idx:
            i, j    = idx[fa], idx[fb]
            C[i, j] = rho
            C[j, i] = rho
    from scipy.linalg import eigh
    eigvals, eigvecs = eigh(C)
    eigvals          = np.clip(eigvals, 1e-8, None)
    C_psd            = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d                = np.sqrt(np.diag(C_psd))
    C_psd            = C_psd / np.outer(d, d)
    np.fill_diagonal(C_psd, 1.0)
    return C_psd


def _iman_conover(feature_matrix, corr_matrix, rng):
    n, p = feature_matrix.shape
    try:
        chol = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        chol = np.linalg.cholesky(corr_matrix + np.eye(p) * 1e-7)
    Z      = rng.standard_normal((n, p)) @ chol.T
    result = feature_matrix.copy().astype(float)
    for j in range(p):
        z_ranks      = np.argsort(np.argsort(Z[:, j]))
        sorted_vals  = np.sort(result[:, j])
        result[:, j] = sorted_vals[z_ranks]
    return result


def _wildfire_ppf(uniform_q, mode, n_grid=60_000, seed=777):
    rng_loc = np.random.default_rng(seed)
    if mode == "high_bimodal":
        n_lo  = int(n_grid * 0.70)
        samp  = np.concatenate([
            rng_loc.beta(1.5, 8,    n_lo) * 100,
            rng_loc.beta(3,   2, n_grid - n_lo) * 100,
        ])
    elif mode == "medium_bimodal":
        n_lo  = int(n_grid * 0.80)
        samp  = np.concatenate([
            rng_loc.beta(1.5, 8,       n_lo) * 100,
            rng_loc.beta(2.5, 2.5, n_grid - n_lo) * 100,
        ])
    else:
        samp  = rng_loc.beta(0.5, 4, n_grid) * 100
    samp   = np.clip(samp, 0.0, 100.0)
    grid_q = np.linspace(0.0, 1.0, n_grid)
    return np.interp(uniform_q, grid_q, np.sort(samp))


def _assign_tier(premium):
    t = pd.cut(
        premium,
        bins=[0, 1_000, 2_000, 3_500, np.inf],
        labels=TIER_ORDER,
    )
    return t.astype(str)


def generate_homeowners_data(n_samples: int = N_SAMPLES,
                             random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    np.random.seed(random_state)
    n = n_samples
    print(f"\nGenerating {n:,} synthetic homeowners policies…")

    # ── STEP 1: State assignment ──────────────────────────────────────────────
    states   = list(STATE_CONFIG.keys())
    weights  = np.array([STATE_CONFIG[s][0] for s in states])
    weights /= weights.sum()
    State    = rng.choice(states, size=n, p=weights)

    wf_mode_arr  = np.array([STATE_CONFIG[s][1] for s in State])
    hail_lam_arr = np.array([STATE_CONFIG[s][2] for s in State], dtype=float)
    flood_sc_arr = np.array([STATE_CONFIG[s][3] for s in State], dtype=float)
    terr_p_arr   = np.array([STATE_CONFIG[s][4] for s in State])
    aoi_mult_arr = np.array([STATE_CONFIG[s][5] for s in State], dtype=float)

    # ── STEP 2: Independent marginal generation ───────────────────────────────
    Year_Built         = rng.integers(1950, 2024, n).astype(float)
    Square_Footage     = rng.normal(2200, 600, n).clip(800, 5000)
    CLUE_Loss_Count    = rng.poisson(0.30, n).clip(0, 8).astype(float)
    Credit_Score       = rng.normal(700, 80, n).clip(300, 850)
    Protection_Class   = rng.integers(1, 11, n).astype(float)
    Roof_Age_Applicant = rng.integers(1, 31, n).astype(float)
    price_per_sqft     = rng.uniform(150, 260, n) * aoi_mult_arr
    AOI                = (Square_Footage * price_per_sqft).clip(80_000, 2_000_000)
    rcv_per_sqft       = rng.uniform(140, 220, n)
    RCV_Appraised      = (Square_Footage * rcv_per_sqft).clip(60_000, 1_500_000)
    Building_Code_Compliance = rng.uniform(30.0, 100.0, n)
    Wildfire_Exposure_Daily  = rng.beta(0.5, 2.0, n) * 100
    Fire_Hydrant_Distance    = rng.lognormal(-0.5, 0.8, n).clip(0.05, 10.0)
    Tree_Canopy_Density      = rng.beta(2, 5, n) * 100
    Crime_Severity_Index     = rng.normal(50, 20, n).clip(0, 100)
    Pluvial_Flood_Depth      = rng.exponential(5, n).clip(0, 36)
    Slope_Steepness          = rng.exponential(10, n).clip(0, 45)
    Hail_Frequency           = rng.poisson(1.5, n).astype(float)
    Roof_Vulnerability_Satellite = (Roof_Age_Applicant + rng.normal(1.0, 4.0, n)).clip(0.0, 38.0)
    Water_Loss_Recency_Months    = rng.uniform(1.0, 120.0, n)
    Deductible_raw = rng.choice([500.0, 1000.0, 2000.0, 5000.0], n, p=[0.15, 0.50, 0.25, 0.10])

    # ── STEP 3: Iman-Conover rank correlation ─────────────────────────────────
    feat_raw = np.column_stack([
        Year_Built, Square_Footage, CLUE_Loss_Count, Credit_Score,
        Protection_Class, AOI, Roof_Age_Applicant, Roof_Vulnerability_Satellite,
        Wildfire_Exposure_Daily, Water_Loss_Recency_Months, RCV_Appraised,
        Fire_Hydrant_Distance, Tree_Canopy_Density, Crime_Severity_Index,
        Pluvial_Flood_Depth, Building_Code_Compliance, Slope_Steepness,
        Hail_Frequency, Deductible_raw,
    ])
    corr_mat  = _build_psd_corr(_COPULA_FEATS, CORRELATION_PAIRS)
    print("  Applying Iman-Conover rank correlation (copula)…")
    feat_corr = _iman_conover(feat_raw, corr_mat, rng)

    (Year_Built, Square_Footage, CLUE_Loss_Count, Credit_Score,
     Protection_Class, AOI, Roof_Age_Applicant, Roof_Vulnerability_Satellite,
     Wildfire_Exposure_Daily, Water_Loss_Recency_Months, RCV_Appraised,
     Fire_Hydrant_Distance, Tree_Canopy_Density, Crime_Severity_Index,
     Pluvial_Flood_Depth, Building_Code_Compliance, Slope_Steepness,
     Hail_Frequency, Deductible_raw) = feat_corr.T

    Year_Built               = np.clip(Year_Built.round(0),  1950, 2023).astype(int)
    Square_Footage           = np.clip(Square_Footage, 800, 5000).round(0)
    CLUE_Loss_Count          = np.clip(CLUE_Loss_Count.round(0), 0, 8).astype(int)
    Credit_Score             = np.clip(Credit_Score, 300, 850).round(0)
    Protection_Class         = np.clip(Protection_Class.round(0), 1, 10).astype(int)
    Roof_Age_Applicant       = np.clip(Roof_Age_Applicant.round(0), 1, 30).astype(int)
    AOI                      = np.clip(AOI, 80_000, 2_000_000).round(0)
    RCV_Appraised            = np.clip(RCV_Appraised, 60_000, 1_500_000).round(0)
    Building_Code_Compliance = np.clip(Building_Code_Compliance, 30, 100).round(0).astype(int)
    Fire_Hydrant_Distance    = np.clip(Fire_Hydrant_Distance, 0.05, 10.0).round(3)
    Tree_Canopy_Density      = np.clip(Tree_Canopy_Density, 0, 100).round(2)
    Crime_Severity_Index     = np.clip(Crime_Severity_Index, 0, 100).round(2)
    Slope_Steepness          = np.clip(Slope_Steepness, 0, 45).round(2)
    Hail_Frequency           = np.clip(Hail_Frequency.round(0), 0, 15).astype(int)
    Roof_Vulnerability_Satellite = np.clip(Roof_Vulnerability_Satellite, 0, 38).round(2)
    _ded_levels = np.array([500.0, 1000.0, 2000.0, 5000.0])
    Deductible  = _ded_levels[
        np.argmin(np.abs(Deductible_raw[:, None] - _ded_levels[None, :]), axis=1)
    ].astype(int)

    # ── STEP 4: State-conditional quantile mapping ────────────────────────────
    from scipy.stats import rankdata as _rankdata

    wf_unif = _rankdata(Wildfire_Exposure_Daily) / (n + 1)
    wf_new  = np.zeros(n)
    for s, cfg in STATE_CONFIG.items():
        mask = (State == s)
        if not mask.any(): continue
        wf_new[mask] = _wildfire_ppf(wf_unif[mask], cfg[1])
    Wildfire_Exposure_Daily = np.clip(wf_new, 0, 100).round(2)

    hail_unif = np.clip(_rankdata(Hail_Frequency) / (n + 1), 0.001, 0.999)
    hail_new  = np.zeros(n)
    for s, cfg in STATE_CONFIG.items():
        mask = (State == s)
        if not mask.any(): continue
        hail_new[mask] = stats.poisson.ppf(hail_unif[mask], mu=cfg[2])
    Hail_Frequency = np.clip(hail_new.round(0), 0, 15).astype(int)

    flood_unif = np.clip(_rankdata(Pluvial_Flood_Depth) / (n + 1), 0.001, 0.999)
    flood_new  = np.zeros(n)
    for s, cfg in STATE_CONFIG.items():
        mask = (State == s)
        if not mask.any(): continue
        flood_new[mask] = stats.expon.ppf(flood_unif[mask], scale=cfg[3])
    Pluvial_Flood_Depth = np.clip(flood_new, 0, 36).round(2)

    territory_cats = ["Urban", "Suburban", "Rural"]
    Territory = np.empty(n, dtype=object)
    for s, cfg in STATE_CONFIG.items():
        mask = (State == s)
        if not mask.any(): continue
        Territory[mask] = rng.choice(territory_cats, size=mask.sum(), p=cfg[4])

    # ── STEP 5: Categorical features ──────────────────────────────────────────
    Construction_Type      = rng.choice(["Frame", "Masonry", "Fire Resistive"], n, p=[0.70, 0.20, 0.10])
    Attic_Ventilation      = rng.choice(["Poor", "Adequate", "Excellent"],       n, p=[0.30, 0.50, 0.20])
    Soil_Liquefaction_Risk = rng.choice(["Low", "Moderate", "High"],             n, p=[0.70, 0.20, 0.10])
    Fire_Alarm             = rng.binomial(1, 0.40, n).astype(bool)
    Burglar_Alarm          = rng.binomial(1, 0.30, n).astype(bool)

    # ── STEP 6: CLUE-conditional water loss recency ───────────────────────────
    has_prior_claim = (CLUE_Loss_Count > 0)
    wlr_unif        = _rankdata(Water_Loss_Recency_Months) / (n + 1)
    Water_Loss_Recency_Months = np.where(
        has_prior_claim,
        np.clip((wlr_unif * 35 + 1).round(0), 1, 36),
        120,
    ).astype(int)

    # ── STEP 7: Credit score suppression (CA, MA) ─────────────────────────────
    Credit_Score_Suppressed          = np.isin(State, sorted(CREDIT_SUPPRESSED_STATES))
    Credit_Score                     = Credit_Score.copy()
    Credit_Score[Credit_Score_Suppressed] = 700.0

    # ── STEP 8: Derived features ──────────────────────────────────────────────
    Dwelling_Age      = (2026 - Year_Built).astype(int)
    RCV_Overstatement = np.maximum(0.0, AOI - RCV_Appraised)

    # ── STEP 9: Legacy log-signal ─────────────────────────────────────────────
    is_frame    = (Construction_Type == "Frame").astype(float)
    is_urban    = (Territory == "Urban").astype(float)
    terr_factor = np.where(Territory == "Urban",  0.12,
                  np.where(Territory == "Rural", -0.08, 0.0))
    ded_factor  = np.where(Deductible == 500,   0.08,
                  np.where(Deductible == 2000, -0.05,
                  np.where(Deductible == 5000, -0.10, 0.0)))

    legacy_raw = (
          (Protection_Class - 5) * 0.06
        + CLUE_Loss_Count * 0.20
        + (700 - Credit_Score) / 1000 * 1.50
        + Dwelling_Age * 0.008
        + is_frame * 0.18
        + terr_factor
        + ded_factor
        - Fire_Alarm.astype(float) * 0.08
        - Burglar_Alarm.astype(float) * 0.06
        + Roof_Age_Applicant * 0.005
        + (is_frame * (Protection_Class > 6)) * 0.12
        + ((CLUE_Loss_Count >= 2) * (Deductible <= 500)) * 0.10
        + (is_urban * (Protection_Class > 6)) * 0.08
        + ((Roof_Age_Applicant > 20) * (Hail_Frequency >= 3)) * 0.10
    )

    # ── STEP 10: Modern log-signal — SPLIT INTO TWO SEPARATE BUDGETS ─────────
    #
    # Interaction spec change: single combined scalar (0.20) is replaced by:
    #   DGP_MODERN_MAIN_SCALAR = 0.16  (non-linear main effects)
    #   DGP_INTERACTION_SCALAR = 0.12  (pairwise interactions)
    #
    # Total variance budget preserved: 0.16² + 0.12² = 0.0400 = 0.20²
    # Interactions now get 36% of modern budget (was ~20%).
    # Top 3 interaction coefficients amplified ~30%.

    exp_water_decay = np.exp(-Water_Loss_Recency_Months / 12.0)

    # Non-linear main effects (GA2M univariate shape function territory)
    modern_main_raw = (
          0.0003 * np.maximum(0, Wildfire_Exposure_Daily - 30) ** 2     # CONVEX wildfire
        + (Roof_Vulnerability_Satellite / 20) ** 2 * 0.18                # CONVEX roof decay
        + np.log1p(Fire_Hydrant_Distance) * 0.10                         # LOG diminishing
        + (Building_Code_Compliance < 60).astype(float) * 0.15           # THRESHOLD
        + exp_water_decay * 0.18                                          # EXPONENTIAL DECAY
    )

    # Pairwise interaction effects — amplified top 3 (~+30% vs original)
    rcv_overstatement_norm = RCV_Overstatement / 100_000
    interaction_raw = (
        # Wildfire × Roof Vulnerability — 0.28 → 0.36 (flagship)
          (Wildfire_Exposure_Daily / 100) * (Roof_Vulnerability_Satellite / 20) * 0.36
        # Water Recency × Tree Canopy — 0.22 → 0.28
        + exp_water_decay * (Tree_Canopy_Density / 100) * 0.28
        # RCV Overstatement × Crime — 0.18 → 0.24
        + rcv_overstatement_norm * (Crime_Severity_Index / 100) * 0.24
        # Pluvial Flood × Dwelling Age — 0.18 → 0.20 (kept)
        + (Pluvial_Flood_Depth > 15).astype(float) * (Dwelling_Age > 35).astype(float) * 0.20
        # Slope × Wildfire — 0.14 → 0.18
        + (Slope_Steepness / 45) * (Wildfire_Exposure_Daily / 100) * 0.18
        # Hail × Roof Vulnerability — 0.18 → 0.20 (kept)
        + (Hail_Frequency > 3).astype(float) * (Roof_Vulnerability_Satellite > 18).astype(float) * 0.20
    )

    # ── STEP 11: Standardise and scale each budget separately ────────────────
    legacy_scaled = (
        (legacy_raw - legacy_raw.mean()) / legacy_raw.std()
    ) * DGP_LEGACY_SCALAR

    modern_main_scaled = (
        (modern_main_raw - modern_main_raw.mean()) / modern_main_raw.std()
    ) * DGP_MODERN_MAIN_SCALAR

    interaction_scaled = (
        (interaction_raw - interaction_raw.mean()) / interaction_raw.std()
    ) * DGP_INTERACTION_SCALAR

    noise = rng.normal(0.0, DGP_NOISE_SIGMA, n)

    # ── STEP 12: Expected Pure Premium (AOI-anchored) ─────────────────────────
    RATE_PER_1K   = 3.75
    base_log      = np.log(RATE_PER_1K) + np.log(AOI / 1_000.0)
    log_epp       = base_log + legacy_scaled + modern_main_scaled + interaction_scaled + noise
    epp_raw       = np.clip(np.exp(log_epp), PREMIUM_FLOOR, None)
    lambda_freq   = np.clip((epp_raw / epp_raw.mean()) * BASE_CLAIM_RATE, 0.001, 0.40)
    severity_mean = epp_raw / lambda_freq
    Expected_Pure_Premium = epp_raw

    # ── STEP 13: Simulate claims ──────────────────────────────────────────────
    Claim_Count  = rng.poisson(lambda_freq)
    Claim_Amount = np.zeros(n)
    has_claim    = Claim_Count > 0
    n_claims     = has_claim.sum()
    if n_claims > 0:
        raw_sev = rng.gamma(shape=2.5, scale=severity_mean[has_claim] / 2.5, size=n_claims)
        Claim_Amount[has_claim] = raw_sev * Claim_Count[has_claim]
        sev_cap = np.percentile(Claim_Amount[has_claim], 99.5)
        Claim_Amount = np.minimum(Claim_Amount, sev_cap)

    # ── STEP 14: Risk tier labels ─────────────────────────────────────────────
    Risk_Tier = _assign_tier(Expected_Pure_Premium)

    # ── STEP 15: Assemble DataFrame ───────────────────────────────────────────
    data = pd.DataFrame({
        "State":                          State,
        "Year_Built":                     Year_Built,
        "Square_Footage":                 Square_Footage.round(0),
        "CLUE_Loss_Count":                CLUE_Loss_Count,
        "Credit_Score":                   Credit_Score.round(0),
        "Construction_Type":              Construction_Type,
        "Protection_Class":               Protection_Class,
        "AOI":                            AOI.round(0),
        "Deductible":                     Deductible,
        "Territory":                      Territory,
        "Roof_Age_Applicant":             Roof_Age_Applicant,
        "Fire_Alarm":                     Fire_Alarm,
        "Burglar_Alarm":                  Burglar_Alarm,
        "Roof_Vulnerability_Satellite":   Roof_Vulnerability_Satellite,
        "Wildfire_Exposure_Daily":        Wildfire_Exposure_Daily,
        "Water_Loss_Recency_Months":      Water_Loss_Recency_Months,
        "RCV_Appraised":                  RCV_Appraised.round(0),
        "Fire_Hydrant_Distance":          Fire_Hydrant_Distance,
        "Tree_Canopy_Density":            Tree_Canopy_Density,
        "Crime_Severity_Index":           Crime_Severity_Index,
        "Pluvial_Flood_Depth":            Pluvial_Flood_Depth,
        "Building_Code_Compliance":       Building_Code_Compliance,
        "Slope_Steepness":                Slope_Steepness,
        "Attic_Ventilation":              Attic_Ventilation,
        "Hail_Frequency":                 Hail_Frequency,
        "Soil_Liquefaction_Risk":         Soil_Liquefaction_Risk,
        "Dwelling_Age":                   Dwelling_Age,
        "RCV_Overstatement":              RCV_Overstatement.round(0),
        "Credit_Score_Suppressed":        Credit_Score_Suppressed,
        "Risk_Tier":                      Risk_Tier,
        "Expected_Pure_Premium":          Expected_Pure_Premium.round(2),
        "Claim_Count":                    Claim_Count,
        "Claim_Amount":                   Claim_Amount.round(2),
    })

    _print_validation(data, corr_mat, legacy_raw, modern_main_raw, interaction_raw, noise)
    return data


def _print_validation(df, corr_mat, legacy_raw, modern_main_raw, interaction_raw, noise):
    n      = len(df)
    epp    = df["Expected_Pure_Premium"]
    cc     = df["Claim_Count"]
    ca     = df["Claim_Amount"]
    has_cl = cc > 0

    print("\n" + "=" * 62)
    print("  DATA GENERATION — VALIDATION SUMMARY")
    print("=" * 62)
    print("\n  PREMIUM DISTRIBUTION")
    print(f"    Policies generated   : {n:,}")
    print(f"    Mean  EPP            : ${epp.mean():>10,.0f}  (target $1,500–$2,200)")
    print(f"    Median EPP           : ${epp.median():>10,.0f}")
    print(f"    p5  / p95            : ${np.percentile(epp, 5):>8,.0f} / ${np.percentile(epp, 95):>8,.0f}")
    print(f"    Min / Max            : ${epp.min():>8,.0f} / ${epp.max():>8,.0f}")

    print("\n  CLAIM CALIBRATION")
    sev_arr = ca[has_cl]
    print(f"    Claim rate           : {has_cl.mean():.2%}   (target 5.3–5.6%)")
    print(f"    Mean severity        : ${sev_arr.mean():>10,.0f}")

    print("\n  RISK TIER DISTRIBUTION  (approx target: 30/30/25/15)")
    for tier in ["Low", "Moderate", "Elevated", "High"]:
        pct = (df["Risk_Tier"] == tier).mean() * 100
        print(f"    {tier:<12}: {pct:5.1f}%")

    print("\n  STATE DISTRIBUTION")
    for state, cnt in df["State"].value_counts().sort_values(ascending=False).items():
        print(f"    {state:<8}: {cnt:>7,}  ({cnt/n:.1%})")

    suppressed = df["Credit_Score_Suppressed"].sum()
    print(f"\n  CREDIT SUPPRESSION  (CA + MA): {suppressed:,} ({suppressed/n:.1%})")

    print("\n  DGP VARIANCE DECOMPOSITION (interaction spec — split budget)")
    print(f"    Legacy std (raw)          : {legacy_raw.std():.4f}")
    print(f"    Main effects std (raw)    : {modern_main_raw.std():.4f}")
    print(f"    Interaction std (raw)     : {interaction_raw.std():.4f}")
    print(f"    Noise std                 : {noise.std():.4f}")
    print(f"    Scaled legacy             : {DGP_LEGACY_SCALAR:.2f}  (target 0.32)")
    print(f"    Scaled main effects       : {DGP_MODERN_MAIN_SCALAR:.2f}  (target 0.16)")
    print(f"    Scaled interaction        : {DGP_INTERACTION_SCALAR:.2f}  (target 0.12)")
    total_modern = DGP_MODERN_MAIN_SCALAR**2 + DGP_INTERACTION_SCALAR**2
    print(f"    Modern variance budget    : {total_modern:.4f} (= 0.20² = {0.20**2:.4f} ✓)")

    print("\n  CORRELATION SPOT-CHECKS  (target | actual Pearson ρ)")
    spot_checks = [
        ("Roof_Age_Applicant", "Year_Built",              -0.80),
        ("AOI",                "Square_Footage",           0.85),
        ("RCV_Appraised",      "Square_Footage",           0.88),
        ("Building_Code_Compliance", "Year_Built",         0.70),
        ("Credit_Score",       "CLUE_Loss_Count",         -0.30),
        ("Wildfire_Exposure_Daily", "Tree_Canopy_Density", 0.38),
    ]
    for fa, fb, target in spot_checks:
        if fa in df.columns and fb in df.columns:
            actual = df[fa].corr(df[fb])
            flag   = "✓" if abs(actual - target) < 0.15 else "✗ CHECK"
            print(f"    {fa:<32} × {fb:<28}: target={target:+.2f}  actual={actual:+.3f}  {flag}")

    print("\n  WILDFIRE DISTRIBUTION (mean by state)")
    for state, val in df.groupby("State")["Wildfire_Exposure_Daily"].mean().sort_values(ascending=False).items():
        print(f"    {state:<8}: {val:5.1f}")
    print("=" * 62)


if __name__ == "__main__":
    df = generate_homeowners_data(n_samples=N_SAMPLES, random_state=RANDOM_STATE)
    out = os.path.join("data", "synthetic_homeowners_data.csv")
    os.makedirs("data", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df):,} policies to '{out}'")
    print("Run 'python baseline_glm.py' next to train the GLM.\n")