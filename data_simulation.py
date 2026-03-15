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
#   8.  Legacy log-signal  (linear; standardised × 0.38)  ← GLM territory
#   9.  Modern log-signal  (non-linear + interactions; standardised × 0.14) ← GA2M territory
#  10.  Noise (σ = 0.21)
#  11.  log(λ_i) = base_rate + legacy_scaled + modern_scaled + noise
#  12.  Severity mean (AOI-anchored, log-linear, mostly GLM-recoverable)
#  13.  Expected_Pure_Premium = λ_i × severity_i  (oracle target)
#  14.  Claim simulation: Poisson(λ_calibrated) × Gamma(2.5, sev/2.5)
#  15.  Validation printout with spot-checks
#
# Design targets (OOS, evaluated in baseline_glm.py + residual_model.py):
#   GLM R²  vs EPP : 0.58–0.65
#   EBM ΔR²        : 0.07–0.12
#   Claim rate     : 5.3–5.6%  (III/ISO 2023 benchmark)
#   Mean EPP       : $1,500–$2,200
# ==============================================================================

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

# Allow running this file directly from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    N_SAMPLES,
    RANDOM_STATE,
    PREMIUM_FLOOR,
    DGP_LEGACY_SCALAR,    # 0.38
    DGP_MODERN_SCALAR,    # 0.14
    DGP_NOISE_SIGMA,      # 0.21
    BASE_CLAIM_RATE,      # 0.055
    BASE_SEVERITY,        # 15_000 (base severity; actual mean ~$30K after AOI scaling)
    BASE_LOG_FREQ,        # log(0.055)
    STATE_CONFIG,
    CORRELATION_PAIRS,
    COPULA_CONTINUOUS_FEATURES,
    CREDIT_SUPPRESSED_STATES,
    TIER_BOUNDARIES,
    TIER_ORDER,
)

# ── Copula feature list (18 from config + Deductible for Credit correlation) ──
# Note: Deductible is ordinal so Iman-Conover works correctly on it.
_COPULA_FEATS = COPULA_CONTINUOUS_FEATURES + ["Deductible"]
_N_COPULA     = len(_COPULA_FEATS)   # 19

# Index lookup for copula matrix construction
_COPULA_IDX   = {f: i for i, f in enumerate(_COPULA_FEATS)}

# ── CORRELATION_PAIRS from config already includes Credit_Score ↔ Deductible ──


# ── Helper: Build PSD correlation matrix ──────────────────────────────────────

def _build_psd_corr(feature_names, pairs):
    """
    Assemble a symmetric correlation matrix from (feat_a, feat_b, rho) triples.
    Projects to the nearest positive semi-definite matrix by clipping negative
    eigenvalues to 1e-8, then re-normalises the diagonal to 1.
    """
    n = len(feature_names)
    idx = {f: i for i, f in enumerate(feature_names)}
    C = np.eye(n, dtype=float)
    for fa, fb, rho in pairs:
        if fa in idx and fb in idx:
            i, j = idx[fa], idx[fb]
            C[i, j] = rho
            C[j, i] = rho
    # Nearest PSD via eigenvalue clipping (scipy.linalg.eigh is more stable)
    from scipy.linalg import eigh
    eigvals, eigvecs = eigh(C)
    eigvals = np.clip(eigvals, 1e-8, None)
    C_psd   = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Re-normalise diagonal → correlation matrix
    d       = np.sqrt(np.diag(C_psd))
    C_psd   = C_psd / np.outer(d, d)
    np.fill_diagonal(C_psd, 1.0)
    return C_psd


# ── Helper: Iman-Conover rank permutation ────────────────────────────────────

def _iman_conover(feature_matrix, corr_matrix, rng):
    """
    Iman-Conover rank correlation: rearrange each column of feature_matrix so
    the resulting rank correlations match corr_matrix, while preserving each
    column's original marginal distribution exactly.

    Parameters
    ----------
    feature_matrix : (n, p) ndarray  — columns independently drawn from their marginals
    corr_matrix    : (p, p) ndarray  — target rank correlation matrix (PSD)
    rng            : np.random.Generator

    Returns
    -------
    result : (n, p) ndarray  — same marginals, target rank structure
    """
    n, p = feature_matrix.shape
    # Cholesky factor; fall back to adding jitter if not PD
    try:
        chol = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        jitter = np.eye(p) * 1e-7
        chol   = np.linalg.cholesky(corr_matrix + jitter)

    Z      = rng.standard_normal((n, p)) @ chol.T   # correlated normals
    result = feature_matrix.copy().astype(float)
    for j in range(p):
        z_ranks     = np.argsort(np.argsort(Z[:, j]))   # ordinal ranks 0..n-1
        sorted_vals = np.sort(result[:, j])
        result[:, j] = sorted_vals[z_ranks]
    return result


# ── Helper: State-conditional wildfire inverse CDF ────────────────────────────

def _wildfire_ppf(uniform_q, mode, n_grid=60_000, seed=777):
    """
    Map uniform quantiles to the wildfire distribution for a given state mode.

    Modes
    -----
    "high_bimodal"   : 70 % Beta(1.5, 8)×100  +  30 % Beta(3, 2)×100
    "medium_bimodal" : 80 % Beta(1.5, 8)×100  +  20 % Beta(2.5, 2.5)×100
    "low"            : Beta(0.5, 4)×100
    """
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
    else:   # "low"
        samp  = rng_loc.beta(0.5, 4, n_grid) * 100

    samp = np.clip(samp, 0.0, 100.0)
    grid_q = np.linspace(0.0, 1.0, n_grid)
    return np.interp(uniform_q, grid_q, np.sort(samp))


# ── Helper: Assign risk tier label ────────────────────────────────────────────

def _assign_tier(premium):
    """Map Expected_Pure_Premium to Low / Moderate / Elevated / High tier."""
    t = pd.cut(
        premium,
        bins=[0, 1_000, 2_000, 3_500, np.inf],
        labels=TIER_ORDER,
    )
    return t.astype(str)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def generate_homeowners_data(n_samples: int = N_SAMPLES,
                             random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Generate a synthetic homeowners insurance portfolio.

    Parameters
    ----------
    n_samples    : number of policies (default 100,000)
    random_state : global seed for reproducibility

    Returns
    -------
    pd.DataFrame with 29 feature columns + 3 target columns + metadata columns
    """
    rng = np.random.default_rng(random_state)
    np.random.seed(random_state)   # scikit-learn / scipy compatibility
    n = n_samples
    print(f"\nGenerating {n:,} synthetic homeowners policies…")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — STATE ASSIGNMENT
    # ─────────────────────────────────────────────────────────────────────────
    states   = list(STATE_CONFIG.keys())
    weights  = np.array([STATE_CONFIG[s][0] for s in states])
    weights /= weights.sum()
    State    = rng.choice(states, size=n, p=weights)

    # Vectorised state-level lookups
    wf_mode_arr  = np.array([STATE_CONFIG[s][1] for s in State])    # wildfire mode string
    hail_lam_arr = np.array([STATE_CONFIG[s][2] for s in State],
                             dtype=float)                            # Poisson λ
    flood_sc_arr = np.array([STATE_CONFIG[s][3] for s in State],
                             dtype=float)                            # Exp scale
    terr_p_arr   = np.array([STATE_CONFIG[s][4] for s in State])    # (n, 3) probs
    aoi_mult_arr = np.array([STATE_CONFIG[s][5] for s in State],
                             dtype=float)                            # $/sqft multiplier

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — INDEPENDENT MARGINAL GENERATION
    # (Iman-Conover will impose the target rank correlations in Step 3.)
    # ─────────────────────────────────────────────────────────────────────────

    # ── Legacy 12 main-effect features ──────────────────────────────────────
    Year_Built         = rng.integers(1950, 2024, n).astype(float)
    Square_Footage     = rng.normal(2200, 600, n).clip(800, 5000)
    CLUE_Loss_Count    = rng.poisson(0.30, n).clip(0, 8).astype(float)
    Credit_Score       = rng.normal(700, 80, n).clip(300, 850)
    Protection_Class   = rng.integers(1, 11, n).astype(float)
    Roof_Age_Applicant = rng.integers(1, 31, n).astype(float)

    # AOI: state-adjusted $/sqft  (copula enforces ρ=0.85 with Square_Footage)
    price_per_sqft = rng.uniform(150, 260, n) * aoi_mult_arr
    AOI            = (Square_Footage * price_per_sqft).clip(80_000, 2_000_000)

    # RCV_Appraised: independently drawn but copula will enforce ρ=0.88 with Sqft
    rcv_per_sqft  = rng.uniform(140, 220, n)
    RCV_Appraised = (Square_Footage * rcv_per_sqft).clip(60_000, 1_500_000)

    # Building Code: global marginal; copula enforces ρ=0.70 with Year_Built
    Building_Code_Compliance = rng.uniform(30.0, 100.0, n)

    # ── Modern enrichment features ───────────────────────────────────────────
    # Wildfire: global reference distribution (will be state-mapped in Step 4)
    Wildfire_Exposure_Daily = rng.beta(0.5, 2.0, n) * 100

    # Fire Hydrant Distance: log-normal (copula enforces ρ=0.50 with PC)
    Fire_Hydrant_Distance = rng.lognormal(-0.5, 0.8, n).clip(0.05, 10.0)

    # Tree Canopy Density: Beta
    Tree_Canopy_Density = rng.beta(2, 5, n) * 100

    # Crime Severity: Normal
    Crime_Severity_Index = rng.normal(50, 20, n).clip(0, 100)

    # Pluvial Flood: global reference (will be state-mapped in Step 4)
    Pluvial_Flood_Depth = rng.exponential(5, n).clip(0, 36)

    # Slope Steepness: exponential (copula enforces ρ=0.32 with Wildfire)
    Slope_Steepness = rng.exponential(10, n).clip(0, 45)

    # Hail Frequency: global reference Poisson (will be state-mapped in Step 4)
    Hail_Frequency = rng.poisson(1.5, n).astype(float)

    # Roof Vulnerability: pre-seeded with Roof_Age to preserve natural correlation;
    # copula will not enforce additional correlation (no pair specified in config)
    Roof_Vulnerability_Satellite = (
        Roof_Age_Applicant + rng.normal(1.0, 4.0, n)
    ).clip(0.0, 38.0)

    # Water Loss Recency: global uniform; will be CLUE-conditioned in Step 5
    Water_Loss_Recency_Months = rng.uniform(1.0, 120.0, n)

    # Deductible: ordinal categorical — copula enforces ρ=0.25 with Credit_Score
    Deductible_raw = rng.choice(
        [500.0, 1000.0, 2000.0, 5000.0], n,
        p=[0.15, 0.50, 0.25, 0.10],
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — IMAN-CONOVER RANK CORRELATION (Gaussian Copula Surrogate)
    # ─────────────────────────────────────────────────────────────────────────
    # Assemble 19-column matrix (18 config features + Deductible)
    feat_raw = np.column_stack([
        Year_Built, Square_Footage, CLUE_Loss_Count, Credit_Score,
        Protection_Class, AOI, Roof_Age_Applicant, Roof_Vulnerability_Satellite,
        Wildfire_Exposure_Daily, Water_Loss_Recency_Months, RCV_Appraised,
        Fire_Hydrant_Distance, Tree_Canopy_Density, Crime_Severity_Index,
        Pluvial_Flood_Depth, Building_Code_Compliance, Slope_Steepness,
        Hail_Frequency, Deductible_raw,
    ])  # shape (n, 19)

    # Build 19×19 PSD correlation matrix
    # CORRELATION_PAIRS already contains the Credit_Score ↔ Deductible pair
    corr_mat = _build_psd_corr(_COPULA_FEATS, CORRELATION_PAIRS)

    print("  Applying Iman-Conover rank correlation (copula)…")
    feat_corr = _iman_conover(feat_raw, corr_mat, rng)

    # Unpack correlated columns
    (Year_Built, Square_Footage, CLUE_Loss_Count, Credit_Score,
     Protection_Class, AOI, Roof_Age_Applicant, Roof_Vulnerability_Satellite,
     Wildfire_Exposure_Daily, Water_Loss_Recency_Months, RCV_Appraised,
     Fire_Hydrant_Distance, Tree_Canopy_Density, Crime_Severity_Index,
     Pluvial_Flood_Depth, Building_Code_Compliance, Slope_Steepness,
     Hail_Frequency, Deductible_raw) = feat_corr.T

    # Restore valid ranges after rank permutation (permutation preserves rank but
    # can produce trivially out-of-bound values at the edges when multiple features
    # share boundary clips)
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

    # Snap Deductible to the four valid levels (nearest-level mapping)
    _ded_levels = np.array([500.0, 1000.0, 2000.0, 5000.0])
    Deductible  = _ded_levels[
        np.argmin(np.abs(Deductible_raw[:, None] - _ded_levels[None, :]), axis=1)
    ].astype(int)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 — STATE-CONDITIONAL QUANTILE MAPPING
    # Preserve copula rank structure while imposing state-specific marginals.
    # Method: compute the uniform quantile of each correlated value in the
    # global distribution, then map that quantile through the state's CDF inverse.
    # ─────────────────────────────────────────────────────────────────────────
    from scipy.stats import rankdata as _rankdata

    # ── Wildfire ──────────────────────────────────────────────────────────────
    wf_unif = _rankdata(Wildfire_Exposure_Daily) / (n + 1)
    wf_new  = np.zeros(n)
    for s, cfg in STATE_CONFIG.items():
        mask = (State == s)
        if not mask.any():
            continue
        wf_new[mask] = _wildfire_ppf(wf_unif[mask], cfg[1])  # cfg[1] = wildfire_mode
    Wildfire_Exposure_Daily = np.clip(wf_new, 0, 100).round(2)

    # ── Hail Frequency  (Poisson inverse CDF) ─────────────────────────────────
    hail_unif = _rankdata(Hail_Frequency) / (n + 1)
    hail_unif = np.clip(hail_unif, 0.001, 0.999)
    hail_new  = np.zeros(n)
    for s, cfg in STATE_CONFIG.items():
        mask = (State == s)
        if not mask.any():
            continue
        hail_new[mask] = stats.poisson.ppf(hail_unif[mask], mu=cfg[2])
    Hail_Frequency = np.clip(hail_new.round(0), 0, 15).astype(int)

    # ── Pluvial Flood Depth  (Exponential inverse CDF) ────────────────────────
    flood_unif = _rankdata(Pluvial_Flood_Depth) / (n + 1)
    flood_unif = np.clip(flood_unif, 0.001, 0.999)
    flood_new  = np.zeros(n)
    for s, cfg in STATE_CONFIG.items():
        mask = (State == s)
        if not mask.any():
            continue
        flood_new[mask] = stats.expon.ppf(flood_unif[mask], scale=cfg[3])
    Pluvial_Flood_Depth = np.clip(flood_new, 0, 36).round(2)

    # ── Territory — drawn from state-specific probabilities (categorical) ─────
    territory_cats = ["Urban", "Suburban", "Rural"]
    Territory = np.empty(n, dtype=object)
    for s, cfg in STATE_CONFIG.items():
        mask = (State == s)
        if not mask.any():
            continue
        Territory[mask] = rng.choice(territory_cats, size=mask.sum(), p=cfg[4])

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5 — CATEGORICAL FEATURES (independent of copula)
    # ─────────────────────────────────────────────────────────────────────────
    Construction_Type      = rng.choice(
        ["Frame", "Masonry", "Fire Resistive"], n, p=[0.70, 0.20, 0.10])
    Attic_Ventilation      = rng.choice(
        ["Poor", "Adequate", "Excellent"], n, p=[0.30, 0.50, 0.20])
    Soil_Liquefaction_Risk = rng.choice(
        ["Low", "Moderate", "High"], n, p=[0.70, 0.20, 0.10])
    Fire_Alarm             = rng.binomial(1, 0.40, n).astype(bool)
    Burglar_Alarm          = rng.binomial(1, 0.30, n).astype(bool)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6 — WATER LOSS RECENCY (CLUE-conditional post-processing)
    # Re-map the copula-correlated uniform quantile to:
    #   • 1–36  months for policies with prior claims (CLUE > 0)
    #   • 120   months for policies with no prior claims (no signal)
    # ─────────────────────────────────────────────────────────────────────────
    has_prior_claim = (CLUE_Loss_Count > 0)
    wlr_unif        = _rankdata(Water_Loss_Recency_Months) / (n + 1)
    Water_Loss_Recency_Months = np.where(
        has_prior_claim,
        np.clip((wlr_unif * 35 + 1).round(0), 1, 36),
        120,
    ).astype(int)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7 — CREDIT SCORE SUPPRESSION (CA, MA — regulatory simulation)
    # ─────────────────────────────────────────────────────────────────────────
    Credit_Score_Suppressed = np.isin(State, sorted(CREDIT_SUPPRESSED_STATES))
    Credit_Score            = Credit_Score.copy()
    Credit_Score[Credit_Score_Suppressed] = 700.0   # Portfolio median

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 8 — DERIVED FEATURES
    # ─────────────────────────────────────────────────────────────────────────
    Dwelling_Age      = (2026 - Year_Built).astype(int)
    RCV_Overstatement = np.maximum(0.0, AOI - RCV_Appraised)   # moral-hazard gap

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 9 — LEGACY LOG-SIGNAL  (linear effects → GLM territory)
    # Includes all 4 GLM-engineered interaction effects from config.GLM_INTERACTIONS
    # ─────────────────────────────────────────────────────────────────────────
    is_frame = (Construction_Type == "Frame").astype(float)
    is_urban = (Territory == "Urban").astype(float)

    # Territory factor
    terr_factor = np.where(Territory == "Urban",    0.12,
                  np.where(Territory == "Rural",   -0.08, 0.0))

    # Deductible factor
    ded_factor  = np.where(Deductible == 500,   0.08,
                  np.where(Deductible == 2000, -0.05,
                  np.where(Deductible == 5000, -0.10, 0.0)))

    legacy_raw = (
        # ── 12 main effects (linear — GLM fully recovers these) ──────────────
          (Protection_Class - 5) * 0.06          # PC surcharge (ISO relativities)
        + CLUE_Loss_Count * 0.20                  # prior claims surcharge
        + (700 - Credit_Score) / 1000 * 1.50      # credit inverse risk
        + Dwelling_Age * 0.008                    # building age (0.08 per decade)
        + is_frame * 0.18                         # frame construction surcharge
        + terr_factor                             # territory (Urban/Rural)
        + ded_factor                              # deductible factor
        - Fire_Alarm.astype(float) * 0.08        # monitored alarm credit
        - Burglar_Alarm.astype(float) * 0.06     # burglar alarm credit
        + Roof_Age_Applicant * 0.005              # continuous roof age
        # ── 4 GLM-engineered interactions (ISO-grounded) ─────────────────────
        + (is_frame * (Protection_Class > 6)) * 0.12             # I1: Frame × High PC
        + ((CLUE_Loss_Count >= 2) * (Deductible <= 500)) * 0.10  # I2: Claims × Low Ded
        + (is_urban * (Protection_Class > 6)) * 0.08             # I3: Urban × High PC
        + ((Roof_Age_Applicant > 20) * (Hail_Frequency >= 3)) * 0.10  # I4: Roof × Hail
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 10 — MODERN LOG-SIGNAL  (non-linear + interaction effects → GA2M)
    # Shapes match Supplement S-A.3 table exactly.
    # NOTE: Fire Hydrant Distance is +log1p (farther = higher risk surcharge).
    #       The supplement's "−log1p × 0.10" appears to be a sign error relative
    #       to the actuarial interpretation (farther from hydrant = more risk).
    #       Using + to match ISO PPC mechanics.
    # ─────────────────────────────────────────────────────────────────────────
    modern_raw = (
        # ── Non-linear main effects ──────────────────────────────────────────
          # CONVEX wildfire: flat <30, accelerating >30 (S-A.3: 0.0003 × max(0, score−30)²)
          0.0003 * np.maximum(0, Wildfire_Exposure_Daily - 30) ** 2
          # CONVEX roof decay: quadratic degradation curve
        + (Roof_Vulnerability_Satellite / 20) ** 2 * 0.18
          # LOG diminishing surcharge: first ½ mile critical, flattens beyond 3 mi
        + np.log1p(Fire_Hydrant_Distance) * 0.10
          # THRESHOLD: pre-2000 code compliance cliff below 60 %
        + (Building_Code_Compliance < 60).astype(float) * 0.15
          # EXPONENTIAL DECAY: water loss recency (recent claims = high repeat risk)
        + np.exp(-Water_Loss_Recency_Months / 12.0) * 0.18
        # ── 6 pairwise interactions (GA2M must-include pairs) ─────────────────
          # Wildfire × Roof Vulnerability (ember ignition — super-multiplicative)
        + (Wildfire_Exposure_Daily / 100) * (Roof_Vulnerability_Satellite / 20) * 0.28
          # Water Recency × Tree Canopy (moisture retention cycle)
        + np.exp(-Water_Loss_Recency_Months / 12.0) * (Tree_Canopy_Density / 100) * 0.22
          # RCV Overstatement × Crime (moral hazard signal)
        + (RCV_Overstatement / 100_000) * (Crime_Severity_Index / 100) * 0.18
          # Pluvial Flood × Dwelling Age (pre-code foundations)
        + (Pluvial_Flood_Depth > 15).astype(float) * (Dwelling_Age > 35).astype(float) * 0.18
          # Slope × Wildfire (debris-flow / ember transport)
        + (Slope_Steepness / 45) * (Wildfire_Exposure_Daily / 100) * 0.14
          # Hail Frequency × Roof Vulnerability (accumulated micro-damage)
        + (Hail_Frequency > 3).astype(float) * (Roof_Vulnerability_Satellite > 18).astype(float) * 0.18
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 11 — STANDARDISE AND SCALE  (R² budget: 0.38 / 0.14 / 0.21)
    # After standardisation each component has unit std.
    # Scaling to σ = DGP_LEGACY_SCALAR (0.38) or DGP_MODERN_SCALAR (0.14)
    # ensures the theoretical R² budget is met in log(EPP) space.
    # ─────────────────────────────────────────────────────────────────────────
    legacy_scaled = (
        (legacy_raw - legacy_raw.mean()) / legacy_raw.std()
    ) * DGP_LEGACY_SCALAR

    modern_scaled = (
        (modern_raw - modern_raw.mean()) / modern_raw.std()
    ) * DGP_MODERN_SCALAR

    noise = rng.normal(0.0, DGP_NOISE_SIGMA, n)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 12 — EXPECTED PURE PREMIUM  (oracle target) — AOI-anchored
    #
    # Design: log(EPP_i) = base_log_i + legacy_scaled_i + modern_scaled_i + noise_i
    #
    # base_log_i = log(rate_per_1k × state_aoi_mult) + log(AOI_i / 1000)
    #   → rate_per_1k ≈ $3.75 per $1,000 of coverage (HO industry rate)
    #   → state aoi_mult already baked into AOI via price_per_sqft
    #   → Mean AOI ≈ $440K  →  exposure ≈ 440  →  base EPP ≈ $1,650 mean
    #
    # This structure mirrors ISO HO rate filing mechanics and ensures:
    #   • Mean EPP ≈ $1,500–$2,200 (state-weighted, AOI-anchored)
    #   • R² budget preserved: Var(log EPP) = 0.38² + 0.14² + 0.21² + Var(base_log)
    #   • GLM captures legacy_scaled + base_log; EBM captures modern_scaled
    #
    # Lambda and severity are then DERIVED from EPP:
    #   λ_i = BASE_CLAIM_RATE × EPP_i / mean(EPP)    → mean(λ) = 5.5% exactly
    #   μ_i = EPP_i / λ_i                              → E[Claim_Count × sev] = EPP_i
    # ─────────────────────────────────────────────────────────────────────────
    RATE_PER_1K   = 3.75   # $3.75 per $1,000 AOI — calibrated for mean EPP ≈ $1,700

    base_log      = np.log(RATE_PER_1K) + np.log(AOI / 1_000.0)
    log_epp       = base_log + legacy_scaled + modern_scaled + noise

    epp_raw       = np.exp(log_epp)
    epp_raw       = np.clip(epp_raw, PREMIUM_FLOOR, None)

    # Calibrate λ so mean(λ) = BASE_CLAIM_RATE exactly (5.5%)
    lambda_freq   = (epp_raw / epp_raw.mean()) * BASE_CLAIM_RATE
    lambda_freq   = np.clip(lambda_freq, 0.001, 0.40)

    # Severity mean derived from EPP and λ (E[claims] = λ × μ = EPP by construction)
    severity_mean = epp_raw / lambda_freq   # ≈ mean(EPP) / 0.055 ≈ $30K–$36K

    Expected_Pure_Premium = epp_raw   # already floored

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 13 — SIMULATE CLAIMS
    # Claim_Count  ~ Poisson(λ_i)
    # Claim_Amount ~ Claim_Count × Gamma(shape=2.5, scale=μ_i/2.5) per policy
    # ─────────────────────────────────────────────────────────────────────────
    Claim_Count  = rng.poisson(lambda_freq)
    Claim_Amount = np.zeros(n)
    has_claim    = Claim_Count > 0
    n_claims     = has_claim.sum()

    if n_claims > 0:
        # Per-claim severity draw; multiply by Claim_Count to get total loss
        raw_sev = rng.gamma(
            shape=2.5,
            scale=severity_mean[has_claim] / 2.5,
            size=n_claims,
        )
        Claim_Amount[has_claim] = raw_sev * Claim_Count[has_claim]

        # Cap aggregate claim amount at 99.5th percentile (reduce extreme outliers)
        sev_cap = np.percentile(Claim_Amount[has_claim], 99.5)
        Claim_Amount = np.minimum(Claim_Amount, sev_cap)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 14 — RISK TIER LABELS
    # ─────────────────────────────────────────────────────────────────────────
    Risk_Tier = _assign_tier(Expected_Pure_Premium)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 15 — ASSEMBLE DATAFRAME
    # ─────────────────────────────────────────────────────────────────────────
    data = pd.DataFrame({
        # ── Geographic identifier ──────────────────────────────────────────
        "State":                          State,
        # ── Legacy 12 main-effect features ────────────────────────────────
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
        # ── Modern 13 enrichment features ─────────────────────────────────
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
        # ── Derived features ────────────────────────────────────────────────
        "Dwelling_Age":                   Dwelling_Age,
        "RCV_Overstatement":              RCV_Overstatement.round(0),
        # ── Metadata / flags ────────────────────────────────────────────────
        "Credit_Score_Suppressed":        Credit_Score_Suppressed,
        "Risk_Tier":                      Risk_Tier,
        # ── Oracle targets ──────────────────────────────────────────────────
        "Expected_Pure_Premium":          Expected_Pure_Premium.round(2),
        "Claim_Count":                    Claim_Count,
        "Claim_Amount":                   Claim_Amount.round(2),
    })

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 16 — COMPREHENSIVE VALIDATION PRINTOUT
    # ─────────────────────────────────────────────────────────────────────────
    _print_validation(data, corr_mat, legacy_raw, modern_raw, noise)

    return data


# ── Validation helper ─────────────────────────────────────────────────────────

def _print_validation(df, corr_mat, legacy_raw, modern_raw, noise):
    """Print comprehensive data-quality checks after generation."""
    n = len(df)
    epp    = df["Expected_Pure_Premium"]
    cc     = df["Claim_Count"]
    ca     = df["Claim_Amount"]
    has_cl = cc > 0

    print("\n" + "=" * 62)
    print("  DATA GENERATION — VALIDATION SUMMARY")
    print("=" * 62)

    # ── Premium distribution ─────────────────────────────────────────────────
    print("\n  PREMIUM DISTRIBUTION")
    print(f"    Policies generated   : {n:,}")
    print(f"    Mean  EPP            : ${epp.mean():>10,.0f}  (target $1,500–$2,200)")
    print(f"    Median EPP           : ${epp.median():>10,.0f}")
    print(f"    p5  / p95            : ${np.percentile(epp, 5):>8,.0f} / ${np.percentile(epp, 95):>8,.0f}")
    print(f"    Min / Max            : ${epp.min():>8,.0f} / ${epp.max():>8,.0f}")

    # ── Claim calibration ────────────────────────────────────────────────────
    print("\n  CLAIM CALIBRATION")
    claim_rate = has_cl.mean()
    sev_arr    = ca[has_cl]
    print(f"    Claim rate           : {claim_rate:.2%}   (target 5.3–5.6%)")
    print(f"    Mean severity        : ${sev_arr.mean():>10,.0f}  (expected ~$28K–$36K given EPP/rate)")
    print(f"    p99.5 severity cap   : ${sev_arr.quantile(0.995) if len(sev_arr) else 0:>10,.0f}")
    print(f"    Total claims         : {has_cl.sum():,}")

    # ── Risk tier distribution ───────────────────────────────────────────────
    print("\n  RISK TIER DISTRIBUTION  (approx target: 30/30/25/15)")
    for tier in ["Low", "Moderate", "Elevated", "High"]:
        pct = (df["Risk_Tier"] == tier).mean() * 100
        print(f"    {tier:<12}: {pct:5.1f}%")

    # ── State distribution ───────────────────────────────────────────────────
    print("\n  STATE DISTRIBUTION")
    state_counts = df["State"].value_counts().sort_values(ascending=False)
    for state, cnt in state_counts.items():
        print(f"    {state:<8}: {cnt:>7,}  ({cnt/n:.1%})")

    # ── Credit score suppression ─────────────────────────────────────────────
    suppressed = df["Credit_Score_Suppressed"].sum()
    print(f"\n  CREDIT SUPPRESSION  (CA + MA): {suppressed:,} policies "
          f"({suppressed/n:.1%}) set to portfolio median 700")

    # ── DGP variance decomposition ───────────────────────────────────────────
    print("\n  DGP VARIANCE DECOMPOSITION")
    leg_std   = legacy_raw.std()
    mod_std   = modern_raw.std()
    noise_std = noise.std()
    print(f"    Legacy signal std (before scaling)  : {leg_std:.4f}")
    print(f"    Modern signal std (before scaling)  : {mod_std:.4f}")
    print(f"    Noise std                           : {noise_std:.4f}")
    print(f"    Scaled legacy std                   : {DGP_LEGACY_SCALAR:.2f}  (target 0.38)")
    print(f"    Scaled modern std                   : {DGP_MODERN_SCALAR:.2f}  (target 0.14)")
    print(f"    Theoretical max GLM R²  (signal budget): "
          f"{DGP_LEGACY_SCALAR**2 / (DGP_LEGACY_SCALAR**2 + DGP_MODERN_SCALAR**2 + DGP_NOISE_SIGMA**2):.3f}")
    print(f"    Theoretical max EBM R²  (signal budget): "
          f"{(DGP_LEGACY_SCALAR**2 + DGP_MODERN_SCALAR**2) / (DGP_LEGACY_SCALAR**2 + DGP_MODERN_SCALAR**2 + DGP_NOISE_SIGMA**2):.3f}")

    # ── Correlation spot-checks ──────────────────────────────────────────────
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
            print(f"    {fa:<32} × {fb:<28}: target={target:+.2f}  "
                  f"actual={actual:+.3f}  {flag}")

    # ── Wildfire distribution by state ───────────────────────────────────────
    print("\n  WILDFIRE DISTRIBUTION (mean by state — high-fire states should be >30)")
    wf_by_state = df.groupby("State")["Wildfire_Exposure_Daily"].mean().sort_values(ascending=False)
    for state, val in wf_by_state.items():
        print(f"    {state:<8}: {val:5.1f}")

    print("=" * 62)


# ── Script entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = generate_homeowners_data(n_samples=N_SAMPLES, random_state=RANDOM_STATE)

    out_dir  = "data"
    out_path = os.path.join(out_dir, "synthetic_homeowners_data.csv")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df):,} policies to '{out_path}'")
    print("Run 'python baseline_glm.py' next to train the GLM.\n")
