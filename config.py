# ==============================================================================
# config.py
# Central configuration for ResiScore™ GLM + GA2M Residual Demo
#
# All path constants, model hyperparameters, DGP scalars, demo parameters,
# geographic state config, Gaussian copula correlation pairs, and EBM
# must-include interaction specs live here.
#
# All other scripts import from this file — no magic numbers elsewhere.
# ==============================================================================

import numpy as np

# ── Directory & File Paths ────────────────────────────────────────────────────
DATA_DIR   = "data"
MODEL_DIR  = "models"

RAW_DATA_PATH      = f"{DATA_DIR}/synthetic_homeowners_data.csv"
BASELINE_DATA_PATH = f"{DATA_DIR}/synthetic_homeowners_data_with_baseline.csv"
FINAL_DATA_PATH    = f"{DATA_DIR}/final_predictions.csv"

# GLM model artifacts
FREQ_MODEL_PATH    = f"{MODEL_DIR}/freq_glm.pkl"
SEV_MODEL_PATH     = f"{MODEL_DIR}/sev_glm.pkl"
PREPROCESSOR_PATH  = f"{MODEL_DIR}/glm_preprocessor.pkl"

# Legacy sklearn model paths (kept for backward compatibility with app.py if needed)
LEGACY_FREQ_PATH   = f"{MODEL_DIR}/legacy_freq_model.pkl"
LEGACY_SEV_PATH    = f"{MODEL_DIR}/legacy_sev_model.pkl"

# EBM residual model
EBM_MODEL_PATH     = f"{MODEL_DIR}/ebm_residual_model.pkl"

# Model metadata (performance metrics, feature lists, training date)
METADATA_PATH      = f"{MODEL_DIR}/model_metadata.json"

# ── Data Generation Parameters ────────────────────────────────────────────────
N_SAMPLES    = 100_000    # 100K for reliable interaction detection (Gelman 16× rule)
RANDOM_STATE = 42

# ── Premium / Risk Parameters ─────────────────────────────────────────────────
PREMIUM_FLOOR     = 300      # Hard floor ($/policy-year) — matches DGP construction
MIN_UPLIFT        = 0.65     # Max downward adjustment: −35%
MAX_UPLIFT        = 1.60     # Max upward adjustment:  +60%
UNDERPRICE_THRESH = 0.20     # Policies > 20% underpriced flagged as "hidden dangers"
SEVERITY_CAP_PCT  = 0.995    # Cap claim severity at 99.5th percentile

# ── Train / Test Split ────────────────────────────────────────────────────────
TEST_SIZE = 0.20   # 80 / 20 split

# ── DGP Variance Scalars ──────────────────────────────────────────────────────
# Controls how much variance each component contributes to log(premium):
#   Signal variance = 0.32² + 0.20² = 0.1024 + 0.04 = 0.1424
#   Noise variance  = 0.25² = 0.0625
#   Total (excl base_log) = 0.2049
#   Base_log variance (AOI-driven) ≈ 0.091
#   Full total ≈ 0.296
#   Theoretical GLM R² ≈ (0.091 + 0.1024) / 0.296 ≈ 0.65
#   Theoretical ΔR² ≈ 0.04 / 0.296 ≈ 0.135
#   Theoretical combined R² ≈ 0.233 / 0.296 ≈ 0.79
# → Theoretical max R² ≈ 0.65, leaving substantial noise floor
DGP_LEGACY_SCALAR = 0.32    # Variance scalar for legacy (linear) signal component
DGP_MODERN_SCALAR = 0.20    # Variance scalar for modern (non-linear) signal component
DGP_NOISE_SIGMA   = 0.25    # Std of irreducible noise in log-premium space

# ── Base Rates (DGP claim calibration) ───────────────────────────────────────
BASE_CLAIM_RATE   = 0.055    # Target: ~5.5% of policies have ≥1 claim (III/ISO 2023)
BASE_SEVERITY     = 15_000   # Target average severity per claim: ~$15K (blended HO)
BASE_LOG_FREQ     = np.log(BASE_CLAIM_RATE)   # log(0.055) ≈ −2.90

# ── GLM Hyperparameters ───────────────────────────────────────────────────────
FREQ_ALPHA = 1e-4    # L2 regularisation (used only if falling back to sklearn)
SEV_ALPHA  = 1e-4

# ── EBM / GA2M Hyperparameters ────────────────────────────────────────────────
EBM_INTERACTIONS  = 15     # Total interactions (6 forced + 9 auto)
EBM_MAX_BINS      = 256
EBM_MAX_INT_BINS  = 32
EBM_LEARNING_RATE = 0.02   # Lower than default → smoother, more stable shape functions
EBM_OUTER_BAGS    = 8      # Variance reduction via bagging
EBM_INNER_BAGS    = 0

# ── App Display Parameters ────────────────────────────────────────────────────
APP_PORT          = 8050
APP_DEBUG         = False
N_DEMO_POLICIES   = 200    # Policies available in the Policy Lens dropdown
RECLASS_SAMPLE    = 15_000 # Sample size for reclassification scatter chart
CHART_HEIGHT_SM   = 300
CHART_HEIGHT_MD   = 400
CHART_HEIGHT_LG   = 500

# ── Risk Tier Thresholds (applied to Final_Pure_Premium) ─────────────────────
# Approximate boundaries for Low / Moderate / Elevated / High tiers
# These are calibrated so ~30/30/25/15% of the portfolio falls in each tier
TIER_BOUNDARIES = {
    "Low":      (0,     1_200),
    "Moderate": (1_200, 2_200),
    "Elevated": (2_200, 4_000),
    "High":     (4_000, float("inf")),
}
TIER_ORDER = ["Low", "Moderate", "Elevated", "High"]

# ── GLM Feature Lists ─────────────────────────────────────────────────────────
# 12 main effects + 4 engineered actuary interactions = 16 total
GLM_MAIN_EFFECTS = [
    "Dwelling_Age",        # derived: 2026 - Year_Built (more interpretable than raw year)
    "Square_Footage",
    "CLUE_Loss_Count",
    "Credit_Score",
    "Construction_Type",
    "Protection_Class",
    "AOI",
    "Deductible",
    "Territory",
    "Roof_Age_Applicant",
    "Fire_Alarm",
    "Burglar_Alarm",
]

GLM_INTERACTIONS = [
    "Urban_HighPC",        # (Territory=="Urban") & (PC > 6)
    "OldRoof_HighHail",    # (Roof_Age > 20) & (Hail_Freq >= 3)
    "Frame_HighPC",        # (Construction=="Frame") & (PC > 6)   [NEW — S12]
    "FreqClaims_LowDed",   # (CLUE_Count >= 2) & (Deductible <= 500)  [NEW — S12]
]

GLM_ALL_FEATURES = GLM_MAIN_EFFECTS + GLM_INTERACTIONS

GLM_CAT_COLS = [
    "Construction_Type", "Territory", "Deductible",
    "Fire_Alarm", "Burglar_Alarm",
    # Interaction binary columns treated as categorical for OneHotEncoder
    "Urban_HighPC", "OldRoof_HighHail", "Frame_HighPC", "FreqClaims_LowDed",
]
GLM_NUM_COLS = [c for c in GLM_ALL_FEATURES if c not in GLM_CAT_COLS]

# ── EBM / GA2M Feature Lists ──────────────────────────────────────────────────
# 25 base features + 3 derived = 28 total input features for EBM
EBM_BASE_FEATURES = [
    # Legacy features (EBM can capture their non-linear effects)
    "Year_Built", "Square_Footage", "CLUE_Loss_Count", "Credit_Score",
    "Construction_Type", "Protection_Class", "AOI", "Deductible",
    "Territory", "Roof_Age_Applicant", "Fire_Alarm", "Burglar_Alarm",
    # Modern features (new signals GLM never saw)
    "Roof_Vulnerability_Satellite", "Wildfire_Exposure_Daily",
    "Water_Loss_Recency_Months", "RCV_Appraised", "Fire_Hydrant_Distance",
    "Tree_Canopy_Density", "Crime_Severity_Index", "Pluvial_Flood_Depth",
    "Building_Code_Compliance", "Slope_Steepness", "Attic_Ventilation",
    "Hail_Frequency", "Soil_Liquefaction_Risk",
]

EBM_DERIVED_FEATURES = [
    "Dwelling_Age",        # 2026 - Year_Built — explicit non-linear age signal
    "RCV_Overstatement",   # max(0, AOI - RCV_Appraised) — moral hazard signal
    "Log_AOI",             # log(AOI) — stabilises right tail for interaction terms
]

EBM_ALL_FEATURES = EBM_BASE_FEATURES + EBM_DERIVED_FEATURES  # 28 features

EBM_CAT_COLS = [
    "Construction_Type", "Territory", "Deductible",
    "Fire_Alarm", "Burglar_Alarm", "Attic_Ventilation", "Soil_Liquefaction_Risk",
]

# ── EBM Must-Include Interaction Pairs ────────────────────────────────────────
# These 6 pairwise interactions must be present in the trained EBM.
# They correspond to the non-linear interaction effects embedded in the DGP.
# Specified as (feature_name_A, feature_name_B) — converted to index tuples at training time.
MUST_INCLUDE_INTERACTIONS = [
    ("Wildfire_Exposure_Daily",    "Roof_Vulnerability_Satellite"),  # WUI × roof condition
    ("Water_Loss_Recency_Months",  "Tree_Canopy_Density"),           # moisture retention cycle
    ("RCV_Overstatement",          "Crime_Severity_Index"),          # moral hazard
    ("Pluvial_Flood_Depth",        "Dwelling_Age"),                  # pre-code foundations
    ("Slope_Steepness",            "Wildfire_Exposure_Daily"),       # debris flow
    ("Hail_Frequency",             "Roof_Vulnerability_Satellite"),  # accumulated damage
]

# ── Gaussian Copula Correlation Pairs ─────────────────────────────────────────
# Applied to continuous / ordinal features only.
# Categorical features (Construction_Type, Territory, etc.) are generated
# conditionally on state or independently per STATE_CONFIG.
# All unspecified pairs default to 0.0 (independent).
CORRELATION_PAIRS = [
    # (Feature_A,                  Feature_B,                  target_rho)
    ("Roof_Age_Applicant",         "Year_Built",                -0.80),  # older home = older roof
    ("AOI",                        "Square_Footage",             0.85),  # coverage tracks size
    ("RCV_Appraised",              "Square_Footage",             0.88),  # appraisal tracks size
    ("AOI",                        "RCV_Appraised",              0.82),  # coverage ~ appraisal
    ("Building_Code_Compliance",   "Year_Built",                 0.70),  # newer = better code
    ("Wildfire_Exposure_Daily",    "Tree_Canopy_Density",        0.38),  # WUI has vegetation
    ("Wildfire_Exposure_Daily",    "Slope_Steepness",            0.32),  # fire on slopes
    ("Fire_Hydrant_Distance",      "Protection_Class",           0.50),  # rural = far + high PC
    ("Crime_Severity_Index",       "Protection_Class",           0.25),  # weak positive
    ("Pluvial_Flood_Depth",        "Slope_Steepness",           -0.20),  # flat areas flood more
    ("Hail_Frequency",             "Wildfire_Exposure_Daily",   -0.15),  # hail belt ≠ fire belt
    ("Credit_Score",               "CLUE_Loss_Count",           -0.30),  # behavioural correlation
    ("Credit_Score",               "Deductible",                 0.25),  # higher credit → higher ded
    ("Square_Footage",             "Year_Built",                 0.20),  # newer homes slightly larger
]

# Ordered list of continuous features that participate in the copula
# (must include both endpoints of every pair above, in a consistent order)
COPULA_CONTINUOUS_FEATURES = [
    "Year_Built",
    "Square_Footage",
    "CLUE_Loss_Count",
    "Credit_Score",
    "Protection_Class",
    "AOI",
    "Roof_Age_Applicant",
    "Roof_Vulnerability_Satellite",
    "Wildfire_Exposure_Daily",
    "Water_Loss_Recency_Months",
    "RCV_Appraised",
    "Fire_Hydrant_Distance",
    "Tree_Canopy_Density",
    "Crime_Severity_Index",
    "Pluvial_Flood_Depth",
    "Building_Code_Compliance",
    "Slope_Steepness",
    "Hail_Frequency",
]

# ── State Configuration ───────────────────────────────────────────────────────
# Each state entry:
#   weight            — portfolio share (sums to 1.0)
#   wildfire_mode     — "high_bimodal" | "medium_bimodal" | "low"
#   hail_lambda       — Poisson λ for hail frequency
#   flood_scale       — exponential scale for pluvial flood depth
#   territory_probs   — [P(Urban), P(Suburban), P(Rural)]
#   aoi_multiplier    — multiplies base $/sqft for regional construction cost
#
# Wildfire modes:
#   "high_bimodal"   : 70% Beta(1.5, 8)×100 + 30% Beta(3, 2)×100
#   "medium_bimodal" : 80% Beta(1.5, 8)×100 + 20% Beta(2.5, 2.5)×100
#   "low"            : Beta(0.5, 4)×100
#
# Credit score suppression: CA and MA simulate regulatory suppression —
#   Credit_Score is set to portfolio median (700) for these states, and
#   Credit_Score_Suppressed = True is set as a boolean flag column.

STATE_CONFIG = {
    #          weight  wf_mode          hail_λ  flood_scale  territory_probs     aoi_mult
    "CA":     (0.15,  "high_bimodal",   0.8,    4.0,        [0.35, 0.45, 0.20], 1.40),
    "TX":     (0.14,  "low",            2.8,    5.0,        [0.30, 0.45, 0.25], 0.85),
    "FL":     (0.12,  "low",            0.5,    9.0,        [0.35, 0.50, 0.15], 1.10),
    "NY":     (0.10,  "low",            0.7,    3.0,        [0.55, 0.35, 0.10], 1.30),
    "CO":     (0.08,  "medium_bimodal", 2.5,    3.5,        [0.20, 0.50, 0.30], 1.00),
    "OK":     (0.07,  "low",            3.2,    4.5,        [0.15, 0.45, 0.40], 0.75),
    "LA":     (0.07,  "low",            0.6,    8.5,        [0.30, 0.45, 0.25], 0.80),
    "MA":     (0.07,  "low",            0.9,    3.5,        [0.45, 0.40, 0.15], 1.25),
    "WA":     (0.06,  "medium_bimodal", 0.6,    4.0,        [0.35, 0.45, 0.20], 1.15),
    "GA":     (0.06,  "low",            1.5,    5.5,        [0.30, 0.50, 0.20], 0.90),
    "Other":  (0.08,  "low",            1.2,    4.0,        [0.30, 0.50, 0.20], 1.00),
}

# States that suppress credit score as a rating factor (regulatory)
CREDIT_SUPPRESSED_STATES = {"CA", "MA"}

# ── Colour Palette (used by app.py) ──────────────────────────────────────────
NAVY   = "#0D1B2A"
GOLD   = "#C9A84C"
TEAL   = "#2EC4B6"
RED    = "#E63946"
GREEN  = "#2DC653"
AMBER  = "#F4A261"
GREY   = "#8D9EAD"
WHITE  = "#FFFFFF"
BG     = "#F0F2F5"
CARD   = "#FFFFFF"

# Tier colours
TIER_COLORS = {
    "Low":      GREEN,
    "Moderate": TEAL,
    "Elevated": AMBER,
    "High":     RED,
}
