# ==============================================================================
# config.py
# Central configuration for the Homeowners GLM + GA2M Residual Demo
# ==============================================================================

import numpy as np

# ── Directory & File Paths ────────────────────────────────────────────────────
DATA_DIR   = "data"
MODEL_DIR  = "models"

RAW_DATA_PATH      = f"{DATA_DIR}/synthetic_homeowners_data.csv"
BASELINE_DATA_PATH = f"{DATA_DIR}/synthetic_homeowners_data_with_baseline.csv"
FINAL_DATA_PATH    = f"{DATA_DIR}/final_predictions.csv"

FREQ_MODEL_PATH    = f"{MODEL_DIR}/freq_glm.pkl"
SEV_MODEL_PATH     = f"{MODEL_DIR}/sev_glm.pkl"
PREPROCESSOR_PATH  = f"{MODEL_DIR}/glm_preprocessor.pkl"

LEGACY_FREQ_PATH   = f"{MODEL_DIR}/legacy_freq_model.pkl"
LEGACY_SEV_PATH    = f"{MODEL_DIR}/legacy_sev_model.pkl"

EBM_MODEL_PATH     = f"{MODEL_DIR}/ebm_residual_model.pkl"
METADATA_PATH      = f"{MODEL_DIR}/model_metadata.json"

# ── Data Generation Parameters ────────────────────────────────────────────────
N_SAMPLES    = 100_000
RANDOM_STATE = 42

# ── Premium / Risk Parameters ─────────────────────────────────────────────────
PREMIUM_FLOOR     = 300
MIN_UPLIFT        = 0.65
MAX_UPLIFT        = 1.60
UNDERPRICE_THRESH = 0.20
SEVERITY_CAP_PCT  = 0.995

# ── Train / Test Split ────────────────────────────────────────────────────────
TEST_SIZE = 0.20

# ── DGP Variance Scalars ──────────────────────────────────────────────────────
# Theoretical GLM R² ≈ (0.091 + 0.1024) / 0.296 ≈ 0.65
# Theoretical ΔR²    ≈ 0.04 / 0.296 ≈ 0.135
#
# Modern signal now split into two separate components:
#   DGP_MODERN_MAIN_SCALAR  — non-linear main effects  (GA2M shape functions)
#   DGP_INTERACTION_SCALAR  — pairwise interactions    (GA2M interaction surfaces)
#
# Total modern variance budget: 0.16² + 0.12² = 0.0256 + 0.0144 = 0.04 = 0.20²
# Interactions now get 36% of the modern budget instead of competing within a
# single combined signal where they received ~20%.
DGP_LEGACY_SCALAR      = 0.32    # Legacy (linear) signal — GLM territory
DGP_MODERN_MAIN_SCALAR = 0.16    # Non-linear main effects — was part of combined 0.20
DGP_INTERACTION_SCALAR = 0.12    # Pairwise interactions — dedicated budget (NEW)
DGP_NOISE_SIGMA        = 0.25    # Irreducible noise in log-premium space

# ── Base Rates (DGP claim calibration) ───────────────────────────────────────
BASE_CLAIM_RATE   = 0.055
BASE_SEVERITY     = 15_000
BASE_LOG_FREQ     = np.log(BASE_CLAIM_RATE)

# ── GLM Hyperparameters ───────────────────────────────────────────────────────
FREQ_ALPHA = 1e-4
SEV_ALPHA  = 1e-4

# ── EBM / GA2M Hyperparameters ────────────────────────────────────────────────
EBM_INTERACTIONS  = 15
EBM_MAX_BINS      = 256
EBM_MAX_INT_BINS  = 32
EBM_LEARNING_RATE = 0.02
EBM_OUTER_BAGS    = 8
EBM_INNER_BAGS    = 0

# ── App Display Parameters ────────────────────────────────────────────────────
APP_PORT          = 8050
APP_DEBUG         = False
N_DEMO_POLICIES   = 200
RECLASS_SAMPLE    = 15_000
CHART_HEIGHT_SM   = 300
CHART_HEIGHT_MD   = 400
CHART_HEIGHT_LG   = 500

# ── Risk Tier Thresholds ─────────────────────────────────────────────────────
TIER_BOUNDARIES = {
    "Low":      (0,     1_200),
    "Moderate": (1_200, 2_200),
    "Elevated": (2_200, 4_000),
    "High":     (4_000, float("inf")),
}
TIER_ORDER = ["Low", "Moderate", "Elevated", "High"]

# ── GLM Feature Lists ─────────────────────────────────────────────────────────
GLM_MAIN_EFFECTS = [
    "Dwelling_Age", "Square_Footage", "CLUE_Loss_Count", "Credit_Score",
    "Construction_Type", "Protection_Class", "AOI", "Deductible",
    "Territory", "Roof_Age_Applicant", "Fire_Alarm", "Burglar_Alarm",
]

GLM_INTERACTIONS = [
    "Urban_HighPC", "OldRoof_HighHail", "Frame_HighPC", "FreqClaims_LowDed",
]

GLM_ALL_FEATURES = GLM_MAIN_EFFECTS + GLM_INTERACTIONS

GLM_CAT_COLS = [
    "Construction_Type", "Territory", "Deductible",
    "Fire_Alarm", "Burglar_Alarm",
    "Urban_HighPC", "OldRoof_HighHail", "Frame_HighPC", "FreqClaims_LowDed",
]
GLM_NUM_COLS = [c for c in GLM_ALL_FEATURES if c not in GLM_CAT_COLS]

# ── EBM / GA2M Feature Lists ──────────────────────────────────────────────────
EBM_BASE_FEATURES = [
    "Year_Built", "Square_Footage", "CLUE_Loss_Count", "Credit_Score",
    "Construction_Type", "Protection_Class", "AOI", "Deductible",
    "Territory", "Roof_Age_Applicant", "Fire_Alarm", "Burglar_Alarm",
    "Roof_Vulnerability_Satellite", "Wildfire_Exposure_Daily",
    "Water_Loss_Recency_Months", "RCV_Appraised", "Fire_Hydrant_Distance",
    "Tree_Canopy_Density", "Crime_Severity_Index", "Pluvial_Flood_Depth",
    "Building_Code_Compliance", "Slope_Steepness", "Attic_Ventilation",
    "Hail_Frequency", "Soil_Liquefaction_Risk",
]

EBM_DERIVED_FEATURES = [
    "Dwelling_Age", "RCV_Overstatement", "Log_AOI",
]

EBM_ALL_FEATURES = EBM_BASE_FEATURES + EBM_DERIVED_FEATURES

EBM_CAT_COLS = [
    "Construction_Type", "Territory", "Deductible",
    "Fire_Alarm", "Burglar_Alarm", "Attic_Ventilation", "Soil_Liquefaction_Risk",
]

# ── EBM Must-Include Interaction Pairs ────────────────────────────────────────
MUST_INCLUDE_INTERACTIONS = [
    ("Wildfire_Exposure_Daily",    "Roof_Vulnerability_Satellite"),
    ("Water_Loss_Recency_Months",  "Tree_Canopy_Density"),
    ("RCV_Overstatement",          "Crime_Severity_Index"),
    ("Pluvial_Flood_Depth",        "Dwelling_Age"),
    ("Slope_Steepness",            "Wildfire_Exposure_Daily"),
    ("Hail_Frequency",             "Roof_Vulnerability_Satellite"),
]

# ── Gaussian Copula Correlation Pairs ─────────────────────────────────────────
CORRELATION_PAIRS = [
    ("Roof_Age_Applicant",         "Year_Built",                -0.80),
    ("AOI",                        "Square_Footage",             0.85),
    ("RCV_Appraised",              "Square_Footage",             0.88),
    ("AOI",                        "RCV_Appraised",              0.82),
    ("Building_Code_Compliance",   "Year_Built",                 0.70),
    ("Wildfire_Exposure_Daily",    "Tree_Canopy_Density",        0.38),
    ("Wildfire_Exposure_Daily",    "Slope_Steepness",            0.32),
    ("Fire_Hydrant_Distance",      "Protection_Class",           0.50),
    ("Crime_Severity_Index",       "Protection_Class",           0.25),
    ("Pluvial_Flood_Depth",        "Slope_Steepness",           -0.20),
    ("Hail_Frequency",             "Wildfire_Exposure_Daily",   -0.15),
    ("Credit_Score",               "CLUE_Loss_Count",           -0.30),
    ("Credit_Score",               "Deductible",                 0.25),
    ("Square_Footage",             "Year_Built",                 0.20),
]

COPULA_CONTINUOUS_FEATURES = [
    "Year_Built", "Square_Footage", "CLUE_Loss_Count", "Credit_Score",
    "Protection_Class", "AOI", "Roof_Age_Applicant", "Roof_Vulnerability_Satellite",
    "Wildfire_Exposure_Daily", "Water_Loss_Recency_Months", "RCV_Appraised",
    "Fire_Hydrant_Distance", "Tree_Canopy_Density", "Crime_Severity_Index",
    "Pluvial_Flood_Depth", "Building_Code_Compliance", "Slope_Steepness",
    "Hail_Frequency",
]

# ── State Configuration ───────────────────────────────────────────────────────
STATE_CONFIG = {
    "CA":     (0.15,  "high_bimodal",   0.8,    4.0,  [0.35, 0.45, 0.20], 1.40),
    "TX":     (0.14,  "low",            2.8,    5.0,  [0.30, 0.45, 0.25], 0.85),
    "FL":     (0.12,  "low",            0.5,    9.0,  [0.35, 0.50, 0.15], 1.10),
    "NY":     (0.10,  "low",            0.7,    3.0,  [0.55, 0.35, 0.10], 1.30),
    "CO":     (0.08,  "medium_bimodal", 2.5,    3.5,  [0.20, 0.50, 0.30], 1.00),
    "OK":     (0.07,  "low",            3.2,    4.5,  [0.15, 0.45, 0.40], 0.75),
    "LA":     (0.07,  "low",            0.6,    8.5,  [0.30, 0.45, 0.25], 0.80),
    "MA":     (0.07,  "low",            0.9,    3.5,  [0.45, 0.40, 0.15], 1.25),
    "WA":     (0.06,  "medium_bimodal", 0.6,    4.0,  [0.35, 0.45, 0.20], 1.15),
    "GA":     (0.06,  "low",            1.5,    5.5,  [0.30, 0.50, 0.20], 0.90),
    "Other":  (0.08,  "low",            1.2,    4.0,  [0.30, 0.50, 0.20], 1.00),
}

CREDIT_SUPPRESSED_STATES = {"CA", "MA"}

# ── Colour Palette ────────────────────────────────────────────────────────────
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

TIER_COLORS = {
    "Low":      GREEN,
    "Moderate": TEAL,
    "Elevated": AMBER,
    "High":     RED,
}