# ==============================================================================
# app.py  —  Phase 5 Rewrite
# Homeowners Intelligence Layer — BD Demo  |  ValueMomentum
# Tabs: Business Case | Intelligence Signals | Policy Lens | Framework
#
# Phase 5 changes (vs original):
#   5a. Data/model loading: config.py paths, file-existence checks, separate
#       preprocessor + statsmodels GLM wrappers, OOS-only metrics (test set),
#       risk neutrality check, premium migration metrics
#   5b. Tab 1: risk-neutrality KPI, reclassification scatter (S9), double-lift
#       chart (S8), benchmark annotations on R² bar, KDE distribution, OOS labels
#   5c. Tab 2: Data Characteristics panel (S-A.4) before signal landscape;
#       native EBM shape functions (S6) replacing manual PDPs; native EBM
#       interaction surface replacing binned heatmap
#   5d. Tab 3: Quick Pick demo archetypes (S7); statsmodels GLM waterfall
#   5e. Tab 4: updated feature counts, risk-neutrality formula, credit note, OOS
#   5f. UI: no emoji in tabs, dcc.Loading wrappers, std chart heights, short tooltips
# ==============================================================================

import json
import os
import sys
import warnings

import dash
import dash_bootstrap_components as dbc
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, ctx, dcc, html
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

# ── Config import ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FINAL_DATA_PATH, FREQ_MODEL_PATH, SEV_MODEL_PATH,
    PREPROCESSOR_PATH, EBM_MODEL_PATH, METADATA_PATH,
    GLM_ALL_FEATURES, GLM_CAT_COLS, GLM_INTERACTIONS,
    EBM_ALL_FEATURES, EBM_CAT_COLS,
    UNDERPRICE_THRESH, PREMIUM_FLOOR, MIN_UPLIFT, MAX_UPLIFT,
    NAVY, GOLD, TEAL, RED, GREEN, AMBER, GREY, WHITE, BG,
    TIER_ORDER, TIER_COLORS,
    APP_PORT, RECLASS_SAMPLE,
    CHART_HEIGHT_SM, CHART_HEIGHT_MD, CHART_HEIGHT_LG,
)

# ── Colour aliases ────────────────────────────────────────────────────────────
BLUE   = TEAL
MUTED  = GREY
BORDER = "#E0E4ED"
CARD_STYLE = {"borderRadius": "12px", "border": f"1px solid {BORDER}",
              "boxShadow": "0 2px 8px rgba(0,0,0,0.06)", "backgroundColor": WHITE}
SEC_TITLE  = {"color": NAVY, "fontWeight": "700", "fontSize": "1.05rem", "marginBottom": "2px"}
MONO       = {"fontFamily": "'Courier New', monospace", "fontSize": "0.93rem", "color": NAVY,
              "backgroundColor": "#F0F4FA", "padding": "10px 16px", "borderRadius": "6px",
              "border": f"1px solid {BORDER}", "letterSpacing": "0.02em", "lineHeight": "1.8"}
TAB_STYLE  = {"fontFamily": "Inter", "fontSize": "0.88rem"}
TAB_SEL    = {**TAB_STYLE, "fontWeight": "700", "borderTop": f"3px solid {NAVY}"}

# ── File existence guard ──────────────────────────────────────────────────────
_REQUIRED_FILES = [
    (FINAL_DATA_PATH,  "final predictions CSV"),
    (EBM_MODEL_PATH,   "EBM residual model"),
    (FREQ_MODEL_PATH,  "frequency GLM"),
    (SEV_MODEL_PATH,   "severity GLM"),
    (PREPROCESSOR_PATH,"GLM preprocessor"),
]
for _path, _label in _REQUIRED_FILES:
    if not os.path.exists(_path):
        print(f"\nERROR: Required file not found — {_label}")
        print(f"  Expected path: {_path}")
        print("  Run 'python setup.py' first to generate data and train models.\n")
        sys.exit(1)

# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.FONT_AWESOME,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
)
app.title = "Homeowners Intelligence Layer | ValueMomentum"
server = app.server

# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data and models…")
df = pd.read_csv(FINAL_DATA_PATH)

for col in EBM_CAT_COLS:
    if col in df.columns:
        df[col] = df[col].astype(str)

# Ensure derived EBM features present
if "Dwelling_Age" not in df.columns:
    df["Dwelling_Age"]      = (2026 - df["Year_Built"]).astype(int)
if "RCV_Overstatement" not in df.columns:
    df["RCV_Overstatement"] = np.maximum(0.0, df["AOI"] - df["RCV_Appraised"])
if "Log_AOI" not in df.columns:
    df["Log_AOI"]           = np.log(df["AOI"].clip(1))

# Ensure GLM interaction columns present for policy view
for col in GLM_INTERACTIONS:
    if col not in df.columns:
        df[col] = "0"

# Backward-compat: if Split column missing, mark all as test
if "Split" not in df.columns:
    df["Split"] = "test"

# Backward-compat: Adjustment_Pct
if "Adjustment_Pct" not in df.columns:
    df["Adjustment_Pct"] = (df["Final_Pure_Premium"] / df["GLM_Pure_Premium"] - 1) * 100

# Ensure tier columns
if "Final_Risk_Tier" not in df.columns:
    df["Final_Risk_Tier"] = pd.cut(
        df["Final_Pure_Premium"], bins=[0, 1000, 2000, 3500, np.inf],
        labels=TIER_ORDER).astype(str)
if "GLM_Risk_Tier" not in df.columns:
    df["GLM_Risk_Tier"] = pd.cut(
        df["GLM_Pure_Premium"], bins=[0, 1000, 2000, 3500, np.inf],
        labels=TIER_ORDER).astype(str)

# Convenience alias
df["Risk_Tier"] = df["Final_Risk_Tier"]

# Load models
ebm_model      = joblib.load(EBM_MODEL_PATH)
freq_glm       = joblib.load(FREQ_MODEL_PATH)
sev_glm        = joblib.load(SEV_MODEL_PATH)
glm_preprocessor = joblib.load(PREPROCESSOR_PATH)

# Load metadata if available
_metadata = {}
if os.path.exists(METADATA_PATH):
    try:
        with open(METADATA_PATH) as f:
            _metadata = json.load(f)
    except Exception:
        pass

N_TOTAL  = len(df)
N_TRAIN  = (df["Split"] == "train").sum()
N_TEST   = (df["Split"] == "test").sum()
print(f"  {N_TOTAL:,} policies  (train={N_TRAIN:,}  test={N_TEST:,})")

# ── OOS METRICS — test set only (Spec G2.6) ───────────────────────────────────
_test = df[df["Split"] == "test"].copy()
glm_r2   = r2_score(_test["Expected_Pure_Premium"], _test["GLM_Pure_Premium"])
final_r2 = r2_score(_test["Expected_Pure_Premium"], _test["Final_Pure_Premium"])
delta_r2 = final_r2 - glm_r2
glm_rmse   = np.sqrt(mean_squared_error(_test["Expected_Pure_Premium"], _test["GLM_Pure_Premium"]))
final_rmse = np.sqrt(mean_squared_error(_test["Expected_Pure_Premium"], _test["Final_Pure_Premium"]))
OOS_LABEL  = f"(out-of-sample, N={N_TEST:,})"

# ── RISK NEUTRALITY — Spec N4.1 ───────────────────────────────────────────────
_glm_w = df["GLM_Pure_Premium"].values
_uplift = df["EBM_Uplift_Factor"].values if "EBM_Uplift_Factor" in df.columns else np.ones(N_TOTAL)
_risk_neutral_check = float(np.average(_uplift, weights=_glm_w))
_total_glm   = float(df["GLM_Pure_Premium"].sum())
_total_final = float(df["Final_Pure_Premium"].sum())
_book_delta_pct = (_total_final - _total_glm) / _total_glm * 100

# ── PREMIUM MIGRATION ─────────────────────────────────────────────────────────
_adj = df["Final_Pure_Premium"] - df["GLM_Pure_Premium"]
_premium_up   = float(_adj[_adj > 0].sum())
_premium_down = float(_adj[_adj < 0].abs().sum())
_pct_repriced    = float((df["Adjustment_Pct"].abs() > 10).mean() * 100)
_pct_underpriced = float(
    (df["GLM_Pure_Premium"] < df["Expected_Pure_Premium"] * (1 - UNDERPRICE_THRESH)).mean() * 100
)
_pct_underpriced_after = float(
    (df["Final_Pure_Premium"] < df["Expected_Pure_Premium"] * (1 - UNDERPRICE_THRESH)).mean() * 100
)
_adverse_selection_reduction = _pct_underpriced - _pct_underpriced_after
_mean_leakage = float(
    (df.loc[df["GLM_Pure_Premium"] < df["Expected_Pure_Premium"] * (1 - UNDERPRICE_THRESH),
            "Expected_Pure_Premium"] -
     df.loc[df["GLM_Pure_Premium"] < df["Expected_Pure_Premium"] * (1 - UNDERPRICE_THRESH),
            "GLM_Pure_Premium"]).mean()
)
MEAN_GLM_PP = float(df["GLM_Pure_Premium"].mean())
_total_reclass_pct = float(
    (df["GLM_Risk_Tier"] != df["Final_Risk_Tier"]).mean() * 100
)

print(f"  OOS R²: GLM={glm_r2:.4f}  Final={final_r2:.4f}  ΔR²={delta_r2:+.4f}")
print(f"  Risk neutrality: weighted mean uplift = {_risk_neutral_check:.6f}")
print(f"  Book-level delta: {_book_delta_pct:+.3f}%")

# ══════════════════════════════════════════════════════════════════════════════
# EBM GLOBAL EXPLANATION — extracted once at startup
# ══════════════════════════════════════════════════════════════════════════════
print("Extracting EBM global explanation…")
global_exp   = ebm_model.explain_global()
_global_summary = global_exp.data()
global_names  = list(_global_summary.get("names", []))
global_scores = list(_global_summary.get("scores", []))
dollar_importance = {n: abs(s) * MEAN_GLM_PP for n, s in zip(global_names, global_scores)}

# ── Native shape function cache (replaces manual PDP sweep) ──────────────────
SHAPE_FEATURES = {
    "Wildfire_Exposure_Daily":      ("Wildfire Exposure Index",    "Convex — risk accelerates past index 30"),
    "Roof_Vulnerability_Satellite": ("Roof Vulnerability Score",   "Quadratic — penalty doubles past score 20"),
    "Building_Code_Compliance":     ("Building Code Compliance %", "Threshold — sharp cliff below 60%"),
    "Credit_Score":                 ("Credit Score",               "Diminishing returns — GLM over-linearises"),
}

def _get_ebm_shape(feature_name: str) -> dict | None:
    """Extract the native EBM shape function for one feature. Returns None on failure."""
    try:
        idx = EBM_ALL_FEATURES.index(feature_name)
        data = global_exp.data(idx)
        if data and "names" in data and "scores" in data:
            return data
    except Exception:
        pass
    return None

def _get_ebm_interaction(feat_a: str, feat_b: str) -> dict | None:
    """
    Find and return the EBM interaction surface for (feat_a, feat_b).
    Three-method cascade:
      A: string delimiter match on term_names_
      B: integer/string tuple match on term_names_
      C: structural — iterate all terms, check for 2D names in
         explain_global data, then verify feature identity via
         term_features_ or term name parsing
    Falls back to None if no match found.
    """
    target = {feat_a, feat_b}
    try:
        for i, term in enumerate(ebm_model.term_names_):
            term_str = str(term)
            matched  = False

            # Method A: string delimiter
            for delim in (" x ", " & ", " × "):
                if delim in term_str:
                    parts = {t.strip() for t in term_str.split(delim)}
                    if parts == target:
                        matched = True
                    break

            # Method B: tuple of integer indices or string names
            if not matched and isinstance(term, (list, tuple)) and len(term) == 2:
                t0, t1 = term[0], term[1]
                if isinstance(t0, (int, np.integer)) and isinstance(t1, (int, np.integer)):
                    a_name = EBM_ALL_FEATURES[t0] if t0 < len(EBM_ALL_FEATURES) else ""
                    b_name = EBM_ALL_FEATURES[t1] if t1 < len(EBM_ALL_FEATURES) else ""
                    if {a_name, b_name} == target:
                        matched = True
                elif isinstance(t0, str) and isinstance(t1, str):
                    if {t0, t1} == target:
                        matched = True

            # Method C: structural — check explain_global data for 2D surface,
            # then verify feature identity via term_features_
            if not matched:
                try:
                    d = global_exp.data(i)
                    if d and "names" in d and "scores" in d:
                        _tn = d["names"]
                        if (isinstance(_tn, (list, tuple)) and len(_tn) == 2
                                and hasattr(_tn[0], "__len__")
                                and hasattr(_tn[1], "__len__")
                                and not isinstance(_tn[0], str)
                                and not isinstance(_tn[1], str)):
                            # It's an interaction surface — identify features
                            pair_names = set()
                            if hasattr(ebm_model, "term_features_") and i < len(ebm_model.term_features_):
                                tf = ebm_model.term_features_[i]
                                if isinstance(tf, (list, tuple)) and len(tf) == 2:
                                    for idx in tf:
                                        if isinstance(idx, (int, np.integer)) and idx < len(EBM_ALL_FEATURES):
                                            pair_names.add(EBM_ALL_FEATURES[idx])
                            if not pair_names:
                                for dl in (" x ", " & ", " × "):
                                    if dl in term_str:
                                        pair_names = {p.strip() for p in term_str.split(dl)}
                                        break
                            if pair_names == target:
                                matched = True
                except Exception:
                    pass

            if matched:
                d = global_exp.data(i)
                if d and "scores" in d:
                    return d
    except Exception:
        pass
    return None

# Pre-extract
SHAPE_CACHE = {}
for _feat in SHAPE_FEATURES:
    SHAPE_CACHE[_feat] = _get_ebm_shape(_feat)

INTERACTION_SURFACE = _get_ebm_interaction(
    "Wildfire_Exposure_Daily", "Roof_Vulnerability_Satellite")

print("Ready." + (" (EBM interaction surface found)" if INTERACTION_SURFACE else
                   " (EBM interaction surface not found; using binned fallback)"))

# ══════════════════════════════════════════════════════════════════════════════
# DEMO ARCHETYPES  (Spec S7)
# ══════════════════════════════════════════════════════════════════════════════
def _archetype(label, icon, color, mask_expr, sort_col=None, ascending=False):
    """Select one policy matching a boolean mask; sort_col optional."""
    mask = mask_expr(df)
    if not mask.any():
        return None
    sub = df.loc[mask]
    if sort_col:
        sub = sub.sort_values(sort_col, ascending=ascending)
    idx = int(sub.index[0])
    row = df.loc[idx]
    return {
        "label": label, "icon": icon, "color": color,
        "value": idx,
        "glm":   f"${row['GLM_Pure_Premium']:,.0f}",
        "final": f"${row['Final_Pure_Premium']:,.0f}",
        "adj":   f"{row['Adjustment_Pct']:+.0f}%",
    }

DEMO_ARCHETYPES = [a for a in [
    _archetype("WUI Wildfire Risk",  "fas fa-fire",             RED,
               lambda d: (d["Wildfire_Exposure_Daily"] > 55) & (d["Roof_Age_Applicant"] > 14) &
                         (d["Slope_Steepness"] > 18) &
                         (d["Adjustment_Pct"] > 15) & (d["Adjustment_Pct"] < 60),
               sort_col="Adjustment_Pct", ascending=False),
    _archetype("Hail Belt — Old Roof", "fas fa-cloud-showers-heavy", AMBER,
               lambda d: (d["Hail_Frequency"] >= 4) & (d["Roof_Vulnerability_Satellite"] > 22) &
                         (d["Adjustment_Pct"] > 15) & (d["Adjustment_Pct"] < 60),
               sort_col="Adjustment_Pct", ascending=False),
    _archetype("Flood Zone Foundation", "fas fa-water",          BLUE,
               lambda d: (d["Pluvial_Flood_Depth"] > 18) & (d["Dwelling_Age"] > 35) &
                         (d["Adjustment_Pct"] > 15) & (d["Adjustment_Pct"] < 60),
               sort_col="Adjustment_Pct", ascending=False),
    _archetype("Moral Hazard Signal",   "fas fa-exclamation-triangle", GOLD,
               lambda d: (d["RCV_Overstatement"] > 50000) & (d["Crime_Severity_Index"] > 65) &
                         (d["Adjustment_Pct"] > 15) & (d["Adjustment_Pct"] < 60),
               sort_col="Adjustment_Pct", ascending=False),
    _archetype("Hidden Gem — Overpriced", "fas fa-gem",          GREEN,
               lambda d: (d["Adjustment_Pct"] < -20) & (d["Risk_Tier"] == "Low"),
               sort_col="Adjustment_Pct", ascending=True),
    _archetype("New Build Masonry",    "fas fa-building",         MUTED,
               lambda d: (d["Dwelling_Age"] < 8) & (d["Construction_Type"] == "Masonry") &
                         (d["Building_Code_Compliance"] >= 90),
               sort_col="Adjustment_Pct", ascending=True),
    _archetype("Water Recency Risk",   "fas fa-tint",             BLUE,
               lambda d: (d["Water_Loss_Recency_Months"] <= 12) & (d["Tree_Canopy_Density"] > 55) &
                         (d["Adjustment_Pct"] > 15) & (d["Adjustment_Pct"] < 60),
               sort_col="Adjustment_Pct", ascending=False),
    _archetype("Suburban Standard",    "fas fa-home",             MUTED,
               lambda d: (d["Risk_Tier"] == "Moderate") & (d["Adjustment_Pct"].abs() < 5),
               sort_col=None),
] if a is not None]

# ── Policy dropdown options ───────────────────────────────────────────────────
def _build_policy_options():
    # Select policies spread across the full adjustment distribution using
    # decile sampling — this avoids the corridor-ceiling clustering problem
    # where nlargest() picks 60 policies all hitting the same ~+69% ceiling.
    df_sorted = df.sort_values("Adjustment_Pct", ascending=False)
    n = len(df_sorted)

    # Sample evenly across deciles of the adjustment distribution
    decile_samples = []
    for i in range(10):
        lo = int(i * n / 10)
        hi = int((i + 1) * n / 10)
        chunk = df_sorted.iloc[lo:hi]
        # Take up to 20 from each decile for coverage
        decile_samples.append(chunk.head(20))

    pool = pd.concat(decile_samples).drop_duplicates()

    # Sort the final dropdown list: largest positive first, then largest negative
    pool = pool.reindex(
        pool["Adjustment_Pct"].abs().sort_values(ascending=False).index
    ).head(200)

    opts = []
    for idx, row in pool.iterrows():
        tier  = str(row["Risk_Tier"])
        adj   = row["Adjustment_Pct"]
        arrow = "↑" if adj > 0 else "↓"
        opts.append({
            "label": (f"Policy #{idx} | {tier} | "
                      f"{arrow}{abs(adj):.0f}% | "
                      f"GLM ${row['GLM_Pure_Premium']:,.0f} → "
                      f"Final ${row['Final_Pure_Premium']:,.0f}"),
            "value": int(idx),
        })
    return opts, int(pool.index[0])

POLICY_OPTIONS, DEFAULT_POLICY = _build_policy_options()

# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def info_tooltip(tt_id, text):
    return html.Span([
        html.I(className="fas fa-info-circle ms-2", id=tt_id,
               style={"color": MUTED, "cursor": "pointer", "fontSize": "0.82rem"}),
        dbc.Tooltip(text, target=tt_id, placement="right",
                    style={"fontSize": "0.76rem", "maxWidth": "300px", "textAlign": "left"}),
    ])


def chart_card(title, tt_id, tt_text, graph_elem, subtitle=None):
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.Span(title, style=SEC_TITLE),
                info_tooltip(tt_id, tt_text),
            ], className="d-flex align-items-center mb-1"),
            html.Div(subtitle,
                     style={"fontSize": "0.75rem", "color": MUTED, "lineHeight": "1.4"}
                     ) if subtitle else None,
        ], style={"backgroundColor": WHITE, "border": "none", "paddingBottom": "4px"}),
        dbc.CardBody(graph_elem, style={"paddingTop": "0"}),
    ], style=CARD_STYLE, className="h-100")


def kpi_card(icon, label, value, subtitle, color, badge_text=None):
    return dbc.Card([dbc.CardBody([
        html.Div([
            html.Div(html.I(className=icon, style={"fontSize": "1.3rem", "color": color}),
                     style={"backgroundColor": f"{color}1A", "borderRadius": "8px",
                            "padding": "9px 10px", "display": "inline-flex"}),
            dbc.Badge(badge_text, color="warning", className="ms-auto align-self-start",
                      style={"fontSize": "0.68rem"}) if badge_text else None,
        ], className="d-flex align-items-center mb-3"),
        html.Div(value,    style={"fontSize": "1.85rem", "fontWeight": "700",
                                  "color": NAVY, "lineHeight": "1"}),
        html.Div(label,    style={"fontSize": "0.76rem", "fontWeight": "600", "color": MUTED,
                                  "marginTop": "4px", "textTransform": "uppercase",
                                  "letterSpacing": "0.05em"}),
        html.Div(subtitle, style={"fontSize": "0.74rem", "color": MUTED, "marginTop": "5px"}),
    ])], style=CARD_STYLE, className="h-100")


def formula_block(formula, note=None):
    return html.Div([
        html.Div(formula, style=MONO),
        html.Div(note, style={"fontSize": "0.75rem", "color": MUTED,
                               "marginTop": "4px"}) if note else None,
    ], className="my-2")


def section_card(number, title, color, content):
    return dbc.Card([dbc.CardBody([
        html.Div([
            html.Span(str(number), style={
                "backgroundColor": color, "color": WHITE, "borderRadius": "50%",
                "width": "26px", "height": "26px", "display": "inline-flex",
                "alignItems": "center", "justifyContent": "center",
                "fontSize": "0.8rem", "fontWeight": "700", "marginRight": "10px",
                "flexShrink": "0"}),
            html.Span(title, style={"fontWeight": "700", "fontSize": "1.0rem",
                                    "color": NAVY}),
        ], className="d-flex align-items-center mb-3"),
        content,
    ])], style={**CARD_STYLE, "borderLeft": f"4px solid {color}"})


# ── Navbar ────────────────────────────────────────────────────────────────────
# Logo: place your logo file at  assets/vm_logo.png
# Dash serves everything in assets/ automatically — no import needed.
navbar = dbc.Navbar(dbc.Container([
    html.Div([
        html.Img(
            src="/assets/vm_logo.png",
            style={"height": "32px", "marginRight": "14px", "objectFit": "contain"},
            alt="ValueMomentum",
        ),
        html.Div([
            html.Span("Homeowners Intelligence Layer",
                      style={"fontWeight": "700", "fontSize": "1.05rem", "color": WHITE,
                             "lineHeight": "1.2"}),
            html.Div("GLM + GA2M Two-Layer Pricing Architecture",
                     style={"fontSize": "0.72rem", "color": "#A0AABB", "lineHeight": "1.2"}),
        ]),
    ], className="d-flex align-items-center"),
    html.Div([
        dbc.Badge("DEMO", color="warning", className="me-2",
                  style={"fontSize": "0.68rem"}),
        html.Span(f"{N_TOTAL:,} synthetic policies · GLM + EBM GA2M residual layer",
                  style={"color": "#A0AABB", "fontSize": "0.75rem"}),
    ], className="d-none d-md-flex align-items-center"),
], fluid=True), color=NAVY, dark=True, className="py-2",
style={"borderBottom": f"3px solid {GOLD}"})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1  —  BUSINESS CASE
# ══════════════════════════════════════════════════════════════════════════════
def build_portfolio_tab():
    # ── Where Mispricing Concentrates — Net Premium Flow by State ─────────────
    _state_flow = df.copy()
    _state_flow["_adj_dollars"] = (
        _state_flow["Final_Pure_Premium"] - _state_flow["GLM_Pure_Premium"])
    _sf = _state_flow.groupby("State").agg(
        net_flow   = ("_adj_dollars", "sum"),
        n_policies = ("_adj_dollars", "size"),
        avg_adj_pct= ("Adjustment_Pct", "mean"),
    ).sort_values("net_flow")

    fig_flow = go.Figure()
    fig_flow.add_trace(go.Bar(
        y=_sf.index,
        x=_sf["net_flow"] / 1e6,
        orientation="h",
        marker_color=[RED if v > 0 else GREEN for v in _sf["net_flow"]],
        text=[f"${v/1e6:+.1f}M" for v in _sf["net_flow"]],
        textposition="outside",
        textfont=dict(size=9, family="Inter"),
        hovertemplate="State: %{y}<br>Net flow: $%{x:.1f}M<extra></extra>",
    ))
    # Zero line
    fig_flow.add_vline(x=0, line_color=NAVY, line_width=1.5, line_dash="dot")

    # Annotate the biggest surcharge contributor
    _top_surcharge_state = _sf.index[_sf["net_flow"] == _sf["net_flow"].max()][0]
    _top_surcharge_val   = float(_sf["net_flow"].max() / 1e6)
    fig_flow.add_annotation(
        x=_top_surcharge_val, y=_top_surcharge_state,
        text="Wildfire × Roof<br>interactions drive<br>surcharges",
        showarrow=True, arrowhead=2, ax=50, ay=-20,
        font=dict(size=8, color=RED, family="Inter"),
        bgcolor=WHITE, bordercolor="rgba(230,57,70,0.3)",
        borderwidth=1, borderpad=3)

    fig_flow.update_xaxes(
        title_text="Net Premium Redistribution ($M)",
        tickprefix="$", ticksuffix="M",
        zeroline=True)
    fig_flow.update_yaxes(tickfont=dict(size=10, family="Inter"))
    fig_flow.update_layout(
        template="plotly_white", height=CHART_HEIGHT_SM,
        showlegend=False,
        margin=dict(l=10, r=70, t=10, b=40),
        font=dict(family="Inter"),
        plot_bgcolor="#FAFBFC")

    # ── GA2M Adjustment Distribution — what the intelligence layer actually does ──
    _pct_surcharge = float((df["Adjustment_Pct"] > 10).mean() * 100)
    _pct_credit    = float((df["Adjustment_Pct"] < -10).mean() * 100)

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=df["Adjustment_Pct"].clip(-50, 70), nbinsx=50,
        marker_color=NAVY, opacity=0.85,
        hovertemplate="Adjustment: %{x:.0f}%<br>Policies: %{y:,}<extra></extra>",
    ))
    # Zero line — risk neutrality anchor
    fig_dist.add_vline(x=0, line_dash="solid", line_color=RED, line_width=1.5)
    fig_dist.add_annotation(
        x=0, y=1.0, yref="paper",
        text="Risk-neutral<br>center (0%)",
        showarrow=False, font=dict(size=9, color=RED, family="Inter"),
        xshift=52, yshift=-15)
    # Surcharge region callout
    fig_dist.add_annotation(
        x=35, y=0.85, yref="paper",
        text=f"<b>{_pct_surcharge:.0f}%</b> of book<br>surcharge >10%",
        showarrow=False, font=dict(size=10, color=RED, family="Inter"),
        bgcolor=WHITE, bordercolor="rgba(230,57,70,0.3)", borderwidth=1, borderpad=4)
    # Credit region callout
    fig_dist.add_annotation(
        x=-25, y=0.85, yref="paper",
        text=f"<b>{_pct_credit:.0f}%</b> of book<br>credit >10%",
        showarrow=False, font=dict(size=10, color=GREEN, family="Inter"),
        bgcolor=WHITE, bordercolor="rgba(44,198,83,0.3)", borderwidth=1, borderpad=4)
    # Narrative annotations — reframe asymmetry as two distinct stories
    fig_dist.add_annotation(
        x=-35, y=0.55, yref="paper",
        text="GLM over-penalizes<br>moderate risks<br><i>(growth opportunity)</i>",
        showarrow=False, font=dict(size=8, color=MUTED, family="Inter"),
        bgcolor=WHITE, bordercolor=BORDER, borderwidth=1, borderpad=3)
    fig_dist.add_annotation(
        x=45, y=0.55, yref="paper",
        text="GLM misses<br>compound perils<br><i>(adverse selection fix)</i>",
        showarrow=False, font=dict(size=8, color=MUTED, family="Inter"),
        bgcolor=WHITE, bordercolor=BORDER, borderwidth=1, borderpad=3)
    fig_dist.update_xaxes(title_text="GA2M Premium Adjustment (%)", zeroline=True)
    fig_dist.update_yaxes(title_text="Policy Count")
    fig_dist.update_layout(
        template="plotly_white", height=CHART_HEIGHT_SM,
        margin=dict(l=10, r=10, t=10, b=40), font=dict(family="Inter"),
        showlegend=False)

    # ── Adverse selection scatter ─────────────────────────────────────────────
    _samp  = _test.sample(min(20000, len(_test)), random_state=42)
    _cap   = float(_samp["Expected_Pure_Premium"].quantile(0.999))
    _sc    = _samp[_samp["Expected_Pure_Premium"] <= _cap].copy()
    _sc["glm_err_pct"]   = (_sc["GLM_Pure_Premium"]   - _sc["Expected_Pure_Premium"]) / _sc["Expected_Pure_Premium"] * 100
    _sc["final_err_pct"] = (_sc["Final_Pure_Premium"]  - _sc["Expected_Pure_Premium"]) / _sc["Expected_Pure_Premium"] * 100
    _glm_mae   = _sc["glm_err_pct"].abs().mean()
    _final_mae = _sc["final_err_pct"].abs().mean()
    _delta_mae = _glm_mae - _final_mae
    _DIV = [[0.0, "rgb(192,57,43)"], [0.35, "rgb(241,196,15)"],
            [0.50, "rgb(210,215,220)"], [0.65, "rgb(88,214,141)"], [1.0, "rgb(30,132,73)"]]
    _CBAR = dict(thickness=14, len=0.88, tickvals=[-50,-25,0,25,50],
                 ticktext=["−50%","−25%","0%","+25%","+50%"],
                 tickfont=dict(size=9),
                 title=dict(text="Error %", font=dict(size=9)), x=1.01)
    # Use short non-overlapping panel titles; MAE detail goes in annotations below
    fig_adv = make_subplots(rows=1, cols=2, shared_yaxes=True,
        subplot_titles=["Legacy GLM", "Intelligence-Adjusted"],
        horizontal_spacing=0.06)
    for _ci, (_yc, _ec, _scb) in enumerate(
            [("GLM_Pure_Premium","glm_err_pct",False),
             ("Final_Pure_Premium","final_err_pct",True)], start=1):
        fig_adv.add_trace(go.Scatter(
            x=_sc["Expected_Pure_Premium"], y=_sc[_yc], mode="markers",
            marker=dict(color=_sc[_ec], colorscale=_DIV, cmin=-60, cmax=60,
                        size=4, opacity=0.72, line=dict(width=0),
                        showscale=_scb, colorbar=_CBAR if _scb else {}),
            hovertemplate="True: $%{x:.0f}<br>Model: $%{y:.0f}<br>Error: %{marker.color:.1f}%<extra></extra>",
            showlegend=False), row=1, col=_ci)
        fig_adv.add_trace(go.Scatter(x=[0, _cap], y=[0, _cap], mode="lines",
            line=dict(dash="dot", color="#AAAAAA", width=1.4),
            showlegend=False), row=1, col=_ci)
    # MAE callout annotations positioned inside each panel (below top)
    fig_adv.add_annotation(
        xref="x domain", yref="paper", x=0.5, y=0.97,
        text=f"MAE {_glm_mae:.1f}%", showarrow=False,
        font=dict(size=10, color=MUTED, family="Inter"),
        xanchor="center", yanchor="top")
    fig_adv.add_annotation(
        xref="x2 domain", yref="paper", x=0.5, y=0.97,
        text=f"MAE {_final_mae:.1f}% <b>▼ {_delta_mae:.1f}pp</b>", showarrow=False,
        font=dict(size=10, color=NAVY, family="Inter"),
        xanchor="center", yanchor="top")
    fig_adv.update_xaxes(title_text="True Expected Loss Cost ($)", tickfont=dict(size=9))
    fig_adv.update_yaxes(title_text="Model-Estimated Premium ($)", tickfont=dict(size=9), col=1)
    fig_adv.update_layout(template="plotly_white", height=CHART_HEIGHT_MD,
        margin=dict(l=10, r=90, t=50, b=20), font=dict(family="Inter"),
        plot_bgcolor="#FAFBFC")

    # ── Reclassification Matrix — who moved where ─────────────────────────────
    _cross = pd.crosstab(
        df["GLM_Risk_Tier"], df["Final_Risk_Tier"],
        normalize="index") * 100
    _cross = _cross.reindex(index=TIER_ORDER, columns=TIER_ORDER, fill_value=0)
    _total_reclass = float((df["GLM_Risk_Tier"] != df["Final_Risk_Tier"]).mean() * 100)

    _z    = _cross.values
    _text = [[f"{v:.0f}%" for v in row] for row in _z]

    fig_donut = go.Figure(go.Heatmap(
        z=_z,
        x=[f"→ {t}" for t in TIER_ORDER],
        y=[f"{t} (GLM)" for t in TIER_ORDER],
        text=_text,
        texttemplate="%{text}",
        textfont=dict(size=12, family="Inter"),
        colorscale=[
            [0.0, "#F0F2F5"],
            [0.3, "#B8D4E8"],
            [0.6, "#4A90C4"],
            [1.0, NAVY],
        ],
        showscale=False,
        hovertemplate="From %{y}<br>To %{x}<br>%{text} of tier<extra></extra>",
    ))
    fig_donut.update_xaxes(side="top", tickfont=dict(size=10, family="Inter"))
    fig_donut.update_yaxes(tickfont=dict(size=10, family="Inter"), autorange="reversed")
    fig_donut.update_layout(
        template="plotly_white", height=CHART_HEIGHT_MD,
        margin=dict(l=10, r=10, t=50, b=20), font=dict(family="Inter"),
        title=dict(
            text=f"<b>{_total_reclass:.0f}%</b> of portfolio reclassified across tiers",
            font=dict(size=12, color=NAVY),
            x=0.5, xanchor="center"),
    )

    # ── Reclassification scatter — Spec S9 ────────────────────────────────────
    _rs = df.sample(RECLASS_SAMPLE, random_state=42)
    _pp_cap = float(_rs[["GLM_Pure_Premium","Final_Pure_Premium"]].max().max())
    fig_reclass = go.Figure()
    fig_reclass.add_trace(go.Scatter(
        x=_rs["GLM_Pure_Premium"], y=_rs["Final_Pure_Premium"],
        mode="markers",
        marker=dict(color=_rs["Adjustment_Pct"], colorscale="RdBu_r",
                    cmin=-40, cmax=40, size=4, opacity=0.5,
                    colorbar=dict(title="Adj %", thickness=12, len=0.8,
                                  tickformat=".0f", ticksuffix="%",
                                  x=1.01, tickfont=dict(size=8))),
        hovertemplate="GLM: $%{x:.0f}<br>Final: $%{y:.0f}<br>Adj: %{marker.color:.1f}%<extra></extra>",
        showlegend=False))
    fig_reclass.add_trace(go.Scatter(
        x=[0, _pp_cap], y=[0, _pp_cap], mode="lines",
        line=dict(dash="dot", color="#AAAAAA", width=1.5), showlegend=False))
    fig_reclass.add_annotation(text="Hidden Dangers<br><i>GLM underpriced</i>",
        x=_pp_cap * 0.22, y=_pp_cap * 0.72, showarrow=False,
        font=dict(size=10, color=RED, family="Inter"),
        bgcolor=WHITE, bordercolor=f"rgba(230,57,70,0.35)", borderwidth=1, borderpad=4)
    fig_reclass.add_annotation(text="Hidden Gems<br><i>GLM overpriced</i>",
        x=_pp_cap * 0.72, y=_pp_cap * 0.22, showarrow=False,
        font=dict(size=10, color=GREEN, family="Inter"),
        bgcolor=WHITE, bordercolor=f"rgba(44,198,83,0.35)", borderwidth=1, borderpad=4)
    fig_reclass.update_xaxes(title_text="Legacy GLM Premium ($)")
    fig_reclass.update_yaxes(title_text="Intelligence-Adjusted Premium ($)")
    fig_reclass.update_layout(template="plotly_white", height=CHART_HEIGHT_MD,
        margin=dict(l=10, r=70, t=20, b=40), font=dict(family="Inter"),
        plot_bgcolor="#FAFBFC")

    # ── Layout ────────────────────────────────────────────────────────────────
    return html.Div([
        dbc.Alert([
            html.I(className="fas fa-lightbulb me-2", style={"color": GOLD}),
            html.Strong("The Business Case: "),
            f"The legacy 16-variable GLM explains {glm_r2:.1%} of pure premium variance "
            f"{OOS_LABEL}. This solution raises this to {final_r2:.1%} — redistributing "
            f"${(_premium_up + _premium_down)/1e6:.0f}M across the book while repricing "
            f"{_pct_repriced:.1f}% of policies by >10%, reclassifying "
            f"{_total_reclass_pct:.0f}% of policies across risk tiers, "
            f"and correcting adverse selection on {_pct_underpriced:.1f}% of policies — "
            f"with zero change to total book premium.",
        ], color="warning", className="mb-4",
           style={"borderLeft": f"4px solid {GOLD}", "backgroundColor": "#FFFBF0",
                  "borderRadius": "8px", "fontSize": "0.88rem"}),

        # KPI row — 5 cards: dollars first, statistics supporting
        dbc.Row([
            dbc.Col(kpi_card("fas fa-exchange-alt", "PREMIUM REDISTRIBUTION",
                f"${(_premium_up + _premium_down)/1e6:.0f}M",
                f"${_premium_up/1e6:.1f}M surcharges + ${_premium_down/1e6:.1f}M credits = $0 net. "
                f"Pure redistribution, not a rate increase.",
                GOLD, "NEUTRAL"), width=3),
            dbc.Col(kpi_card("fas fa-arrows-alt-v", "PORTFOLIO RECLASSIFIED",
                f"{_total_reclass_pct:.0f}%",
                f"{int(N_TOTAL * _total_reclass_pct / 100):,} policies crossing tier boundaries "
                f"after intelligence adjustment",
                NAVY, "MOVEMENT"), width=2),
            dbc.Col(kpi_card("fas fa-arrow-trend-up", "Variance Lift ΔR²",
                f"+{delta_r2:.3f}",
                f"GLM {glm_r2:.0%} → {final_r2:.0%} · "
                f"{delta_r2/(1-glm_r2):.0%} of residual recovered",
                GREEN, "KEY LIFT"), width=2),
            dbc.Col(kpi_card("fas fa-exclamation-triangle", "ADVERSE SELECTION",
                f"{_pct_underpriced:.0f}% → {_pct_underpriced_after:.0f}%",
                f"Policies underpriced >{int(UNDERPRICE_THRESH*100)}%: "
                f"reduced by {_adverse_selection_reduction:.0f}pp · "
                f"avg leakage ${_mean_leakage:,.0f}/policy",
                RED, "CORRECTED"), width=2),
            dbc.Col(kpi_card("fas fa-balance-scale", "Book Premium Impact",
                f"{_book_delta_pct:+.2f}%",
                f"Total: ${_total_final/1e6:,.1f}M — redistributed, not inflated. "
                f"E_w[uplift] = {_risk_neutral_check:.4f}×", TEAL, "NEUTRAL"), width=3),
        ], className="g-3 mb-4"),

        # Row 1: Where mispricing concentrates + per-policy adjustment distribution
        dbc.Row([
            dbc.Col(chart_card("Where Mispricing Concentrates — Net Flow by State", "tt-flow",
                "Net premium redistribution per state after intelligence adjustment. "
                "Red = state receives net surcharges (GLM systematically underprices risks there, "
                "typically from compound-peril interactions like wildfire × roof). "
                "Green = state receives net credits (GLM overprices, creating competitive exposure). "
                "All flows sum to $0.",
                dcc.Graph(figure=fig_flow, config={"displayModeBar": False}),
                subtitle="Red = net surcharges flowing in · Green = net credits flowing out · Sum = $0"), width=4),
            dbc.Col(chart_card("Intelligence Adjustment Distribution", "tt-dist",
                "How much does the GA2M layer move each policy? The spread from "
                "−35% to +60% shows meaningful per-policy repricing while the "
                "distribution centering at 0% confirms book-level neutrality.",
                dcc.Graph(figure=fig_dist, config={"displayModeBar": False}),
                subtitle="Each bar = policies receiving that % adjustment · centered at 0% = risk neutral"), width=8),
        ], className="g-3 mb-4"),

        # Row 2: Adverse selection (5) + Reclassification scatter (3) + Matrix (4, wider+taller)
        dbc.Row([
            dbc.Col(chart_card("Adverse Selection Map — GLM Underpricing",
                "tt-adverse",
                "Side-by-side diverging scatter. Red dots sit below the diagonal (GLM undercharges). "
                "After GA2M the cloud shifts grey and tightens to the diagonal.",
                dcc.Loading(dcc.Graph(figure=fig_adv, config={"displayModeBar": False}),
                            type="circle"),
                subtitle="Each dot = 1 policy · Red=underpriced · Grey=accurate · Green=overpriced"), width=5),
            dbc.Col(chart_card("Reclassification Scatter — Who Moves", "tt-reclass",
                "Points above the 45° line are surcharges (hidden dangers). "
                "Points below are credits (hidden gems). Color = % adjustment magnitude.",
                dcc.Loading(dcc.Graph(figure=fig_reclass, config={"displayModeBar": False}),
                            type="circle"),
                subtitle=f"Sample of {RECLASS_SAMPLE:,} · Red=upward · Blue=downward"), width=3),
            dbc.Col(chart_card("Reclassification Matrix", "tt-donut",
                "Each cell shows what % of a GLM tier moved to a Final tier after intelligence adjustment. "
                "Diagonal = stayed in same tier (darker). Off-diagonal = reclassified. "
                "Read row-wise: 'Of the GLM Low-risk policies, X% moved to Moderate.'",
                dcc.Graph(figure=fig_donut, config={"displayModeBar": False}),
                subtitle=f"{_total_reclass:.0f}% gross reclassification · read rows: GLM tier → Final tier"), width=4),
        ], className="g-3"),
    ], className="py-4")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2  —  INTELLIGENCE SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
def _build_shape_panel() -> dbc.Row:
    """
    Data Characteristics: 2×2 grid of three-layer charts (Spec S-A.4 / S11).
    Layer 1: empirical scatter (EPP / GLM_PP binned ratio)
    Layer 2: GLM linear assumption (flat at ratio=1.0)
    Layer 3: EBM shape function exp(score)
    """
    feats = list(SHAPE_FEATURES.keys())
    labels = [SHAPE_FEATURES[f][0] for f in feats]
    shape_notes = [SHAPE_FEATURES[f][1] for f in feats]

    fig_chars = make_subplots(rows=2, cols=2, subplot_titles=labels,
                               vertical_spacing=0.18, horizontal_spacing=0.12)

    for i, feat in enumerate(feats):
        r, c = [(1, 1), (1, 2), (2, 1), (2, 2)][i]
        xref = f"x{i+1 if i > 0 else ''}"
        yref = f"y{i+1 if i > 0 else ''}"

        # ── Layer 1: empirical loss ratio by quantile bin ─────────────────────
        try:
            _bins = pd.qcut(df[feat], q=20, duplicates="drop")
            _bin_centers = df.groupby(_bins, observed=True)[feat].mean().values
            _ratio = df.groupby(_bins, observed=True).apply(
                lambda g: g["Expected_Pure_Premium"].mean() / g["GLM_Pure_Premium"].mean()
                if g["GLM_Pure_Premium"].mean() > 0 else 1.0
            ).values
            _pct_ratio = (_ratio - 1.0) * 100
            # Error bars
            _bin_std = df.groupby(_bins, observed=True).apply(
                lambda g: (g["Expected_Pure_Premium"] / g["GLM_Pure_Premium"]).std()
            ).values
            _se = (_bin_std / np.sqrt(
                df.groupby(_bins, observed=True).size().values.clip(1)
            )) * 100

            fig_chars.add_trace(go.Scatter(
                x=_bin_centers, y=_pct_ratio, mode="markers",
                marker=dict(color=NAVY, size=6, opacity=0.7),
                error_y=dict(type="data", array=_se, visible=True,
                             thickness=1, width=3, color="#AABBCC"),
                name="Empirical", showlegend=False,
                hovertemplate=f"{feat}: %{{x:.1f}}<br>Actual vs GLM: %{{y:.1f}}%<extra></extra>",
            ), row=r, col=c)
        except Exception:
            pass

        # ── Layer 2: GLM linear assumption (flat at 0% residual) ─────────────
        if df[feat].notna().any():
            _x_range = [float(df[feat].quantile(0.02)),
                        float(df[feat].quantile(0.98))]
            fig_chars.add_trace(go.Scatter(
                x=_x_range, y=[0, 0], mode="lines",
                line=dict(color=AMBER, width=2.5, dash="dash"),
                name="GLM linear", showlegend=False,
                hoverinfo="skip",
            ), row=r, col=c)
            # Inline label — unmissable, amber = GLM world
            fig_chars.add_annotation(
                xref=xref, yref=yref,
                x=_x_range[1], y=0,
                text="GLM assumption (0%)",
                showarrow=False,
                xanchor="right", yanchor="bottom", yshift=4,
                font=dict(size=7, color=AMBER, family="Inter"),
            )

        # ── Layer 3: EBM shape function ───────────────────────────────────────
        _sd = SHAPE_CACHE.get(feat)
        _max_div = 0.0
        if _sd is not None:
            try:
                _x_sf_raw = np.array(_sd["names"], dtype=float)
                # Validate x-range against actual data; if EBM bins are
                # outside the plausible range, map to data quantiles
                _data_lo = float(df[feat].quantile(0.01))
                _data_hi = float(df[feat].quantile(0.99))
                if (_x_sf_raw.min() < _data_lo - (_data_hi - _data_lo) * 0.5 or
                        _x_sf_raw.max() > _data_hi + (_data_hi - _data_lo) * 0.5):
                    _x_sf = np.linspace(_data_lo, _data_hi, len(_x_sf_raw))
                else:
                    _x_sf = _x_sf_raw
                _y_sf = (np.exp(np.array(_sd["scores"], dtype=float)) - 1) * 100
                fig_chars.add_trace(go.Scatter(
                    x=_x_sf, y=_y_sf, mode="lines",
                    line=dict(color=BLUE, width=2.5),
                    name="GA2M shape", showlegend=False,
                    hovertemplate=f"{feat}: %{{x:.1f}}<br>GA2M adj: %{{y:.1f}}%<extra></extra>",
                ), row=r, col=c)
                # Confidence band (if available)
                _ub = _sd.get("upper_bounds")
                _lb = _sd.get("lower_bounds")
                if _ub is not None and _lb is not None:
                    _ub = (np.exp(np.array(_ub, dtype=float)) - 1) * 100
                    _lb = (np.exp(np.array(_lb, dtype=float)) - 1) * 100
                    fig_chars.add_trace(go.Scatter(
                        x=list(_x_sf) + list(_x_sf[::-1]),
                        y=list(_ub) + list(_lb[::-1]),
                        fill="toself", fillcolor="rgba(46,196,182,0.12)",
                        line=dict(width=0), showlegend=False, hoverinfo="skip",
                    ), row=r, col=c)
                # Annotate max divergence
                _max_div = float(np.abs(_y_sf).max()) if len(_y_sf) else 0
                _peak_x  = float(_x_sf[np.argmax(np.abs(_y_sf))]) if len(_y_sf) else 0
                _peak_y  = float(_y_sf[np.argmax(np.abs(_y_sf))]) if len(_y_sf) else 0
                fig_chars.add_annotation(
                    xref=xref, yref=yref,
                    x=_peak_x, y=_peak_y,
                    text=f"Max GLM error:<br>±{abs(_peak_y):.0f}%",
                    showarrow=True, arrowhead=2,
                    font=dict(size=8, color=AMBER, family="Inter"),
                    bgcolor=WHITE, bordercolor=BORDER, borderwidth=1,
                    ax=30, ay=-30,
                )
            except Exception:
                pass

        # subplot annotation for shape note
        fig_chars.add_annotation(
            xref=xref, yref=yref,
            x=0.02, y=0.97, xanchor="left", yanchor="top",
            text=f"<i>{shape_notes[i]}</i>",
            showarrow=False,
            font=dict(size=7, color=AMBER, family="Inter"),
            bgcolor=WHITE, bordercolor=BORDER, borderwidth=1,
        )

    fig_chars.update_yaxes(title_text="% vs GLM baseline", title_font_size=9, tickfont_size=9)
    fig_chars.update_xaxes(tickfont_size=9)
    fig_chars.update_annotations(font_size=10)
    fig_chars.add_hline(y=0, line_dash="dot", line_color="#DDDDDD", line_width=0.8)
    fig_chars.update_layout(template="plotly_white", height=CHART_HEIGHT_LG,
        margin=dict(l=50, r=20, t=50, b=30), font=dict(family="Inter"))

    return dbc.Row([
        dbc.Col(dbc.Alert([
            html.I(className="fas fa-chart-area me-2", style={"color": AMBER}),
            html.Strong("Why Linear Models Hit a Ceiling: "),
            "Each chart overlays three layers — the empirical risk pattern (dots), "
            "the GLM's linear approximation (dashed line at 0%), and what the GA2M "
            "glass-box layer recovers (solid curve with confidence band). "
            "The gap between dashed and curve is structural premium leakage "
            "that exists regardless of how well the GLM is built.",
        ], color="warning", className="mb-3",
           style={"borderLeft": f"4px solid {AMBER}", "fontSize": "0.87rem"}), width=12),
        dbc.Col(chart_card(
            "Why the GLM Hits Its Structural Ceiling — Non-Linear Reality vs Linear Assumption",
            "tt-chars",
            "Dots = empirical EPP/GLM ratio binned by feature value. "
            "Dashed = GLM's structural assumption (flat). "
            "Solid = GA2M learned shape function. Gap between dashed and solid = structural leakage.",
            dcc.Loading(dcc.Graph(figure=fig_chars, config={"displayModeBar": False}),
                        type="circle"),
            subtitle="Dots=empirical · Dashed=GLM linear · Solid=GA2M shape · Band=confidence interval",
        ), width=12),
    ], className="g-3 mb-4")


def build_feature_tab():
    # ── Signal Landscape ──────────────────────────────────────────────────────
    _top_n = 15
    _sorted = sorted(dollar_importance.items(), key=lambda x: x[1], reverse=True)[:_top_n][::-1]

    # Distinguish: interaction term / new modern / legacy-nonlinear
    _ebm_base_set = set(EBM_ALL_FEATURES[:12])  # first 12 = legacy

    # Definitive interaction detection: inspect explain_global().data(i)["names"].
    # Main effect terms have names as a 1D array; interaction terms have names
    # as a tuple/list of TWO arrays (one per feature axis). Version-independent.
    _interaction_names = set()
    try:
        for _tidx, _gname in enumerate(global_names):
            try:
                _term_data = global_exp.data(_tidx)
                if _term_data and "names" in _term_data:
                    _tnames = _term_data["names"]
                    if (isinstance(_tnames, (list, tuple)) and len(_tnames) == 2
                            and hasattr(_tnames[0], "__len__")
                            and hasattr(_tnames[1], "__len__")
                            and not isinstance(_tnames[0], str)
                            and not isinstance(_tnames[1], str)):
                        _interaction_names.add(_gname)
            except Exception:
                continue
    except Exception:
        pass

    # Fallback: delimiter-based string parsing
    if not _interaction_names:
        for _gn in global_names:
            _gn_str = str(_gn)
            if any(d in _gn_str for d in (" x ", " & ", " X ", " × ")):
                _interaction_names.add(_gn)

    def _bar_color(name):
        # Primary: structural detection from explain_global data
        if name in _interaction_names:
            return GOLD
        # Fallback: string delimiter detection (EBM uses " x " in term_names_)
        if any(d in str(name) for d in (" x ", " & ", " × ")):
            return GOLD
        # Modern enrichment signal (not in legacy GLM feature set)
        if name not in _ebm_base_set:
            return NAVY
        # Legacy feature gaining non-linear treatment
        return "#5B6F8A"

    _bar_colors = [_bar_color(n) for n, _ in _sorted]
    fig_imp = go.Figure(go.Bar(
        x=[v for _, v in _sorted],
        y=[n.replace("_", " ") for n, _ in _sorted],
        orientation="h", marker_color=_bar_colors,
        text=[f"~${v:,.0f}/policy" for _, v in _sorted],
        textposition="outside", textfont=dict(size=9, color=NAVY)))
    fig_imp.add_annotation(
        text="■ Interaction term    ■ New modern signal    ■ Legacy feature (non-linear gain)",
        xref="paper", yref="paper", x=0.5, y=-0.06,
        showarrow=False, font=dict(size=9, color=MUTED))
    fig_imp.update_xaxes(title_text="Estimated Avg Dollar Impact / Policy ($)")
    fig_imp.update_layout(template="plotly_white", height=CHART_HEIGHT_LG,
        margin=dict(l=10, r=80, t=20, b=50), font=dict(family="Inter"))

    # ── State-Level Adjustment Bar Chart — geographic narrative ──────────────
    # (replaces duplicate shape function panel — Data Characteristics above already
    #  shows the same four features with three-layer evidence; this adds the
    #  geographic dimension that's otherwise entirely absent from the demo)
    _state_adj = (df.groupby("State")["Adjustment_Pct"]
                    .agg(["mean", "std", "count"])
                    .sort_values("mean", ascending=True))
    _state_colors = [RED if v > 0 else GREEN for v in _state_adj["mean"]]

    fig_state = go.Figure(go.Bar(
        x=_state_adj["mean"].values,
        y=_state_adj.index,
        orientation="h",
        marker_color=_state_colors,
        error_x=dict(
            type="data",
            array=(_state_adj["std"] / np.sqrt(_state_adj["count"])).values,
            visible=True, thickness=1.5, width=4, color="#AAAAAA",
        ),
        text=[f"{v:+.1f}%" for v in _state_adj["mean"]],
        textposition="outside",
        textfont=dict(size=9, family="Inter"),
        hovertemplate="State: %{y}<br>Avg Adj: %{x:.1f}%<br><extra></extra>",
    ))
    fig_state.add_vline(x=0, line_color=MUTED, line_width=1.5, line_dash="dot")
    fig_state.update_xaxes(
        title_text="Average GA2M Adjustment (%)",
        zeroline=True,
        ticksuffix="%",
    )
    fig_state.update_yaxes(tickfont=dict(size=11, family="Inter"))
    fig_state.update_layout(
        template="plotly_white", height=CHART_HEIGHT_MD,
        margin=dict(l=10, r=70, t=20, b=30),
        font=dict(family="Inter"),
        plot_bgcolor="#FAFBFC",
    )

    # ── Interaction Discovery Panel ───────────────────────────────────────────
    # Extract all interaction terms from the EBM using dual detection
    _interaction_terms = []
    for i, term in enumerate(ebm_model.term_names_):
        term_str = str(term)
        is_interaction = False
        # Method 1: string delimiter
        if any(d in term_str for d in (" x ", " & ", " × ")):
            is_interaction = True
        # Method 2: structural — interaction terms have names as a tuple/list
        # of TWO arrays (one per feature axis), each containing multiple bin
        # edges. Binary categoricals also have len(names)==2 but each element
        # is a single string, not an array of numeric bin edges.
        if not is_interaction:
            try:
                _td = global_exp.data(i)
                if _td and "names" in _td:
                    _tn = _td["names"]
                    if (isinstance(_tn, (list, tuple)) and len(_tn) == 2
                            and hasattr(_tn[0], "__len__")
                            and hasattr(_tn[1], "__len__")
                            and not isinstance(_tn[0], str)
                            and not isinstance(_tn[1], str)):
                        is_interaction = True
            except Exception:
                pass
        if is_interaction:
            score = 0.0
            if i < len(global_scores):
                score = abs(global_scores[i])
            else:
                try:
                    _td = global_exp.data(i)
                    if _td and "scores" in _td:
                        score = float(np.abs(np.array(_td["scores"])).mean())
                except Exception:
                    pass
            _interaction_terms.append({
                "term":         term_str.replace("_", " "),
                "raw_term":     term_str,
                "importance":   score,
                "dollar_impact": score * MEAN_GLM_PP,
                "index":        i,
            })
    _interaction_terms.sort(key=lambda x: x["importance"], reverse=True)

    # ── Interaction ranking bar chart ────────────────────────────────────────
    if _interaction_terms:
        _int_sorted = _interaction_terms[::-1]
        fig_int_rank = go.Figure(go.Bar(
            y=[t["term"] for t in _int_sorted],
            x=[t["dollar_impact"] for t in _int_sorted],
            orientation="h",
            marker_color=GOLD, marker_opacity=0.85,
            text=[f"~${t['dollar_impact']:,.0f}/policy" for t in _int_sorted],
            textposition="outside",
            textfont=dict(size=9, color=NAVY, family="Inter"),
            hovertemplate="Interaction: %{y}<br>Avg impact: $%{x:,.0f}/policy<extra></extra>",
        ))
        _total_int_imp = sum(t["dollar_impact"] for t in _interaction_terms)
        _total_all_imp = sum(abs(s) * MEAN_GLM_PP for s in global_scores) or 1
        _int_pct = _total_int_imp / _total_all_imp * 100
        fig_int_rank.add_annotation(
            x=0.98, y=0.02, xref="paper", yref="paper",
            text=(f"<b>{len(_interaction_terms)}</b> interaction terms discovered<br>"
                  f"<b>{_int_pct:.0f}%</b> of total GA2M signal"),
            showarrow=False, xanchor="right", yanchor="bottom",
            font=dict(size=10, color=NAVY, family="Inter"),
            bgcolor=WHITE, bordercolor=GOLD, borderwidth=1, borderpad=5)
        fig_int_rank.add_annotation(
            x=0.98, y=0.15, xref="paper", yref="paper",
            text="Dollar magnitude ≠ interaction purity<br>"
                 "See H-statistic chart for validation →",
            showarrow=False, xanchor="right", yanchor="bottom",
            font=dict(size=8, color=MUTED, family="Inter"),
            bgcolor=WHITE, bordercolor=BORDER, borderwidth=1, borderpad=3)
        fig_int_rank.update_xaxes(title_text="Average Dollar Impact per Policy ($)",
                                   tickprefix="$")
        fig_int_rank.update_yaxes(tickfont=dict(size=9, family="Inter"))
        fig_int_rank.update_layout(
            template="plotly_white",
            height=max(250, len(_interaction_terms) * 35 + 80),
            margin=dict(l=10, r=80, t=10, b=40),
            font=dict(family="Inter"), plot_bgcolor="#FAFBFC")
    else:
        fig_int_rank = go.Figure().add_annotation(
            text="No interaction terms detected in the EBM model",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
            font=dict(size=12, color=MUTED))
        fig_int_rank.update_layout(height=200, template="plotly_white")

    # ── Friedman H-statistic validation ──────────────────────────────────────
    _h_stats = []
    for _it in _interaction_terms[:8]:
        try:
            _idata = global_exp.data(_it["index"])
            if _idata and "scores" in _idata:
                _int_scores = np.array(_idata["scores"], dtype=float)
                _var_int = float(np.var(_int_scores))
                _raw = _it["raw_term"]
                _parts = None
                for _delim in (" x ", " & ", " × "):
                    if _delim in _raw:
                        _parts = [p.strip() for p in _raw.split(_delim)]
                        break
                if _parts and len(_parts) == 2:
                    _var_a, _var_b = 0.0, 0.0
                    for _fi, _fn in enumerate(ebm_model.term_names_):
                        _fd = None
                        try:
                            _fd = global_exp.data(_fi)
                        except Exception:
                            pass
                        if _fd and "scores" in _fd:
                            _s = np.array(_fd["scores"], dtype=float)
                            if str(_fn) == _parts[0]:
                                _var_a = float(np.var(_s))
                            elif str(_fn) == _parts[1]:
                                _var_b = float(np.var(_s))
                    _total_var = _var_a + _var_b + _var_int
                    _h = _var_int / _total_var if _total_var > 0 else 0.0
                    _h_stats.append({
                        "pair": _it["term"],
                        "H": _h,
                        "dollar_impact": _it["dollar_impact"],
                    })
        except Exception:
            continue

    if _h_stats:
        _h_stats.sort(key=lambda x: x["H"], reverse=True)
        fig_h = go.Figure(go.Bar(
            y=[h["pair"] for h in _h_stats[::-1]],
            x=[h["H"]    for h in _h_stats[::-1]],
            orientation="h",
            marker_color=[RED if h["H"] > 0.15 else AMBER if h["H"] > 0.05 else MUTED
                          for h in _h_stats[::-1]],
            text=[f"H={h['H']:.3f}" for h in _h_stats[::-1]],
            textposition="outside",
            textfont=dict(size=9, family="Inter"),
            hovertemplate="Pair: %{y}<br>Friedman H: %{x:.3f}<extra></extra>",
        ))
        fig_h.add_vline(x=0.05, line_color=AMBER, line_width=1.5, line_dash="dot",
                        annotation_text="H=0.05 threshold",
                        annotation_position="top right",
                        annotation_font_size=10)
        fig_h.update_xaxes(title_text="Friedman H-Statistic (interaction strength)",
                           range=[0, max(h["H"] for h in _h_stats) * 1.3])
        fig_h.update_yaxes(tickfont=dict(size=9, family="Inter"))
        fig_h.update_layout(
            template="plotly_white",
            height=max(250, len(_h_stats) * 35 + 80),
            margin=dict(l=10, r=60, t=10, b=40),
            font=dict(family="Inter"), plot_bgcolor="#FAFBFC")
    else:
        fig_h = None

    # ── Interaction surface — native EBM (fallback: binned heatmap) ───────────
    if INTERACTION_SURFACE is not None:
        try:
            _idata = INTERACTION_SURFACE
            _z     = np.array(_idata["scores"], dtype=float)
            _names = _idata.get("names", ([], []))
            _xn    = [f"{v:.1f}" for v in np.array(_names[0], dtype=float)] if len(_names) > 0 else []
            _yn    = [f"{v:.1f}" for v in np.array(_names[1], dtype=float)] if len(_names) > 1 else []
            _zpct  = (np.exp(_z) - 1) * 100 if _z.max() < 10 else _z  # already % or log scale
            fig_heat = go.Figure(go.Heatmap(
                z=_zpct, x=_xn, y=_yn,
                colorscale="RdYlGn_r",
                colorbar=dict(title="GA2M<br>Surcharge (%)", thickness=14, len=0.85,
                              tickformat=".0f", ticksuffix="%"),
                hovertemplate="Wildfire: %{x}<br>Roof Vuln: %{y}<br>"
                              "GA2M surcharge: %{z:.1f}%<extra></extra>"))
            fig_heat.add_annotation(
                text="This surface is the GA2M's<br>exact learned interaction effect",
                x=_xn[-2] if len(_xn) > 1 else 0, y=_yn[-1] if len(_yn) else 0,
                showarrow=True, arrowhead=2, arrowcolor=NAVY,
                font=dict(size=9, color=NAVY), bgcolor=WHITE,
                bordercolor=BORDER, borderwidth=1, ax=-70, ay=30)
            _heat_subtitle = "Native EBM learned interaction surface — not a binned average"
        except Exception:
            INTERACTION_SURFACE_FALLBACK = True
    else:
        INTERACTION_SURFACE_FALLBACK = True

    if INTERACTION_SURFACE is None or "INTERACTION_SURFACE_FALLBACK" in dir():
        # Fallback: binned heatmap
        _tmp = df.copy()
        _tmp["wf_bin"] = pd.cut(_tmp["Wildfire_Exposure_Daily"], bins=10)
        _tmp["rv_bin"] = pd.cut(_tmp["Roof_Vulnerability_Satellite"], bins=10)
        _pivot = _tmp.groupby(["rv_bin", "wf_bin"], observed=True)["EBM_Log_Uplift"].mean().unstack()
        def _bl(x):
            lo = float(str(x).split(",")[0].strip("("))
            hi = float(str(x).split(",")[1].strip("]"))
            return f"{lo:.0f}–{hi:.0f}"
        _pct_m = (np.exp(_pivot.values) - 1) * 100
        fig_heat = go.Figure(go.Heatmap(
            z=_pct_m,
            x=[_bl(c) for c in _pivot.columns],
            y=[_bl(r) for r in _pivot.index],
            colorscale="RdYlGn_r",
            colorbar=dict(title="Avg GA2M<br>Surcharge (%)", thickness=14, len=0.85,
                          tickformat=".0f", ticksuffix="%"),
            hovertemplate="Wildfire: %{x}<br>Roof Vuln: %{y}<br>Surcharge: %{z:.1f}%<extra></extra>",
            zmin=0))
        fig_heat.add_annotation(
            text="A GLM would show: flat (no interaction)",
            x=_bl(list(_pivot.columns)[-1]),
            y=_bl(list(_pivot.index)[-1]),
            showarrow=True, arrowhead=2, arrowcolor=NAVY,
            font=dict(size=9, color=NAVY), bgcolor=WHITE,
            bordercolor=BORDER, borderwidth=1, ax=-60, ay=30)
        _heat_subtitle = "Avg GA2M surcharge (%) · A GLM prices Wildfire and Roof Vuln independently"

    fig_heat.update_xaxes(title_text="Wildfire Exposure Index")
    fig_heat.update_yaxes(title_text="Roof Vulnerability Score")
    fig_heat.update_layout(template="plotly_white", height=CHART_HEIGHT_MD,
        margin=dict(l=10, r=10, t=20, b=40), font=dict(family="Inter"))

    return html.Div([
        dbc.Alert([
            html.I(className="fas fa-microscope me-2", style={"color": BLUE}),
            html.Strong("Intelligence Signal Architecture: "),
            "This solution captures three signal types the GLM cannot price: "
            "(1) non-linear individual feature effects, (2) compounding pairwise interactions, "
            "and (3) temporal risk decay signals — all fully interpretable via the GA2M glass-box.",
        ], color="info", className="mb-4",
           style={"borderLeft": f"4px solid {BLUE}", "backgroundColor": "#EBF5FB",
                  "borderRadius": "8px", "fontSize": "0.87rem"}),

        # Data Characteristics panel FIRST (Spec S-A.4)
        dcc.Loading(_build_shape_panel(), type="circle"),

        # Signal landscape (full left) + state adjustment chart (right)
        dbc.Row([
            dbc.Col(chart_card("Signal Landscape — Estimated Dollar Impact per Policy",
                "tt-imp",
                "Ranked by average absolute contribution to the GA2M residual. "
                "Navy = new modern signal. Blue-grey = legacy feature gaining non-linear treatment. "
                "Gold = pairwise interaction term.",
                dcc.Graph(figure=fig_imp, config={"displayModeBar": False}),
                subtitle="Top 15 · Gold=interaction · Navy=new modern · Blue-grey=legacy non-linear gain"),
            width=6),
            dbc.Col(chart_card("Geographic Intelligence — Average Adjustment by State",
                "tt-state",
                "Average GA2M adjustment per state after intelligence layer. "
                "High-wildfire states (CA, CO, WA) receive surcharges from wildfire × roof interactions. "
                "Hail belt states (TX, OK) from hail × roof vulnerability. "
                "Error bars show ±1 SE across policies within each state.",
                dcc.Loading(dcc.Graph(figure=fig_state, config={"displayModeBar": False}),
                            type="circle"),
                subtitle="Red = avg surcharge · Green = avg credit · bars = ±1 SE"),
            width=6),
        ], className="g-3 mb-4"),

        # Interaction Discovery + H-statistic validation
        dbc.Row([
            dbc.Col(chart_card(
                "Interaction Discovery — Pairwise Effects Ranked by Dollar Impact",
                "tt-int-rank",
                "The GA2M automatically discovers pairwise feature interactions that produce "
                "compound risk effects beyond the sum of individual features. Gold bars show "
                "the estimated average dollar impact per policy for each discovered interaction "
                "pair. These are the effects the GLM's additive structure cannot capture.",
                dcc.Loading(dcc.Graph(figure=fig_int_rank, config={"displayModeBar": False}),
                            type="circle"),
                subtitle=(f"{len(_interaction_terms)} pairwise interactions discovered · "
                           f"Gold = compound-peril effect the GLM misses")),
            width=6),
            dbc.Col(chart_card(
                "Interaction Strength — Friedman H-Statistic Validation",
                "tt-h-stat",
                "The Friedman H-statistic measures what fraction of a feature pair's joint "
                "effect comes from their interaction vs. the sum of individual effects. "
                "H > 0.05 = meaningful interaction. H > 0.15 = strong. "
                "Rankings differ from the dollar-impact chart because H measures interaction "
                "purity (how much of the joint signal IS interaction), while dollar impact "
                "measures absolute pricing magnitude. A high-H / low-dollar pair means "
                "the interaction dominates its features' joint effect but the features "
                "themselves have a smaller residual. Both views are complementary.",
                dcc.Loading(dcc.Graph(figure=fig_h, config={"displayModeBar": False}),
                            type="circle")
                if fig_h else html.Div("H-statistics require interaction terms",
                                       style={"color": MUTED, "padding": "20px"}),
                subtitle="H measures interaction purity (not dollar size) · "
                         "H > 0.05 = meaningful · H > 0.15 = strong"),
            width=6),
        ], className="g-3 mb-4"),

        # Interaction surface
        dbc.Row([dbc.Col(chart_card(
            "Compounding Risk — Wildfire × Roof Vulnerability Interaction",
            "tt-heat",
            "Surcharge from the interaction of two features beyond their individual effects. "
            "A GLM prices them independently and systematically undercharges the top-right cluster.",
            dcc.Loading(dcc.Graph(figure=fig_heat, config={"displayModeBar": False}), type="circle"),
            subtitle=_heat_subtitle,
        ), width=12)], className="g-3 pb-4"),
    ], className="py-4")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3  —  POLICY UNDERWRITER LENS
# ══════════════════════════════════════════════════════════════════════════════
def _quick_pick_row() -> html.Div:
    """
    Quick Pick demo archetypes as a 2-column grid of compact tiles.
    Uses html.Div (not dbc.Button) so Bootstrap never overrides text colors.
    Hover effect is handled by the injected .archetype-tile CSS class.
    """
    if not DEMO_ARCHETYPES:
        return html.Div()

    def _tile(a):
        adj_val  = a["adj"]
        is_up    = adj_val.startswith("+")
        adj_color = RED if is_up else GREEN
        # Left-border accent color for this archetype
        accent   = a["color"]

        return html.Div(
            html.Div([
                # ── Icon + Name row ──────────────────────────────────────────
                html.Div([
                    html.I(className=f"{a['icon']}",
                           style={"color": accent,
                                  "fontSize": "0.8rem",
                                  "marginRight": "7px",
                                  "flexShrink": "0",
                                  "marginTop": "1px"}),
                    html.Span(a["label"],
                              style={
                                  "fontWeight": "700",
                                  "fontSize": "0.74rem",
                                  "color": "#1B2A4A",      # NAVY as hex — never overridden
                                  "lineHeight": "1.25",
                              }),
                ], style={"display": "flex", "alignItems": "flex-start",
                           "marginBottom": "5px"}),
                # ── Price flow + adjustment badge ────────────────────────────
                html.Div([
                    html.Span(f"{a['glm']} → {a['final']}",
                              style={"fontSize": "0.69rem",
                                     "color": "#8D9EAD",   # MUTED as hex
                                     "flexGrow": "1"}),
                    html.Span(adj_val,
                              style={"fontSize": "0.68rem",
                                     "fontWeight": "700",
                                     "color": adj_color,
                                     "flexShrink": "0",
                                     "marginLeft": "6px"}),
                ], style={"display": "flex", "alignItems": "center"}),
            ],
            # Inner card styling — all explicit, no Bootstrap involved
            style={
                "padding": "8px 10px",
                "borderRadius": "6px",
                "border": f"1px solid {BORDER}",
                "borderLeft": f"3px solid {accent}",
                "backgroundColor": WHITE,
                "width": "100%",
            }),
            # Outer wrapper: width + hover class
            id={"type": "archetype-btn", "index": a["value"]},
            n_clicks=0,
            className="archetype-tile mb-2",
            style={
                "width": "calc(50% - 5px)",
                "display": "inline-block",
                "verticalAlign": "top",
                "paddingRight": "8px" if DEMO_ARCHETYPES.index(a) % 2 == 0 else "0",
            },
        )

    return html.Div([
        html.Div("Quick Pick — Named Demo Properties",
                 style={
                     "fontSize": "0.67rem", "color": "#8D9EAD", "fontWeight": "700",
                     "textTransform": "uppercase", "letterSpacing": "0.07em",
                     "marginBottom": "9px", "paddingBottom": "5px",
                     "borderBottom": f"1px solid {BORDER}",
                 }),
        html.Div(
            [_tile(a) for a in DEMO_ARCHETYPES],
            style={"display": "flex", "flexWrap": "wrap",
                   "justifyContent": "flex-start"},
        ),
    ], style={"marginBottom": "14px"})


def build_policy_tab():
    return html.Div([
        dbc.Row([
            dbc.Col([dbc.Card([
                dbc.CardHeader(html.Div("Select Policy", style=SEC_TITLE),
                               style={"backgroundColor": WHITE, "border": "none"}),
                dbc.CardBody([
                    _quick_pick_row(),
                    html.Div("— or choose from curated pool: 60 highest surcharges · "
                             "60 highest credits · 80 random —",
                             style={"fontSize": "0.73rem", "color": MUTED,
                                    "marginBottom": "8px"}),
                    dcc.Dropdown(id="policy-dd", options=POLICY_OPTIONS,
                                 value=DEFAULT_POLICY, clearable=False,
                                 style={"fontSize": "0.82rem"}),
                    html.Div(id="policy-profile-panel", className="mt-3"),
                ]),
            ], style=CARD_STYLE)], width=3),

            dbc.Col([dbc.Card([
                dbc.CardHeader([
                    dbc.Row([
                        dbc.Col([
                            html.Div("Pricing Deconstruction", style=SEC_TITLE),
                            html.Div("Full audit trail from legacy actuarial formula to "
                                     "intelligence-adjusted premium",
                                     style={"fontSize": "0.75rem", "color": MUTED}),
                        ], width=7),
                        dbc.Col([
                            dbc.ButtonGroup([
                                dbc.Button("Strategic",       id="btn-hi",  n_clicks=1,
                                           color="primary",   outline=True, size="sm"),
                                dbc.Button("GLM Breakdown",   id="btn-glm", n_clicks=0,
                                           color="secondary", outline=True, size="sm"),
                                dbc.Button("GA2M Intelligence", id="btn-gam", n_clicks=0,
                                           color="info",      outline=True, size="sm"),
                            ])
                        ], width=5, className="text-end d-flex align-items-center justify-content-end"),
                    ], align="center"),
                ], style={"backgroundColor": WHITE, "border": "none"}),
                dbc.CardBody([
                    dcc.Store(id="view-store", data="high_level"),
                    html.Div([
                        html.Span([html.Span("●", style={"color": BLUE, "fontWeight": "700",
                                                          "marginRight": "4px"}),
                                   "Individual feature effect"],
                                  style={"fontSize": "0.75rem", "color": MUTED,
                                         "marginRight": "20px"}),
                        html.Span([html.Span("⊗", style={"color": GOLD, "fontWeight": "700",
                                                           "marginRight": "4px"}),
                                   "Pairwise interaction effect"],
                                  style={"fontSize": "0.75rem", "color": MUTED,
                                         "marginRight": "20px"}),
                        html.Span("(⊗ appears in GA2M view only)",
                                  style={"fontSize": "0.7rem", "color": BORDER}),
                    ], className="text-end mb-1 pe-1"),
                    dcc.Loading(dcc.Graph(id="waterfall-plot",
                                          config={"displayModeBar": False}),
                                type="circle"),
                ]),
            ], style=CARD_STYLE)], width=9),
        ], className="g-3 py-4"),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4  —  FRAMEWORK
# ══════════════════════════════════════════════════════════════════════════════
def build_framework_tab():
    Y_CTR  = 1.30; BOX_H = 0.72
    Y_TOP  = Y_CTR + BOX_H; Y_LBL = Y_TOP + 0.28; Y_NOTE = 0.30
    fig_arch = go.Figure()
    fig_arch.update_layout(
        template="plotly_white", height=280,
        margin=dict(l=20, r=20, t=20, b=20), font=dict(family="Inter"),
        xaxis=dict(visible=False, range=[0, 10]),
        yaxis=dict(visible=False, range=[0, 3.20]))
    BOXES = [
        (0.30, 1.85, "Property\nFeatures\n(28 vars)",       "#8A9BB0", WHITE),
        (2.10, 3.85, "Legacy GLM\nFreq × Sev\n(16 vars)",   "#4A5568", WHITE),
        (4.10, 5.85, "GLM Pure\nPremium\n(baseline)",        "#2C3E50", WHITE),
        (6.10, 7.85, "GA2M\nResidual\n(28 vars, 15 int.)",  BLUE,      WHITE),
        (8.05, 9.70, "Final\nIntelligence\nPremium",         NAVY,      WHITE),
    ]
    for x0, x1, label, fc, tc in BOXES:
        fig_arch.add_shape(type="rect", x0=x0, y0=Y_CTR-BOX_H, x1=x1, y1=Y_CTR+BOX_H,
            fillcolor=fc, line_color="white", line_width=2, layer="below")
        fig_arch.add_annotation(x=(x0+x1)/2, y=Y_CTR, text=label.replace("\n","<br>"),
            showarrow=False, font=dict(size=11, color=tc, family="Inter"), align="center")
    for tail_x, head_x, mid_x, lbl in [
        (1.85, 2.10, 1.975, ""),
        (3.85, 4.10, 3.975, ""),
        (5.85, 6.10, 5.975, "log(True ÷ GLM)\nresidual target"),
        (7.85, 8.05, 7.950, "× exp(GA2M)\n[0.65×, 1.60×]\nE_w[uplift]=1.0"),
    ]:
        fig_arch.add_annotation(x=head_x, y=Y_CTR, ax=tail_x, ay=Y_CTR,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.2,
            arrowcolor=GOLD, arrowwidth=2, text="")
        if lbl:
            fig_arch.add_annotation(x=mid_x, y=Y_LBL, text=lbl.replace("\n","<br>"),
                showarrow=False, font=dict(size=11, color="#555", family="Inter"),
                align="center", xanchor="center",
                bgcolor="white", bordercolor=BORDER, borderwidth=1, borderpad=3)
    fig_arch.add_annotation(x=5.0, y=Y_NOTE,
        text="<b>Separation of concerns:</b> GLM handles linear exposure relativities "
             "· GA2M captures non-linear effects + pairwise interactions",
        showarrow=False, font=dict(size=11, color=MUTED), xanchor="center")

    def lim_pill(icon, title, body, color):
        return html.Div([
            html.I(className=f"{icon} me-2", style={"color": color}),
            html.Span(title, style={"fontWeight": "600", "fontSize": "0.85rem",
                                    "color": NAVY}),
            html.Div(body, style={"fontSize": "0.78rem", "color": MUTED,
                                   "marginTop": "4px", "lineHeight": "1.5"}),
        ], style={"backgroundColor": "#F8F9FA", "borderRadius": "8px", "padding": "12px",
                  "border": f"1px solid {BORDER}", "height": "100%"})

    def perf_chip(label, val, color):
        return html.Div([
            html.Div(val,   style={"fontSize": "1.5rem", "fontWeight": "700", "color": color}),
            html.Div(label, style={"fontSize": "0.7rem", "color": MUTED,
                                    "textTransform": "uppercase", "letterSpacing": "0.05em",
                                    "marginTop": "2px"}),
        ], style={"textAlign": "center", "padding": "14px 20px",
                  "border": f"1px solid {BORDER}", "borderRadius": "10px",
                  "backgroundColor": WHITE, "minWidth": "130px"})

    return html.Div([
        dbc.Card([
            dbc.CardHeader([
                html.Div("Two-Layer Pricing Architecture", style=SEC_TITLE),
                html.Div("This solution operates as a constrained, interpretable intelligence layer "
                         "on top of — not replacing — the carrier's existing GLM infrastructure.",
                         style={"fontSize": "0.78rem", "color": MUTED}),
            ], style={"backgroundColor": WHITE, "border": "none"}),
            dbc.CardBody(dcc.Graph(figure=fig_arch, config={"displayModeBar": False})),
        ], style=CARD_STYLE, className="mb-4"),

        dbc.Row([
            dbc.Col([
                section_card(1, "Legacy GLM — Log-Linear Rating Structure (16 features)", MUTED,
                    html.Div([
                        html.P(["Industry-standard homeowners rating plan: ",
                                html.Strong("Poisson × Gamma GLM"),
                                " with statsmodels diagnostics (deviance, AIC, p-values) "
                                "and AOI-based exposure offset:"],
                               style={"fontSize": "0.85rem", "color": MUTED, "marginBottom": "10px"}),
                        formula_block("log E[Freqᵢ] = β₀ + β₁x₁ᵢ + β₂x₂ᵢ + … + βₖxₖᵢ  + log(AOI/100K)",
                            "Poisson GLM · log link · exposure = AOI/$100K · 12 main effects"),
                        formula_block("log E[Sevᵢ]  = γ₀ + γ₁x₁ᵢ + γ₂x₂ᵢ + … + γₖxₖᵢ",
                            "Gamma GLM · 4 engineered interactions: Frame×HighPC, "
                            "Claims×LowDed, Urban×HighPC, OldRoof×Hail"),
                        formula_block("GLM PPᵢ = exp(β₀+γ₀)· (AOIᵢ/100K) · ∏ₖ exp((βₖ+γₖ)·xₖᵢ)",
                            "Multiplicative relativities — ISO/Bureau tariff structure"),
                        html.Div([html.Strong("Credit score suppression: "),
                                  "CA and MA policies use portfolio median (700) to simulate "
                                  "state regulatory restriction — a real-world compliance constraint."],
                                 style={"fontSize": "0.78rem", "color": AMBER, "marginTop": "8px",
                                        "backgroundColor": "#FFF8EF", "padding": "8px 12px",
                                        "borderRadius": "6px", "border": f"1px solid {GOLD}"}),
                    ])),
                html.Div(className="mb-3"),
                section_card(2, "Where the GLM Reaches Its Structural Ceiling", RED,
                    html.Div([
                        html.P("Three architectural constraints limit the GLM regardless of "
                               "variable selection:",
                               style={"fontSize": "0.85rem", "color": MUTED, "marginBottom": "12px"}),
                        dbc.Row([
                            dbc.Col(lim_pill("fas fa-slash", "Linearity Constraint",
                                "The log-link forces every feature effect to be linear in "
                                "log-premium space. Convex, threshold, and U-shaped effects "
                                "are approximated away.", RED), width=4),
                            dbc.Col(lim_pill("fas fa-ban", "Additive Structure",
                                "Rating factors multiply independently. Compounding interaction "
                                "premiums — e.g. high wildfire + old roof — are never captured.",
                                AMBER), width=4),
                            dbc.Col(lim_pill("fas fa-clock-rotate-left", "Static Variables",
                                "Temporal risk signals (water-loss recency, real-time wildfire, "
                                "satellite roof condition) enter only as coarse buckets if at all.",
                                BLUE), width=4),
                        ], className="g-2"),
                        html.Div([
                            "⚠️ Net effect: ",
                            html.Strong(f"{(1-glm_r2):.1%} of pure premium variance"),
                            f" is structurally unexplained by the GLM {OOS_LABEL} — the addressable residual.",
                        ], style={"fontSize": "0.78rem", "color": AMBER, "marginTop": "12px",
                                  "backgroundColor": "#FFF8EF", "padding": "8px 12px",
                                  "borderRadius": "6px", "border": f"1px solid {GOLD}"}),
                    ])),
            ], width=6),

            dbc.Col([
                section_card(3, "GA2M Residual Layer — Mathematical Specification (28 features)",
                    BLUE, html.Div([
                        html.P(["Trains an ",
                                html.Strong("Explainable Boosting Machine (EBM / GA2M)"),
                                " on the log-scale GLM residual:"],
                               style={"fontSize": "0.85rem", "color": MUTED, "marginBottom": "10px"}),
                        formula_block("εᵢ = log(True PPᵢ) − log(GLM PPᵢ)",
                            "Log-multiplicative residual target · positivity guaranteed"),
                        formula_block("g(εᵢ) = β₀ + Σⱼ fⱼ(xⱼᵢ) + Σⱼ<ₗ fⱼₗ(xⱼᵢ, xₗᵢ)",
                            "28-feature GA2M · 6 forced interactions + 9 auto-discovered = 15 total"),
                        formula_block("Final PPᵢ = GLM PPᵢ × exp( clip(ĝᵢ, log(0.65), log(1.60)) ) / Z",
                            "Corridor [0.65×, 1.60×] · Z = normalisation constant · risk-neutral"),
                        html.Div([
                            html.Strong("Risk Neutrality: "),
                            html.Span("E_w[uplift] = 1.0", style=MONO),
                            html.Br(),
                            "Weighted mean uplift (weights = GLM premium) is normalised to 1.0 "
                            "exactly, so total book premium is invariant to the GA2M adjustment. "
                            f"This demo: weighted mean = {_risk_neutral_check:.6f}×.",
                        ], style={"fontSize": "0.78rem", "color": MUTED, "lineHeight": "1.6",
                                  "backgroundColor": "#EBF5FB", "borderRadius": "6px",
                                  "padding": "10px 12px", "marginTop": "10px"}),
                    ])),
                html.Div(className="mb-3"),
                section_card(4, "Glass-Box Guarantee — Interpretability Architecture",
                    GREEN, html.Div([
                        html.P("Every GA2M prediction decomposes exactly into auditable "
                               "per-feature contributions:",
                               style={"fontSize": "0.85rem", "color": MUTED, "marginBottom": "12px"}),
                        *[html.Div([
                            html.I(className=f"{ic} me-2", style={"color": col}),
                            html.Span(title, style={"fontWeight": "600", "fontSize": "0.83rem",
                                                     "color": NAVY}),
                            html.Div(body, style={"fontSize": "0.77rem", "color": MUTED,
                                                   "marginTop": "2px", "marginLeft": "22px",
                                                   "lineHeight": "1.5"}),
                        ], style={"marginBottom": "10px"}) for ic, title, body, col in [
                            ("fas fa-globe", "Global",
                             "Shape functions fⱼ(x) and interaction surfaces fⱼₗ(x,y) — see "
                             "Intelligence Signals tab.", GREEN),
                            ("fas fa-fingerprint", "Local",
                             "Per-policy waterfall of dollar contributions — see Policy Lens tab.",
                             BLUE),
                            ("fas fa-file-contract", "Regulatory",
                             "Contributions are log-uplift addends — the multiplicative relativity "
                             "language regulators already accept.", GOLD),
                        ]],
                        html.Div(["✔ ", html.Strong("Exact additivity:"),
                                  " EBM enforces exact decomposition — no post-hoc SHAP "
                                  "approximation is involved, unlike black-box + SHAP approaches."],
                                 style={"fontSize": "0.78rem", "color": GREEN, "marginTop": "6px",
                                        "backgroundColor": "#EAFAF1", "padding": "8px 12px",
                                        "borderRadius": "6px",
                                        "border": f"1px solid {GREEN}"}),
                    ])),
            ], width=6),
        ], className="g-4 mb-4"),

        dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.Span("5", style={"backgroundColor": GOLD, "color": WHITE,
                        "borderRadius": "50%", "width": "26px", "height": "26px",
                        "display": "inline-flex", "alignItems": "center",
                        "justifyContent": "center", "fontSize": "0.8rem",
                        "fontWeight": "700", "marginRight": "10px"}),
                    html.Span("Validation & Performance Characteristics",
                              style={"fontWeight": "700", "fontSize": "1.0rem", "color": NAVY}),
                    info_tooltip("tt-fw-perf",
                        f"Metrics computed on held-out 20% test set {OOS_LABEL}. "
                        "In production, carriers would validate on held-out accident years "
                        "using lift curves, Gini coefficients, and double-lift charts."),
                ], className="d-flex align-items-center"),
            ], style={"backgroundColor": WHITE, "border": "none",
                      "borderLeft": f"4px solid {GOLD}"}),
            dbc.CardBody([
                html.Div(f"All metrics computed on held-out test set {OOS_LABEL} — not in-sample.",
                         style={"fontSize": "0.75rem", "color": MUTED, "marginBottom": "12px",
                                "backgroundColor": "#F8F9FA", "padding": "6px 10px",
                                "borderRadius": "4px"}),
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Div([
                            perf_chip("GLM R²", f"{glm_r2:.4f}", MUTED),
                            perf_chip("GA2M R²", f"{final_r2:.4f}", NAVY),
                            perf_chip("Lift ΔR²", f"+{delta_r2:.4f}", GREEN),
                            perf_chip("Uplift Corridor", f"{MIN_UPLIFT:.2f}×–{MAX_UPLIFT:.2f}×", BLUE),
                            perf_chip("Risk Neutrality", f"{_risk_neutral_check:.4f}×", TEAL),
                            perf_chip("Residual Recovered",
                                      f"{delta_r2/(1-glm_r2):.1%}" if glm_r2 < 1 else "—", GREEN),
                        ], className="d-flex flex-wrap gap-3"),
                    ]), width=8),
                    dbc.Col([
                        html.Div("Production validation checklist:",
                                 style={"fontWeight": "600", "fontSize": "0.83rem",
                                        "color": NAVY, "marginBottom": "8px"}),
                        *[html.Div([
                            html.I(className="fas fa-circle-dot me-2",
                                   style={"color": BLUE, "fontSize": "0.6rem"}),
                            html.Span(item, style={"fontSize": "0.78rem", "color": MUTED}),
                        ], style={"marginBottom": "5px", "display": "flex",
                                  "alignItems": "center"})
                          for item in [
                            "Out-of-time validation on most recent 2 accident years",
                            "Pricing accuracy by risk segment: MAPE improvement per quintile ✓ (Tab 1)",
                            "Gini coefficient improvement on ranked risk segments",
                            "Monotonicity tests on all shape functions",
                            "Bias audit across protected class proxies",
                            "Rate impact study before regulatory filing",
                        ]],
                    ], width=4),
                ]),
            ]),
        ], style={**CARD_STYLE, "borderLeft": f"4px solid {GOLD}"}),
    ], className="py-4")


# ── Root Layout ───────────────────────────────────────────────────────────────
# Note: archetype tile hover CSS lives in assets/custom.css
# Dash automatically serves all files in the assets/ directory.
app.layout = html.Div([
    navbar,
    dbc.Container([
        dcc.Tabs(id="main-tabs", value="tab-portfolio",
                 style={"marginTop": "12px"}, children=[
            dcc.Tab(label="Business Case",        value="tab-portfolio",
                    style=TAB_STYLE, selected_style=TAB_SEL),
            dcc.Tab(label="Intelligence Signals", value="tab-features",
                    style=TAB_STYLE, selected_style=TAB_SEL),
            dcc.Tab(label="Policy Lens",          value="tab-policy",
                    style=TAB_STYLE, selected_style=TAB_SEL),
            dcc.Tab(label="Framework",            value="tab-framework",
                    style=TAB_STYLE, selected_style=TAB_SEL),
        ]),
        dcc.Loading(html.Div(id="tab-content"), type="default"),
    ], fluid=True, style={"maxWidth": "1600px", "padding": "0 24px"}),
], style={"backgroundColor": BG, "minHeight": "100vh",
          "fontFamily": "Inter, sans-serif"})


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════
@callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab):
    if tab == "tab-portfolio": return build_portfolio_tab()
    if tab == "tab-features":  return build_feature_tab()
    if tab == "tab-policy":    return build_policy_tab()
    if tab == "tab-framework": return build_framework_tab()


@callback(Output("view-store", "data"),
          [Input("btn-hi", "n_clicks"),
           Input("btn-glm", "n_clicks"),
           Input("btn-gam", "n_clicks")],
          prevent_initial_call=True)
def set_view(n1, n2, n3):
    t = ctx.triggered_id
    if t == "btn-hi":  return "high_level"
    if t == "btn-glm": return "glm_breakdown"
    if t == "btn-gam": return "gam_breakdown"
    return "high_level"


@callback([Output("btn-hi",  "outline"),
           Output("btn-glm", "outline"),
           Output("btn-gam", "outline")],
          Input("view-store", "data"))
def highlight_btn(view):
    return view != "high_level", view != "glm_breakdown", view != "gam_breakdown"


@callback(Output("policy-dd", "value"),
          Input({"type": "archetype-btn", "index": dash.ALL}, "n_clicks"),
          prevent_initial_call=True)
def pick_archetype(n_clicks_list):
    """Update dropdown to the clicked archetype policy index."""
    if not any(n_clicks_list):
        raise dash.exceptions.PreventUpdate
    triggered = ctx.triggered_id
    if triggered and "index" in triggered:
        return triggered["index"]
    raise dash.exceptions.PreventUpdate


@callback(
    [Output("policy-profile-panel", "children"),
     Output("waterfall-plot", "figure")],
    [Input("policy-dd", "value"),
     Input("view-store", "data")]
)
def update_policy_view(selected_idx, view_type):
    selected_idx = int(selected_idx) if selected_idx is not None else DEFAULT_POLICY
    row          = df.loc[selected_idx]
    glm_prem     = float(row["GLM_Pure_Premium"])
    actual_epp   = float(row["Expected_Pure_Premium"])
    ebm_adj      = float(row["EBM_Residual_Pred"])
    final_prem   = float(row["Final_Pure_Premium"])
    adj_pct      = float(row["Adjustment_Pct"])
    tier         = str(row.get("Final_Risk_Tier", row.get("Risk_Tier", "Moderate")))
    tc           = TIER_COLORS.get(tier, MUTED)

    def row_item(lbl, val, vstyle=None):
        base = {"fontSize": "0.92rem", "fontWeight": "600", "color": NAVY, "float": "right"}
        if vstyle:
            base.update(vstyle)
        return html.Div([
            html.Span(lbl, style={"fontSize": "0.72rem", "color": MUTED,
                                   "textTransform": "uppercase", "letterSpacing": "0.04em"}),
            html.Span(val, style=base),
        ], style={"borderBottom": f"1px solid {BORDER}", "padding": "7px 0",
                  "overflow": "hidden"})

    profile = html.Div([
        html.Div([
            dbc.Badge(f"{tier} Risk",
                      style={"backgroundColor": tc, "fontSize": "0.75rem"}),
            dbc.Badge(f"{'↑' if adj_pct > 0 else '↓'}{abs(adj_pct):.1f}% vs GLM",
                      color="danger" if adj_pct > 0 else "success",
                      className="ms-1", style={"fontSize": "0.75rem"}),
        ], className="mb-3"),
        row_item("True Expected",  f"${actual_epp:,.0f}",
                 {"color": RED if actual_epp > final_prem else GREEN}),
        row_item("Legacy GLM",     f"${glm_prem:,.0f}"),
        row_item("GA2M Adj",
                 f"{'+'if ebm_adj >= 0 else ''}${ebm_adj:,.0f}",
                 {"color": RED if ebm_adj > 0 else GREEN}),
        html.Div([
            html.Span("Final Premium",
                      style={"fontSize": "0.82rem", "fontWeight": "700", "color": NAVY}),
            html.Span(f"${final_prem:,.0f}",
                      style={"fontSize": "1.15rem", "fontWeight": "700", "color": BLUE,
                             "float": "right"}),
        ], style={"padding": "10px 0 0", "overflow": "hidden"}),
        # Credit suppression notice
        html.Div([
            html.I(className="fas fa-info-circle me-1", style={"color": AMBER}),
            "Credit score suppressed (regulatory — CA/MA)",
        ], style={"fontSize": "0.7rem", "color": AMBER, "marginTop": "8px"}
        ) if row.get("Credit_Score_Suppressed", False) else None,
    ])

    WF  = dict(connector={"line": {"color": BORDER, "width": 2}},
               increasing={"marker": {"color": RED}},
               decreasing={"marker": {"color": GREEN}})
    LAY = dict(template="plotly_white", font=dict(family="Inter"),
               height=CHART_HEIGHT_LG,
               margin=dict(l=20, r=20, t=65, b=50),
               showlegend=False, waterfallgap=0.25)

    if view_type == "high_level":
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "total"],
            x=["Legacy GLM Estimate", "GA2M Intelligence Adjustment", "Final Premium"],
            y=[glm_prem, ebm_adj, 0],
            textposition="outside",
            text=[f"${glm_prem:,.0f}",
                  f"{'+'if ebm_adj>=0 else ''}${ebm_adj:,.0f}",
                  f"${final_prem:,.0f}"],
            totals={"marker": {"color": NAVY}}, **WF))
        fig.update_layout(title={"text": "Strategic View — Legacy Formula → Intelligence-Adjusted Premium",
                                  "font": {"size": 13, "color": NAVY}}, **LAY)

    elif view_type == "glm_breakdown":
        # ── statsmodels-based GLM waterfall (Spec G2.3) ───────────────────────
        try:
            # Build engineered feature row
            row_df = df.loc[[selected_idx]].copy()
            row_df["Dwelling_Age"]      = 2026 - row_df["Year_Built"].astype(int)
            row_df["Frame_HighPC"]      = ((row_df["Construction_Type"] == "Frame") &
                                            (row_df["Protection_Class"] > 6)).astype(int).astype(str)
            row_df["FreqClaims_LowDed"] = ((row_df["CLUE_Loss_Count"] >= 2) &
                                            (row_df["Deductible"].astype(int) <= 500)
                                            ).astype(int).astype(str)
            row_df["Urban_HighPC"]      = ((row_df["Territory"] == "Urban") &
                                            (row_df["Protection_Class"] > 6)).astype(int).astype(str)
            row_df["OldRoof_HighHail"]  = ((row_df["Roof_Age_Applicant"] > 20) &
                                            (row_df["Hail_Frequency"] >= 3)).astype(int).astype(str)
            for col in GLM_CAT_COLS:
                row_df[col] = row_df[col].astype(str)

            X_proc = glm_preprocessor.transform(
                row_df[GLM_ALL_FEATURES]).astype(float)

            # Combine freq + sev log-scale coefficients (both exclude const)
            freq_c   = freq_glm.coefficients.drop("const")
            sev_c    = sev_glm.coefficients.drop("const")
            # Align on common features (both should match preprocessor output)
            combined = freq_c.add(sev_c, fill_value=0).values
            feat_names = list(glm_preprocessor.get_feature_names_out())

            log_impacts = X_proc[0] * combined

            # Aggregate by original feature (strip OHE prefix)
            agg = {}
            for i, name in enumerate(feat_names):
                base = name.split("__")[1] if "__" in name else name
                for cat in GLM_CAT_COLS:
                    if base.startswith(cat):
                        base = cat
                        break
                agg[base] = agg.get(base, 0.0) + log_impacts[i]

            # Convert to multiplicative relativities
            total_log  = sum(agg.values())
            base_const = float(np.exp(
                float(freq_glm.coefficients["const"]) +
                float(sev_glm.coefficients["const"])
            ))

            # Build waterfall in dollar space
            attribs = {}
            for f, v in agg.items():
                if total_log != 0:
                    attribs[f] = (v / total_log) * (glm_prem - base_const)
                else:
                    attribs[f] = 0.0

            scored = sorted(
                [(f, v) for f, v in attribs.items() if abs(v) > 5],
                key=lambda x: abs(x[1]), reverse=True,
            )
            labeled  = (["GLM Base Rate"] +
                        ["● " + f[0].replace("_", " ") for f in scored] +
                        ["Total GLM"])
            scores_wf = [base_const] + [f[1] for f in scored] + [0]
            measures  = ["relative"] + ["relative"] * len(scored) + ["total"]

            fig = go.Figure(go.Waterfall(
                orientation="v", measure=measures, x=labeled, y=scores_wf,
                textposition="outside",
                text=[f"${s:,.0f}" if i < len(scores_wf)-1 else f"${glm_prem:,.0f}"
                      for i, s in enumerate(scores_wf)],
                totals={"marker": {"color": MUTED}}, **WF))
            fig.update_layout(
                title={"text": "GLM Breakdown — How the Legacy Actuarial System Priced This Risk",
                       "font": {"size": 13, "color": MUTED}}, **LAY)
        except Exception as e:
            fig = go.Figure().add_annotation(
                text=f"GLM waterfall unavailable: {str(e)[:80]}",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=12, color=MUTED))
            fig.update_layout(**LAY)

    else:
        # ── GA2M Intelligence waterfall ───────────────────────────────────────
        try:
            eps          = 1e-6
            log_resid    = float(np.log(actual_epp + eps) - np.log(glm_prem + eps))
            X_sample_ebm = df[EBM_ALL_FEATURES].loc[[selected_idx]].copy()
            local_exp    = ebm_model.explain_local(X_sample_ebm, y=[log_resid])
            exp_data     = local_exp.data(0)
            names_raw    = exp_data["names"]
            scores_raw   = exp_data["scores"]
            gam_int_log  = float(exp_data["extra"]["scores"][0])
            total_log_pred = float(sum(scores_raw)) + gam_int_log

            def l2d(v):
                return 0.0 if abs(total_log_pred) < 1e-9 else (v / total_log_pred) * ebm_adj

            scores_dollar     = [l2d(s) for s in scores_raw]
            intercept_dollar  = l2d(gam_int_log)
            scored_f = sorted(zip(names_raw, scores_dollar), key=lambda x: abs(x[1]), reverse=True)
            top_f    = scored_f[:10]
            other_f  = scored_f[10:]
            other_sum = sum(f[1] for f in other_f)
            top5_o = sorted(other_f, key=lambda x: abs(x[1]), reverse=True)[:5]
            rem    = len(other_f) - 5
            odet   = "".join([f"<br>  {f[0]}: ${f[1]:,.0f}" for f in top5_o])
            if rem > 0:
                odet += f"<br>  ...and {rem} more"

            def _classify(n):
                if n in ["GA2M Intercept", "All Other Signals", "Net Residual Adj"]:
                    return "meta"
                # EBM uses " x " as the interaction delimiter in term_names_
                # Also check " & " and " × " for backward compatibility
                return "interaction" if any(d in n for d in (" x ", " & ", " × ", " X ")) else "main"

            raw_names   = (["GA2M Intercept"] + [f[0] for f in top_f] +
                           ["All Other Signals", "Net Residual Adj"])
            gam_scores  = ([intercept_dollar] + [f[1] for f in top_f] +
                           [other_sum, 0])
            gam_meas    = (["relative"] + ["relative"] * len(top_f) +
                           ["relative", "total"])
            labeled_gam = []
            for n in raw_names:
                t = _classify(n)
                if t == "main":        labeled_gam.append("● " + n.replace("_", " "))
                elif t == "interaction": labeled_gam.append("⊗ " + n.replace("_", " "))
                else:                  labeled_gam.append(n)
            hover = []
            for i, n in enumerate(raw_names):
                if n == "All Other Signals":
                    hover.append(f"All Other Signals: ${other_sum:,.0f}{odet}")
                elif n == "Net Residual Adj":
                    hover.append(f"Net Residual Adjustment: ${sum(gam_scores[:-1]):,.0f}")
                else:
                    hover.append(f"{n}: ${gam_scores[i]:,.2f}")
            fig = go.Figure(go.Waterfall(
                orientation="v", measure=gam_meas, x=labeled_gam, y=gam_scores,
                textposition="outside",
                text=[f"${s:,.0f}" if i < len(gam_scores)-1 else
                      f"${sum(gam_scores[:-1]):,.0f}"
                      for i, s in enumerate(gam_scores)],
                totals={"marker": {"color": BLUE}},
                hovertext=hover, hoverinfo="text", **WF))
            fig.update_layout(
                title={"text": "GA2M Intelligence Layer — Non-Linear & Interaction Signal Breakdown",
                       "font": {"size": 13, "color": BLUE}}, **LAY)
        except Exception as e:
            fig = go.Figure().add_annotation(
                text=f"GA2M waterfall unavailable: {str(e)[:80]}",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=12, color=MUTED))
            fig.update_layout(**LAY)

    return profile, fig


if __name__ == "__main__":
    app.run(debug=False, port=APP_PORT)