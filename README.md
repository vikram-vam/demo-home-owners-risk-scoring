# Homeowners Intelligence Layer

**GLM + GA2M Two-Layer Pricing Architecture — Business Development Demo**

A carrier-facing interactive demo showing how a glass-box GA2M (Explainable Boosting Machine) residual intelligence layer captures non-linear feature effects, compounding pairwise interactions, and temporal risk decay signals that a legacy Poisson × Gamma GLM structurally cannot represent — recovering an incremental **ΔR² of +0.14** while maintaining exact book-level premium neutrality.

**Target audience:** VP Analytics · Chief Actuary · Head of Pricing

---

## Key Results (Out-of-Sample, N=20,000)

| Metric | Value |
|--------|-------|
| Legacy GLM R² | 0.6635 |
| GLM + GA2M R² | 0.8028 |
| Incremental ΔR² | +0.1393 |
| Residual recovered | 41.4% |
| Risk neutrality | E_w[uplift] = 1.000000× |
| Premium at risk (surcharges) | $16.0M across 42K policies |
| Growth opportunity (credits) | $16.0M across 58K policies |
| Net book premium change | $0 |
| Adverse selection exposure | 17.3% of policies underpriced >20% |

---

## Prerequisites

- Python 3.10 or higher
- ~2 GB RAM for training (EBM with 100K policies, 8 outer bags)
- ~2 minutes for the full pipeline run

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data and train models (run once; ~2 min)
python setup.py

# 3. Launch the demo app
python app.py
```

Then open **http://localhost:8050** in your browser.

---

## Architecture

The demo implements a two-layer pricing architecture where the GLM handles linear exposure relativities and the GA2M captures everything the GLM structurally cannot:

```
Property Features ──→ Legacy GLM ──→ GLM Pure Premium ──→ GA2M Residual ──→ Final Premium
    (28 vars)        Freq × Sev         (baseline)        (28 vars,         Intelligence
                     (16 vars)                             15 interactions)   Premium
                                        ↓                       ↓
                                   log(True ÷ GLM)      × exp(GA2M)
                                   residual target       [0.65×, 1.60×]
                                                         E_w[uplift]=1.0
```

**Layer 1 — Baseline GLM:** Poisson frequency × Gamma severity, trained with statsmodels for full actuarial diagnostics (deviance, AIC, p-values). 16 features: 12 standard industry main effects plus 4 engineered interactions (Frame×HighPC, Claims×LowDed, Urban×HighPC, OldRoof×Hail). AOI-based exposure offset. Credit score suppressed for CA/MA (regulatory compliance).

**Layer 2 — GA2M Residual:** Explainable Boosting Machine (InterpretML) trained on log(True/GLM) residuals. 28 features: 12 GLM main effects + 13 modern enrichment signals (satellite roof condition, daily wildfire index, water loss recency, pluvial flood depth, etc.) + 3 derived features. Learns 15 pairwise interactions. Predictions bounded to a [0.65×, 1.60×] corridor and risk-neutrality-normalized so total book premium is unchanged.

**Key property:** `E_w[uplift] = 1.0` — the GA2M layer redistributes premium across the portfolio without inflating the total book. This is pure redistribution, not a rate increase.

---

## Demo Tabs

The interactive Dash application has four tabs following the BD demo arc: **impact → proof → drill-down → architecture**.

### Tab 1: Business Case
Executive-level financial impact. Five KPI cards (Premium at Risk, Growth Opportunity, ΔR², Adverse Selection, Book Premium Impact). Premium flow diverging bar chart showing $16M surcharges vs $16M credits with Net = $0. Signed pricing error by risk quintile revealing systematic GLM mispricing direction. Adverse selection scatter, reclassification scatter, and 4×4 tier migration matrix.

### Tab 2: Intelligence Signals
Technical proof of non-linear signal capture. Three-layer data characteristics panel (empirical dots + GLM linear assumption + GA2M shape function) for wildfire, roof vulnerability, building code, and credit score. Signal landscape ranking top 15 features by dollar impact. Geographic intelligence showing average adjustment by state. Wildfire × Roof Vulnerability interaction heatmap with compounding risk surface up to +60%.

### Tab 3: Policy Lens
Underwriter-level policy drill-down. Eight named risk archetypes (WUI Wildfire, Hail Belt, Flood Zone, Moral Hazard, Hidden Gem, New Build Masonry, Water Recency, Suburban Standard). Three waterfall views: Strategic (3-bar overview), GLM Breakdown (per-feature actuarial audit trail), GA2M Intelligence (non-linear and interaction signal decomposition with exact dollar attribution).

### Tab 4: Framework
Architecture and validation for technical audiences. Pipeline diagram, GLM and GA2M mathematical specifications, three structural ceiling constraints, glass-box guarantee (exact additivity — no SHAP approximation), risk neutrality formula, and validation performance chips with production checklist.

---

## Synthetic Data

100,000 policies across 10 US states (CA, TX, FL, NY, CO, OK, LA, MA, WA, GA) generated with:

- **Iman-Conover rank correlation** (Gaussian copula surrogate) on 19 continuous/ordinal features reproducing realistic inter-feature correlations (Roof Age × Year Built ρ ≈ −0.80, AOI × Square Footage ρ ≈ +0.85, etc.)
- **State-conditional peril distributions:** Wildfire bimodal for CA/CO/WA, hail Poisson by state (TX/OK/CO belt), flood exponential by state (FL/LA)
- **Non-linear DGP shapes:** Convex wildfire, quadratic roof vulnerability, threshold building code, logarithmic hydrant distance, exponential water loss decay, diminishing credit score returns
- **Calibrated variance budget:** Legacy signal σ=0.32, modern signal σ=0.20, noise σ=0.25 — producing GLM R² in 0.58–0.65 range with ΔR² of 0.07–0.14
- **Credit score suppression:** CA and MA policies set to portfolio median (700) simulating state regulatory restrictions

---

## File Inventory

| File | Role |
|------|------|
| `config.py` | Central configuration: paths, hyperparameters, feature lists, state config, correlation matrix, colors |
| `requirements.txt` | Pinned package dependencies |
| `setup.py` | Pipeline orchestrator — data generation → GLM → EBM in sequence |
| `data_simulation.py` | Synthetic data generation (100K policies, copula, 10-state geography) |
| `baseline_glm.py` | Poisson × Gamma GLM (statsmodels, 16 features, exposure offset) |
| `residual_model.py` | EBM GA2M residual layer (InterpretML, 28 features, risk-neutral normalization) |
| `app.py` | Dash web application — 4-tab interactive demo |
| `assets/` | Static assets (logo, custom CSS for archetype tile hover effects) |
| `data/` | Generated CSV outputs (created by `setup.py`) |
| `models/` | Trained model artifacts (.pkl + model_metadata.json) |

### Generated Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| Raw data | `data/synthetic_homeowners_data.csv` | 100K policies with all features |
| GLM-enriched data | `data/synthetic_homeowners_data_with_baseline.csv` | + GLM predictions and split column |
| Final predictions | `data/final_predictions.csv` | + GA2M uplift, final premium, tiers |
| Frequency GLM | `models/freq_glm.pkl` | Poisson statsmodels wrapper |
| Severity GLM | `models/sev_glm.pkl` | Gamma statsmodels wrapper |
| Preprocessor | `models/glm_preprocessor.pkl` | ColumnTransformer (OHE + scaling) |
| EBM model | `models/ebm_residual_model.pkl` | Trained ExplainableBoostingRegressor |
| Metadata | `models/model_metadata.json` | Training metrics, timestamps, config |

---

## Configuration

Key parameters in `config.py`:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `N_SAMPLES` | 100,000 | Portfolio size |
| `TEST_SIZE` | 0.20 | Held-out test set fraction |
| `PREMIUM_FLOOR` | $300 | Minimum policy premium |
| `MIN_UPLIFT` / `MAX_UPLIFT` | 0.65× / 1.60× | GA2M adjustment corridor |
| `DGP_LEGACY_SCALAR` | 0.32 | GLM-accessible signal strength |
| `DGP_MODERN_SCALAR` | 0.20 | GA2M-only signal strength |
| `DGP_NOISE_SIGMA` | 0.25 | Irreducible noise floor |
| `TIER_BOUNDARIES` | 0 / 1200 / 2200 / 4000 / ∞ | Risk tier premium thresholds |
| `APP_PORT` | 8050 | Dash server port |

---

## Technology Stack

- **Python 3.10+** — core runtime
- **statsmodels** — Poisson × Gamma GLM with full actuarial diagnostics
- **InterpretML** — Explainable Boosting Machine (GA2M) with native shape functions
- **Dash + Plotly** — interactive web application and charting
- **dash-bootstrap-components** — responsive layout and UI components
- **pandas / numpy / scikit-learn** — data processing and metrics
- **joblib** — model serialization

---

## Documentation

| Document | Description |
|----------|-------------|
| `Homeowners_Intelligence_Layer_Demo_Narrative.docx` | Presenter guide with talk tracks, stage directions, visual references, and objection handling |
| `Homeowners_Intelligence_Layer_Reference_Manual.docx` | Technical reference: architecture, feature specs, variance budget, tab-by-tab walkthrough, glossary |

---

## Important Notes

**This is a demo, not a production model.** The synthetic data calibration approximates but does not replicate any specific carrier's book. Production deployment requires out-of-time validation on actual loss data, regulatory review, and actuarial sign-off.

**Risk neutrality is enforced, not assumed.** The GLM-premium-weighted normalization guarantees total book premium invariance to machine precision. In production, this would be validated at state/line/program level.

**Glass-box, not black-box.** Every GA2M prediction decomposes exactly into auditable per-feature contributions. No post-hoc SHAP approximation is involved — this is the model's actual internal arithmetic.

---

*© 2026 ValueMomentum Insurance Technology. Confidential.*
