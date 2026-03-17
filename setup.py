#!/usr/bin/env python
# ==============================================================================
# setup.py
# Pipeline orchestrator for GLM + GA2M Residual Demo
#
# Usage:
#   python setup.py                  # full regeneration
#   python setup.py --skip-if-exists # skip if final_predictions.csv exists
#   python setup.py --help           # print this help
#
# Steps (in order):
#   1. Create data/ and models/ directories
#   2. Generate synthetic homeowners data (100K policies)
#   3. Train baseline GLM (Poisson × Gamma, statsmodels)
#   4. Train GA2M residual layer (EBM via InterpretML)
#   5. Print pipeline summary with key metrics
#
# Expected total runtime: 3–8 minutes on a modern laptop (100K records, EBM 8 bags)
# ==============================================================================

import argparse
import os
import sys
import time
import json


# ── Helpers ──────────────────────────────────────────────────────────────────

def banner(msg: str) -> None:
    """Print a formatted section banner."""
    width = 62
    print("\n" + "=" * width)
    print(f"  {msg}")
    print("=" * width)


def elapsed(start: float) -> str:
    """Return human-readable elapsed time string."""
    secs = time.time() - start
    if secs < 60:
        return f"{secs:.1f}s"
    return f"{secs / 60:.1f}min"


def check_imports() -> bool:
    """Verify all required packages are importable before starting."""
    required = [
        ("dash",             "dash"),
        ("dash_bootstrap_components", "dash-bootstrap-components"),
        ("plotly",           "plotly"),
        ("pandas",           "pandas"),
        ("numpy",            "numpy"),
        ("sklearn",          "scikit-learn"),
        ("statsmodels",      "statsmodels"),
        ("interpret",        "interpret"),
        ("joblib",           "joblib"),
        ("scipy",            "scipy"),
    ]
    missing = []
    for module, package in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print("\nERROR: The following packages are missing:")
        for pkg in missing:
            print(f"  pip install {pkg}")
        print("\nInstall all at once with:")
        print("  pip install -r requirements.txt")
        return False
    return True


# ── Pipeline Steps ────────────────────────────────────────────────────────────

def step_create_dirs() -> None:
    """Create data/ and models/ directories if they don't exist."""
    from config import DATA_DIR, MODEL_DIR
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    # Create .gitkeep so empty dirs are tracked by git
    for d in [DATA_DIR, MODEL_DIR]:
        gitkeep = os.path.join(d, ".gitkeep")
        if not os.path.exists(gitkeep):
            open(gitkeep, "w").close()
    print(f"  Directories ready: {DATA_DIR}/, {MODEL_DIR}/")


def step_generate_data() -> dict:
    """Run data_simulation.py and return summary stats."""
    from config import RAW_DATA_PATH, N_SAMPLES, RANDOM_STATE
    import data_simulation

    print(f"  Generating {N_SAMPLES:,} synthetic homeowners policies...")
    t0 = time.time()
    df = data_simulation.generate_homeowners_data(
        n_samples=N_SAMPLES,
        random_state=RANDOM_STATE,
    )
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    dur = elapsed(t0)

    stats = {
        "n_policies":       len(df),
        "mean_premium":     round(float(df["Expected_Pure_Premium"].mean()), 2),
        "median_premium":   round(float(df["Expected_Pure_Premium"].median()), 2),
        "claim_rate":       round(float((df["Claim_Count"] > 0).mean()), 4),
        "mean_severity":    round(
            float(df.loc[df["Claim_Amount"] > 0, "Claim_Amount"].mean()), 2
        ),
        "data_path":        RAW_DATA_PATH,
        "elapsed":          dur,
    }
    print(f"  Data generation complete in {dur}")
    print(f"    Policies    : {stats['n_policies']:,}")
    print(f"    Mean premium: ${stats['mean_premium']:,.2f}")
    print(f"    Claim rate  : {stats['claim_rate']:.2%}")
    return stats


def step_train_glm() -> dict:
    """Run baseline_glm.py and return performance metrics."""
    import baseline_glm
    from config import BASELINE_DATA_PATH, RAW_DATA_PATH

    print(f"  Training baseline GLM on {RAW_DATA_PATH}...")
    t0 = time.time()
    metrics = baseline_glm.run_baseline_glm(RAW_DATA_PATH)
    dur = elapsed(t0)

    if metrics is None:
        metrics = {}
    metrics["elapsed"] = dur
    print(f"  GLM training complete in {dur}")
    return metrics


def step_train_ebm() -> dict:
    """Run residual_model.py and return performance metrics."""
    import residual_model
    from config import BASELINE_DATA_PATH

    print(f"  Training GA2M (EBM) residual layer on {BASELINE_DATA_PATH}...")
    t0 = time.time()
    metrics = residual_model.train_residual_ebm(BASELINE_DATA_PATH)
    dur = elapsed(t0)

    if metrics is None:
        metrics = {}
    metrics["elapsed"] = dur
    print(f"  EBM training complete in {dur}")
    return metrics


def print_summary(data_stats: dict, glm_metrics: dict, ebm_metrics: dict,
                  total_start: float) -> None:
    """Print a formatted pipeline completion summary."""
    from config import (FINAL_DATA_PATH, FREQ_MODEL_PATH, SEV_MODEL_PATH,
                        EBM_MODEL_PATH, METADATA_PATH)

    banner("PIPELINE COMPLETE — DEMO READY")

    print("\n  DATA")
    print(f"    Policies generated : {data_stats.get('n_policies', '?'):,}")
    print(f"    Mean pure premium  : ${data_stats.get('mean_premium', 0):,.2f}")
    print(f"    Claim rate         : {data_stats.get('claim_rate', 0):.2%}")
    print(f"    Mean severity      : ${data_stats.get('mean_severity', 0):,.2f}")
    print(f"    Data step time     : {data_stats.get('elapsed', '?')}")

    print("\n  BASELINE GLM (Poisson × Gamma, statsmodels)")
    print(f"    Train R²           : {glm_metrics.get('train_r2', '?')}")
    print(f"    Test  R²  (OOS)    : {glm_metrics.get('test_r2', '?')}")
    print(f"    Test  RMSE         : {glm_metrics.get('test_rmse', '?')}")
    print(f"    GLM step time      : {glm_metrics.get('elapsed', '?')}")

    print("\n  GA2M RESIDUAL LAYER (EBM, InterpretML)")
    print(f"    Test  R²  (OOS)    : {ebm_metrics.get('final_r2', '?')}")
    print(f"    Incremental ΔR²    : {ebm_metrics.get('delta_r2', '?')}")
    print(f"    Risk neutrality    : {ebm_metrics.get('risk_neutral_check', '?')}")
    print(f"    EBM step time      : {ebm_metrics.get('elapsed', '?')}")

    print("\n  ARTIFACTS")
    for label, path in [
        ("Final data",    FINAL_DATA_PATH),
        ("Freq GLM",      FREQ_MODEL_PATH),
        ("Sev GLM",       SEV_MODEL_PATH),
        ("EBM model",     EBM_MODEL_PATH),
        ("Metadata",      METADATA_PATH),
    ]:
        exists = "✓" if os.path.exists(path) else "✗ MISSING"
        print(f"    {label:<16}: {path}  [{exists}]")

    print(f"\n  TOTAL PIPELINE TIME: {elapsed(total_start)}")
    print("\n  NEXT STEP:")
    print("    python app.py")
    print(f"    Then open: http://localhost:8050\n")


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Demo setup — generates synthetic data and trains the "
            "GLM + GA2M pipeline.\n\n"
            "Run this once before launching app.py.\n"
            "Expected runtime: 3–8 minutes (100K policies, EBM outer_bags=8)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help=(
            "Skip the full pipeline if data/final_predictions.csv already exists. "
            "Useful for iterating on app.py without re-training models."
        ),
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Only regenerate synthetic data; skip GLM and EBM training.",
    )
    parser.add_argument(
        "--glm-only",
        action="store_true",
        help="Only (re-)train the GLM; skip data generation and EBM.",
    )
    parser.add_argument(
        "--ebm-only",
        action="store_true",
        help="Only (re-)train the EBM residual layer; skip data and GLM.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    total_start = time.time()

    banner("GLM + GA2M PIPELINE SETUP")
    print("  Two-layer homeowners risk scoring: GLM baseline + GA2M residual")
    print("  Target audience: Carrier VP Analytics, Chief Actuary, Head of Pricing")

    # ── Import guard ──────────────────────────────────────────────────────────
    print("\n[0/4] Checking package imports...")
    if not check_imports():
        sys.exit(1)
    print("  All required packages found.")

    # ── Skip-if-exists guard ──────────────────────────────────────────────────
    if args.skip_if_exists:
        from config import FINAL_DATA_PATH
        if os.path.exists(FINAL_DATA_PATH):
            print(f"\n  --skip-if-exists: '{FINAL_DATA_PATH}' found.")
            print("  Pipeline skipped. Run 'python app.py' to launch demo.")
            print(f"  (Total check time: {elapsed(total_start)})")
            return

    # ── Step 0: Directories ───────────────────────────────────────────────────
    banner("Step 1/4 — Creating Directories")
    step_create_dirs()

    data_stats, glm_metrics, ebm_metrics = {}, {}, {}

    # ── Step 1: Data ──────────────────────────────────────────────────────────
    if not args.glm_only and not args.ebm_only:
        banner("Step 2/4 — Generating Synthetic Data")
        data_stats = step_generate_data()
    else:
        print("\n[Skipping data generation per --glm-only / --ebm-only flag]")

    # ── Step 2: GLM ───────────────────────────────────────────────────────────
    if not args.data_only and not args.ebm_only:
        banner("Step 3/4 — Training Baseline GLM")
        try:
            glm_metrics = step_train_glm()
        except Exception as exc:
            print(f"\n  ERROR during GLM training: {exc}")
            print("  Check baseline_glm.py and the data file.")
            raise
    else:
        print("\n[Skipping GLM training per --data-only / --ebm-only flag]")

    # ── Step 3: EBM ───────────────────────────────────────────────────────────
    if not args.data_only and not args.glm_only:
        banner("Step 4/4 — Training GA2M Residual Layer (EBM)")
        try:
            ebm_metrics = step_train_ebm()
        except Exception as exc:
            print(f"\n  ERROR during EBM training: {exc}")
            print("  Check residual_model.py and the baseline data file.")
            raise
    else:
        print("\n[Skipping EBM training per --data-only / --glm-only flag]")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(data_stats, glm_metrics, ebm_metrics, total_start)


if __name__ == "__main__":
    main()
