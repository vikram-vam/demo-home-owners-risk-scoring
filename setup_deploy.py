"""
setup_deploy.py
Run this LOCALLY before pushing to Render.

Generates a deployment-scale dataset and trains models at reduced settings
that fit within Render's free-tier memory (512 MB) and produce pkl files
small enough to commit to git (< 50 MB total).

Usage:
    python setup_deploy.py

What it changes vs setup.py:
    N_SAMPLES   : 100,000  →  20,000   (5× smaller — still sufficient for EBM)
    EBM outer_bags: 8      →  4        (halves model size and training time)
    EBM learning_rate: 0.02 → 0.025   (slight increase to compensate fewer bags)

After running:
    git add data/ models/
    git commit -m "Add pre-generated deploy artifacts"
    git push
    # then deploy to Render
"""

import os
import sys
import time

# ── Temporarily override config constants ────────────────────────────────────
import config
config.N_SAMPLES        = 20_000
config.EBM_OUTER_BAGS   = 4
config.EBM_LEARNING_RATE = 0.025

print("=" * 58)
print("  RESISCORE™ — DEPLOY-SCALE PIPELINE")
print("  20K policies · EBM outer_bags=4 · ~3 min runtime")
print("=" * 58)

os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)

# Step 1 — Data
print("\n[1/3] Generating 20,000 synthetic policies…")
t0 = time.time()
import data_simulation
df = data_simulation.generate_homeowners_data(
    n_samples=20_000, random_state=config.RANDOM_STATE)
df.to_csv(config.RAW_DATA_PATH, index=False)
print(f"  Done in {time.time()-t0:.1f}s  →  {config.RAW_DATA_PATH}")

# Step 2 — GLM
print("\n[2/3] Training GLM…")
t0 = time.time()
import baseline_glm
metrics_glm = baseline_glm.run_baseline_glm(config.RAW_DATA_PATH)
print(f"  Done in {time.time()-t0:.1f}s  |  "
      f"Train R²={metrics_glm['train_r2']}  Test R²={metrics_glm['test_r2']}")

# Step 3 — EBM
print("\n[3/3] Training EBM residual layer (outer_bags=4)…")
t0 = time.time()
import residual_model
metrics_ebm = residual_model.train_residual_ebm(config.BASELINE_DATA_PATH)
print(f"  Done in {time.time()-t0:.1f}s  |  "
      f"Final R²={metrics_ebm['final_r2']}  ΔR²={metrics_ebm['delta_r2']}")

# Report file sizes
print("\n  FILE SIZES (confirm all are < 100 MB for git)")
for path in [
    config.FINAL_DATA_PATH,
    config.FREQ_MODEL_PATH,
    config.SEV_MODEL_PATH,
    config.PREPROCESSOR_PATH,
    config.EBM_MODEL_PATH,
]:
    if os.path.exists(path):
        mb = os.path.getsize(path) / 1e6
        flag = "✓" if mb < 100 else "⚠ LARGE — consider git LFS"
        print(f"  {os.path.basename(path):<40}  {mb:6.1f} MB  {flag}")

print("""
NEXT STEPS:
  1. git add data/ models/
  2. git commit -m "Add pre-generated deploy artifacts"
  3. git push
  4. On Render: set Build Command to  pip install -r requirements.txt
               set Start Command to   gunicorn app:server --workers 1 --threads 2 --timeout 120
""")
