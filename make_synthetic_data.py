"""Create synthetic participants for testing the pipeline.

    python make_synthetic_data.py            # 5 patients + 5 controls
    python make_synthetic_data.py --n 8      # 8 + 8

Writes synthetic_data/ in the same format as the real session files (flat folder,
one ACHI_2026_9XX_session-1_sessiondata.json + _physio.csv per participant, trials inside
the JSON) plus ground_truth.csv and _ground_truth/. All names and texts are placeholders.
Existing synthetic_data/ is replaced.
"""
import argparse
import shutil

from achi_pipeline import config, synthetic

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=5, help="participants per group")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

config.use_synthetic()
# safety: only ever delete the synthetic folder, never the real Data/
assert config.DATA_DIR == config.PROJECT_ROOT / "synthetic_data", config.DATA_DIR
if config.DATA_DIR.exists():
    shutil.rmtree(config.DATA_DIR)
synthetic.make_dataset(config.DATA_DIR, n_per_group=args.n, seed=args.seed)
print(f"Wrote {2 * args.n} synthetic participants to {config.DATA_DIR}")
