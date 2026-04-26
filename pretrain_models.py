"""Pre-train ML yield models and save to disk for fast serverless startup."""

import os
import sys
import time
import datetime as dt
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    FieldConfig,
    fetch_eurostat_yields,
    build_training_dataset,
    train_yield_model,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CROPS = ["maize", "wheat"]

def main():
    print("🧠 Pre-training ML yield prediction models...\n")
    t0 = time.time()

    for crop in CROPS:
        print(f"  🤖 Training {crop}...")
        cfg = FieldConfig()
        cfg.crop = crop

        yields = fetch_eurostat_yields(crop)
        if len(yields) < 5:
            print(f"  ⚠️  Insufficient yield data for {crop}, skipping")
            continue

        training_df = build_training_dataset(cfg, yields, None)
        if len(training_df) < 5:
            print(f"  ⚠️  Insufficient training data for {crop}, skipping")
            continue

        model_bundle, scaler, feature_cols, training_df = train_yield_model(training_df)

        payload = {
            "model": model_bundle,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "training_df": training_df,
            "trained_at": dt.datetime.now().isoformat(),
            "crop": crop,
        }

        out_path = os.path.join(OUTPUT_DIR, f"{crop}_model.joblib")
        joblib.dump(payload, out_path, compress=3)
        size_kb = os.path.getsize(out_path) / 1024
        print(f"  ✅ Saved {out_path} ({size_kb:.0f} KB, {len(training_df)} years)\n")

    elapsed = time.time() - t0
    print(f"🧠 Done — {len(CROPS)} models saved to {OUTPUT_DIR}/ in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
