"""
reassemble.py
-------------
Run this ONCE to convert the 5 Kaggle shard files into animal_best.pt
Place this file inside:  Wildlife intrusion/
"""

import os
import zipfile

# ── Paths ──────────────────────────────────────────────────────────────────
SHARD_DIR  = "animal_best_shards"   # folder where you put the 5 files
OUTPUT_PT  = os.path.join("models", "animal_best.pt")

# ── Step 1: Create models/ folder if it doesn't exist ─────────────────────
os.makedirs("models", exist_ok=True)
print("models/ folder ready")

# ── Step 2: Check all 5 shard files are present ───────────────────────────
required = ["data.pkl", "_format_version", "_storage_alignment", "byteorder", "version"]
missing  = [f for f in required if not os.path.exists(os.path.join(SHARD_DIR, f))]

if missing:
    print(f"\nERROR: Missing shard files in '{SHARD_DIR}/':")
    for m in missing:
        print(f"  - {m}")
    print(f"\nMake sure all 5 files are inside the '{SHARD_DIR}/' folder.")
    exit(1)

print(f"All 5 shard files found in '{SHARD_DIR}/'")

# ── Step 3: Reassemble into .pt (zip archive) ─────────────────────────────
# ── Step 3: Reassemble into .pt (zip archive) ─────────────────────────────
with zipfile.ZipFile(OUTPUT_PT, "w", compression=zipfile.ZIP_STORED) as zf:
    for root, dirs, files in os.walk(SHARD_DIR):
        for file in files:
            full_path = os.path.join(root, file)

            # preserve full structure (including data/ and .data/)
            rel_path = os.path.relpath(full_path, SHARD_DIR)
            arcname = os.path.join("archive", rel_path)

            zf.write(full_path, arcname)
            print(f"  packed: {rel_path}")

print(f"\nDone! Saved to: {OUTPUT_PT}")

# ── Step 4: Quick verify ──────────────────────────────────────────────────
print("\nVerifying model loads correctly...")
try:
    from ultralytics import YOLO
    model = YOLO(OUTPUT_PT)
    print("Model loaded successfully!")
    print("Class names:", model.names)
except Exception as e:
    print(f"Load error: {e}")
    print("The .pt file was created but may need ultralytics installed.")
    print("Run:  pip install ultralytics")
