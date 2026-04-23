"""
download_signs.py
==================
Picks the best 30 conversational signs available in the
dataset and downloads only those parquet files.

Run: python download_signs.py
"""

import os
import pandas as pd
import subprocess
import sys

# ── Our final 30 signs — all confirmed available in dataset ──────────
SIGNS_30 = [
    # Greetings / basics
    "hello", "bye", "yes", "no", "please",
    # People / family
    "boy", "girl", "brother", "aunt", "uncle" ,
    # Actions
    "drink", "bird", "sleep", "bath", "blow",
    # Descriptive
    "bad", "better", "all", "any", "another",
    # Places / things
    "home", "bed", "book", "car", "chair",
    # Time / other useful
    "where", "why", "fine", "time", "before",
]

print(f"Target signs ({len(SIGNS_30)}):")
for i, s in enumerate(SIGNS_30, 1):
    print(f"  {i:2d}. {s}")

# ── Verify all are in train.csv ───────────────────────────────────────
df         = pd.read_csv("dataset/train.csv")
available  = set(df["sign"].unique())
missing    = [s for s in SIGNS_30 if s not in available]

if missing:
    print(f"\nWARNING — these signs not in dataset: {missing}")
    print("Remove them from SIGNS_30 and rerun.")
    sys.exit(1)

print(f"\nAll 30 signs confirmed in dataset ✓")

# ── Get file paths for our 30 signs ──────────────────────────────────
our_df  = df[df["sign"].isin(SIGNS_30)]
paths   = our_df["path"].unique().tolist()

print(f"Files to download: {len(paths)}")
print(f"Estimated size:    ~150-200 MB")
print()

# ── Save the signs list for training script ───────────────────────────
os.makedirs("dataset", exist_ok=True)
with open("dataset/signs_list.txt", "w") as f:
    for s in SIGNS_30:
        f.write(s + "\n")
print("Saved dataset/signs_list.txt")

# ── Download each file using kaggle CLI ──────────────────────────────
print("\nStarting download...")
print("This will take a few minutes depending on your internet speed.\n")

success = 0
failed  = []

for i, path in enumerate(paths, 1):
    dest_dir  = os.path.join("dataset", os.path.dirname(path))
    dest_file = os.path.join("dataset", path)

    # Skip if already downloaded
    if os.path.exists(dest_file):
        success += 1
        if i % 50 == 0:
            print(f"  [{i}/{len(paths)}] skipping already downloaded files...")
        continue

    os.makedirs(dest_dir, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle",
            "competitions", "download",
            "-c", "asl-signs",
            "-f", path,
            "-p", dest_dir,
            "--quiet"
        ],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        success += 1
    else:
        failed.append(path)

    # Progress update every 100 files
    if i % 100 == 0 or i == len(paths):
        print(f"  [{i}/{len(paths)}] downloaded {success} files...")

print(f"\n✓ Download complete: {success}/{len(paths)} files")

if failed:
    print(f"✗ Failed: {len(failed)} files")
    with open("dataset/failed_downloads.txt", "w") as f:
        for p in failed:
            f.write(p + "\n")
    print("  Failed paths saved to dataset/failed_downloads.txt")
    print("  You can rerun this script to retry failed files.")

print("\nNext step: python prepare_data.py")