"""
explore_signs.py
=================
Reads train.csv and shows what signs are available
and how many samples each sign has.

Run: python explore_signs.py
"""

import pandas as pd

df = pd.read_csv("dataset/train.csv")

print("Total samples:", len(df))
print("Columns:", df.columns.tolist())
print()

# Count samples per sign
sign_counts = df["sign"].value_counts()

print(f"Total unique signs: {len(sign_counts)}")
print()

# Our 30 target signs — most common conversational words
target_signs = [
    "hello", "thank-you", "yes", "no", "help",
    "please", "stop", "water", "more", "sorry",
    "good", "bad", "name", "what", "where",
    "how", "why", "understand", "again", "fine",
    "tired", "eat", "drink", "home", "work",
    "family", "friend", "love", "time", "money"
]

print("Checking target signs availability:")
print("-" * 45)

available   = []
unavailable = []

for sign in target_signs:
    if sign in sign_counts:
        count = sign_counts[sign]
        available.append(sign)
        print(f"  ✓  {sign:<15}  {count} samples")
    else:
        unavailable.append(sign)
        print(f"  ✗  {sign:<15}  NOT FOUND")

print()
print(f"Available:   {len(available)}/30")
print(f"Missing:     {len(unavailable)}/30")

if unavailable:
    print(f"\nMissing signs: {unavailable}")
    print("\nSuggested replacements from dataset:")
    # Show top signs from dataset that aren't in our list
    all_signs  = set(sign_counts.index.tolist())
    our_signs  = set(target_signs)
    extras     = sorted(all_signs - our_signs)
    print("  Available signs sample:", extras[:40])

print()
print("Saving available sign list to dataset/signs_list.txt")

with open("dataset/signs_list.txt", "w") as f:
    for sign in available:
        f.write(sign + "\n")

print("Done. Next step: python download_signs.py")