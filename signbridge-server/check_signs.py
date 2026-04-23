import pandas as pd

df = pd.read_csv("dataset/train.csv")
available = set(df["sign"].unique())

# Most useful conversational signs to check
candidates = [
    # Greetings / basics
    "hello", "bye", "yes", "no", "please", "sorry", "thankyou",
    "help", "more", "stop", "wait", "finish", "want", "need",
    # People
    "mom", "dad", "baby", "boy", "girl", "man", "woman",
    "friend", "brother", "sister", "family",
    # Actions  
    "eat", "drink", "sleep", "play", "go", "come", "look",
    "help", "like", "love", "know", "think", "feel",
    # Descriptive
    "good", "bad", "hot", "cold", "big", "small", "fast", "slow",
    "happy", "sad", "sick", "tired", "hungry", "hurt",
    # Questions
    "what", "where", "when", "who", "why", "how",
    # Things
    "water", "food", "home", "school", "car", "book", "money",
    "phone", "dog", "cat", "bird",
    # Time
    "now", "today", "tomorrow", "yesterday", "time", "day",
    "morning", "night", "before", "after",
    # Numbers / letters context
    "all", "many", "some", "any", "other", "same", "different",
    "fine", "better", "best", "again", "already",
]

found = []
not_found = []

for s in candidates:
    if s in available:
        count = df[df["sign"] == s].shape[0]
        found.append((s, count))
    else:
        not_found.append(s)

# Sort by sample count descending
found.sort(key=lambda x: -x[1])

print(f"FOUND ({len(found)}):")
for sign, count in found:
    print(f"  {sign:<20} {count} samples")

print(f"\nNOT FOUND ({len(not_found)}):")
print(" ", ", ".join(not_found))

# Pick top 30 by sample count
top30 = [s for s, c in found[:30]]
print(f"\nTOP 30 BY SAMPLE COUNT:")
print(top30)