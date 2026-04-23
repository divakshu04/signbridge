"""
Direct model test - checks if model file can be loaded and if it makes sense
"""
import numpy as np
from tensorflow import keras
import json

# Load config
with open("model_config.json") as f:
    config = json.load(f)

print(f"✓ Config loaded: {config['num_classes']} signs, {config['features']} features")
print(f"  Signs: {config['signs'][:5]}... (hello is index 0, dad is index 19)")

# Load model
try:
    model = keras.models.load_model("signbridge_model.keras")
    print(f"✓ Model loaded successfully")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
except Exception as e:
    print(f"✗ ERROR loading model: {e}")
    exit(1)

# Test with random input (should give somewhat balanced output)
random_input = np.random.randn(1, 30, 258).astype(np.float32)
random_input = np.clip(random_input, -3.0, 3.0)
probs = model.predict(random_input, verbose=0)[0]

top5_idx = np.argsort(probs)[-5:][::-1]
top5_signs = [config['signs'][int(i)] for i in top5_idx]
top5_conf = [float(probs[i]) for i in top5_idx]

print(f"\n🧪 Random input predictions (should be balanced):")
print(f"  Top 5: {list(zip(top5_signs, [f'{c:.3f}' for c in top5_conf]))}")
print(f"  Min prob: {probs.min():.6f}, Max prob: {probs.max():.6f}, Mean: {probs.mean():.6f}")

if probs.max() > 0.95:
    print(f"\n⚠️  WARNING: Model is heavily biased! One sign has {probs.max():.1%} probability")
    print(f"   This suggests the model is corrupted or poorly trained")
else:
    print(f"\n✓ Model output looks reasonable")

# Check if any sign gets 0 probability
zero_probs = np.sum(probs < 1e-6)
if zero_probs > 0:
    print(f"⚠️  {zero_probs} signs have near-zero probability ({zero_probs/30*100:.1f}%)")
