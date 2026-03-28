import tensorflow as tf
import numpy as np
import os

MODEL_DIR = "../models"
latest_version = max([int(i) for i in os.listdir(MODEL_DIR) if i.isdigit()])

print(f"Loading model version: {latest_version}")
model = tf.keras.layers.TFSMLayer(
    f"{MODEL_DIR}/{latest_version}",
    call_endpoint="serve"
)

# Dummy image input
img_array = np.random.rand(1, 256, 256, 3).astype(np.float32)

print("Running prediction...")
predictions = model(img_array)

print("Predictions object type:", type(predictions))
print("Predictions object:", predictions)

import json
try:
    index = int(np.argmax(predictions))
    print("np.argmax(predictions) = ", index)
except Exception as e:
    print("np.argmax error:", e)

try:
    confidence = float(np.max(predictions)) * 100
    print("np.max confidence = ", confidence)
except Exception as e:
    print("np.max error:", e)
