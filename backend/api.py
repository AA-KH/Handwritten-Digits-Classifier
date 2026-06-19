"""
FastAPI backend for MNIST CNN Showcase.
Wraps the CNN from main.py without modifying it.
All CNN logic lives in main.py - this file only adds API endpoints around it.

Note: The forward-pass functions below are copied verbatim from main.py
(inference-only subset). main.py remains completely untouched.
"""

import numpy as np
import base64
import io
import time
import os
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── CNN forward-pass functions (verbatim from main.py, inference only) ───────

def convolve(image, kernel, stride=1, padding=0):
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode="constant")
    output_size = ((image.shape[0] - kernel.shape[0]) // stride) + 1
    convolve_out = np.zeros((output_size, output_size), dtype=np.float32)
    for i in range(output_size):
        for j in range(output_size):
            row = i * stride
            col = j * stride
            convolve_out[i, j] = np.sum(image[row:row + kernel.shape[0], col:col + kernel.shape[1]] * kernel)
    return convolve_out

def conv_layer_forward(image, kernels, conv_biases, stride=1, padding=0):
    feature_maps = []
    for kernel, bias in zip(kernels, conv_biases):
        feature_maps.append(convolve(image, kernel, stride, padding) + bias)
    return np.array(feature_maps)

def relu(feature_maps):
    return np.maximum(0, feature_maps)

def max_pool(relu_out, pool_size=2, stride=2):
    output_size = ((relu_out.shape[0] - pool_size) // stride) + 1
    pool_out = np.zeros((output_size, output_size), dtype=np.float32)
    for i in range(output_size):
        for j in range(output_size):
            row = i * stride
            col = j * stride
            pool_out[i, j] = np.max(relu_out[row:row + pool_size, col:col + pool_size])
    return pool_out

def pool_layer_forward(feature_maps, pool_size=2, stride=2):
    return np.array([max_pool(f, pool_size, stride) for f in feature_maps])

def flatten(pool_out):
    return pool_out.reshape(-1)

def dense_forward(flattened, W, b):
    return np.dot(flattened, W) + b

def softmax(scores):
    scores = scores - np.max(scores)
    exp_values = np.exp(scores)
    return exp_values / np.sum(exp_values)

def forward(image, kernels, conv_biases, W1, b1, W2, b2):
    feature_maps = conv_layer_forward(image, kernels, conv_biases, stride=1, padding=0)
    relu_out = relu(feature_maps)
    pooled_maps = pool_layer_forward(relu_out, pool_size=2, stride=2)
    flattened = flatten(pooled_maps)
    hidden_scores = dense_forward(flattened, W1, b1)
    hidden_relu = relu(hidden_scores)
    output_scores = dense_forward(hidden_relu, W2, b2)
    y_pred = softmax(output_scores)
    cache = {
        "image": image,
        "feature_maps": feature_maps,
        "relu_out": relu_out,
        "pooled_maps": pooled_maps,
        "flattened": flattened,
        "hidden_scores": hidden_scores,
        "hidden_relu": hidden_relu,
    }
    return y_pred, cache


app = FastAPI(title="MNIST CNN Showcase API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global model weights ──────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")

kernels = None
conv_biases = None
W1 = None
b1 = None
W2 = None
b2 = None
model_loaded = False


def load_weights():
    global kernels, conv_biases, W1, b1, W2, b2, model_loaded
    weight_files = ["kernels.npy", "conv_biases.npy", "W1.npy", "b1.npy", "W2.npy", "b2.npy"]
    paths = [os.path.join(MODEL_DIR, f) for f in weight_files]

    if all(os.path.exists(p) for p in paths):
        kernels = np.load(paths[0])
        conv_biases = np.load(paths[1])
        W1 = np.load(paths[2])
        b1 = np.load(paths[3])
        W2 = np.load(paths[4])
        b2 = np.load(paths[5])
        model_loaded = True
        print("✅ Model weights loaded successfully")
        print(f"   kernels: {kernels.shape}, W1: {W1.shape}, W2: {W2.shape}")
    else:
        # Create random weights so the demo still works without trained weights
        print("⚠️  No saved weights found. Using random weights for demo.")
        kernels = np.random.randn(16, 3, 3).astype(np.float32) * np.sqrt(2 / 9)
        conv_biases = np.zeros(16, dtype=np.float32)
        W1 = np.random.randn(2704, 64).astype(np.float32) * np.sqrt(2 / 2704)
        b1 = np.zeros(64, dtype=np.float32)
        W2 = np.random.randn(64, 10).astype(np.float32) * np.sqrt(2 / 64)
        b2 = np.zeros(10, dtype=np.float32)
        model_loaded = False


load_weights()


# ─── Request / Response models ─────────────────────────────────────────────────
class PredictRequest(BaseModel):
    image: str  # base64-encoded PNG from canvas


class FeatureMapData(BaseModel):
    index: int
    data: list
    shape: list
    min_val: float
    max_val: float


class PredictionResponse(BaseModel):
    probabilities: list
    predicted_digit: int
    confidence: float
    latency_ms: float
    feature_maps_raw: list       # 16 feature maps after conv (pre-relu)
    feature_maps_relu: list      # 16 feature maps after relu
    feature_maps_pooled: list    # 16 feature maps after pooling
    hidden_activations: list     # 64-dim hidden layer activations
    preprocessed_image: list     # 28×28 normalised input
    model_loaded: bool


# ─── Image preprocessing ──────────────────────────────────────────────────────
def preprocess_canvas_image(b64_data: str) -> np.ndarray:
    """
    Convert a base64 PNG from the drawing canvas into a 28×28 float32 array
    matching the MNIST input format used by the CNN.

    Steps:
      1. Decode base64 → PIL Image (RGBA)
      2. Composite onto black background (so transparent == black)
      3. Convert to grayscale
      4. Resize to 28×28 with Lanczos anti-aliasing
      5. Normalise pixel values to [0, 1] (float32)
      6. Invert if the image appears to be dark-on-light (user drew on white)
    """
    # Strip data-URL prefix if present
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]

    img_bytes = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

    # Composite onto black background
    background = Image.new("RGBA", img.size, (0, 0, 0, 255))
    background.paste(img, mask=img.split()[3])
    img = background.convert("L")

    # Resize to 28×28
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0

    # MNIST convention: white digit on black background
    # If the image is mostly light, invert it
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    return arr


def normalise_maps(maps: np.ndarray) -> list:
    """Convert feature maps to nested Python lists, keeping raw float32 values."""
    return maps.tolist()


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "kernel_shape": list(kernels.shape),
        "W1_shape": list(W1.shape),
        "W2_shape": list(W2.shape),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictRequest):
    t0 = time.perf_counter()

    # 1. Preprocess
    try:
        image = preprocess_canvas_image(req.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode error: {e}")

    # 2. Run the CNN forward pass (untouched function from main.py)
    y_pred, cache = forward(image, kernels, conv_biases, W1, b1, W2, b2)

    latency_ms = (time.perf_counter() - t0) * 1000

    predicted_digit = int(np.argmax(y_pred))
    confidence = float(y_pred[predicted_digit])

    return PredictionResponse(
        probabilities=[float(p) for p in y_pred],
        predicted_digit=predicted_digit,
        confidence=confidence,
        latency_ms=round(latency_ms, 2),
        feature_maps_raw=normalise_maps(cache["feature_maps"]),
        feature_maps_relu=normalise_maps(cache["relu_out"]),
        feature_maps_pooled=normalise_maps(cache["pooled_maps"]),
        hidden_activations=[float(x) for x in cache["hidden_relu"]],
        preprocessed_image=normalise_maps(image),
        model_loaded=model_loaded,
    )


@app.get("/architecture")
def architecture():
    """Return static CNN architecture metadata for the 3-D explorer."""
    return {
        "layers": [
            {
                "id": "input",
                "name": "Input Image",
                "type": "input",
                "input_shape": [28, 28],
                "output_shape": [28, 28],
                "description": "Raw 28×28 grayscale pixel values, normalised to [0,1].",
                "params": 0,
            },
            {
                "id": "conv1",
                "name": "Convolution Layer",
                "type": "conv",
                "input_shape": [28, 28],
                "output_shape": [16, 26, 26],
                "kernel_size": [3, 3],
                "num_kernels": 16,
                "stride": 1,
                "padding": 0,
                "description": "16 learnable 3×3 filters slide across the image, each detecting a different local pattern (edges, curves, textures). Each filter produces one 26×26 feature map.",
                "params": int(kernels.size + conv_biases.size),
            },
            {
                "id": "relu1",
                "name": "ReLU Activation",
                "type": "relu",
                "input_shape": [16, 26, 26],
                "output_shape": [16, 26, 26],
                "description": "Replaces every negative value with zero. Introduces non-linearity so the network can learn complex patterns, not just linear combinations.",
                "params": 0,
            },
            {
                "id": "pool1",
                "name": "Max Pooling",
                "type": "pool",
                "input_shape": [16, 26, 26],
                "output_shape": [16, 13, 13],
                "pool_size": 2,
                "stride": 2,
                "description": "Divides each feature map into 2×2 non-overlapping patches and keeps only the maximum value. Halves spatial dimensions, providing translation invariance and reducing computation.",
                "params": 0,
            },
            {
                "id": "flatten",
                "name": "Flatten",
                "type": "flatten",
                "input_shape": [16, 13, 13],
                "output_shape": [2704],
                "description": "Unrolls the 3-D tensor (16 × 13 × 13 = 2704 values) into a single 1-D vector so it can be fed into the dense layers.",
                "params": 0,
            },
            {
                "id": "dense1",
                "name": "Hidden Dense Layer",
                "type": "dense",
                "input_shape": [2704],
                "output_shape": [64],
                "description": "Fully connected layer: every one of the 2704 inputs is connected to each of the 64 neurons. Learns high-level combinations of the spatial features extracted by the conv layer.",
                "params": int(W1.size + b1.size),
            },
            {
                "id": "relu2",
                "name": "ReLU Activation",
                "type": "relu",
                "input_shape": [64],
                "output_shape": [64],
                "description": "Second ReLU activation applied to the hidden layer outputs, discarding negative activations.",
                "params": 0,
            },
            {
                "id": "dense2",
                "name": "Output Dense Layer",
                "type": "dense",
                "input_shape": [64],
                "output_shape": [10],
                "description": "Produces 10 raw scores (logits), one per digit class (0–9). The highest score wins before softmax converts them to probabilities.",
                "params": int(W2.size + b2.size),
            },
            {
                "id": "softmax",
                "name": "Softmax",
                "type": "softmax",
                "input_shape": [10],
                "output_shape": [10],
                "description": "Converts raw logits into a probability distribution that sums to 1. Each value represents the model's confidence for that digit class.",
                "params": 0,
            },
        ],
        "total_params": int(kernels.size + conv_biases.size + W1.size + b1.size + W2.size + b2.size),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
