<div align="center">

# 🧠 NeuralVis — Handwritten Digits Classifier

### A Convolutional Neural Network built entirely from scratch in NumPy, with a live interactive visualisation dashboard.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://handwritten-digits-classifier-yx3h.onrender.com)
[![GitHub](https://img.shields.io/badge/GitHub-AA--KH-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AA-KH/Handwritten-Digits-Classifier)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

</div>

---

## ✨ What is this?

**NeuralVis** is a full-stack machine learning project that implements a **Convolutional Neural Network (CNN) entirely from scratch** — no PyTorch, no TensorFlow, no Keras. Every single mathematical operation (convolution, backpropagation, gradient descent) is hand-coded using **NumPy** only.

The project then wraps this raw CNN in a polished interactive web interface where you can:

- ✏️ **Draw any digit (0–9)** on a canvas
- ⚡ **Watch the CNN classify it in real time** with live probability scores
- 🔬 **Inspect every internal layer** — raw feature maps, ReLU activations, pooled maps, and hidden neuron activations
- 🏗️ **Explore the architecture** in an interactive node graph
- 🔄 **Follow the data pipeline** step by step through the network

> **⚠️ Latency Notice:** The hosted demo on Render uses a free-tier server that spins down when idle. For real-time, low-latency predictions, **run the project locally** (see below).

---

## 🎬 Live Demo

**[https://handwritten-digits-classifier-yx3h.onrender.com](https://handwritten-digits-classifier-yx3h.onrender.com)**

> The first request after a period of inactivity may take ~30 seconds while the server wakes up. All subsequent predictions will be fast.

---

## 🧠 The Concepts

This project implements a classic CNN architecture. Here's what every layer does:

### 1. 📥 Input — `28 × 28`
The MNIST dataset images are 28×28 grayscale images. Pixel values are normalised to the range `[0, 1]` (divided by 255). When you draw on the canvas, your drawing is preprocessed into this exact format before being sent to the model.

### 2. 🔲 Convolution Layer — `16 filters, 3×3 kernel, stride 1`
The core operation of any CNN. 16 learnable 3×3 filters slide across the 28×28 input image. Each filter looks at a small 3×3 patch of pixels at a time and computes a weighted sum — effectively detecting a specific low-level pattern (an edge, a curve, a corner). Each filter produces one **feature map** of shape `26×26` (because a 3×3 filter on a 28×28 image with no padding fits in 26 positions per axis). The result is 16 separate feature maps.

```
Output shape: (16, 26, 26)
Parameters:   16 × (3×3 weights + 1 bias) = 160
```

The convolution at each position `(i, j)` is:

$$\text{output}[i,j] = \sum_{m=0}^{2} \sum_{n=0}^{2} \text{image}[i+m,\ j+n] \times \text{kernel}[m,n] + \text{bias}$$

### 3. ⚡ ReLU Activation — `Rectified Linear Unit`
Applied element-wise after convolution. Any negative value becomes zero, positive values pass through unchanged:

$$\text{ReLU}(x) = \max(0, x)$$

This introduces **non-linearity** into the network. Without activation functions, stacking multiple layers would still only be capable of computing a linear transformation — no matter how deep the network. ReLU also helps avoid the vanishing gradient problem.

### 4. 🔽 Max Pooling — `2×2 pool, stride 2`
Divides each 26×26 feature map into non-overlapping 2×2 patches and keeps only the **maximum value** from each patch. This:
- Halves the spatial dimensions from `26×26` → `13×13`
- Makes the representation **translation-invariant** (the model detects the presence of a feature regardless of its exact position)
- Reduces the number of parameters and computation for subsequent layers

```
Output shape: (16, 13, 13)
Parameters:   0 (no learnable weights)
```

### 5. 📐 Flatten
Unrolls the 3D tensor `(16, 13, 13)` into a single 1D vector of length **2704** `(16 × 13 × 13 = 2704)`. This bridges the convolutional part of the network to the fully-connected (dense) layers.

### 6. 🔗 Hidden Dense Layer — `2704 → 64 neurons`
A fully-connected layer where every one of the 2704 inputs connects to each of the 64 output neurons. Each neuron computes:

$$z = W \cdot x + b$$

This layer learns **high-level combinations** of the spatial features extracted by the conv layer.

```
Parameters: 2704 × 64 weights + 64 biases = 173,120
```

### 7. ⚡ ReLU Activation (again)
Same as before — introduces non-linearity into the dense layer.

### 8. 🎯 Output Dense Layer — `64 → 10 neurons`
Produces 10 raw scores (logits), one per digit class (0–9). The neuron with the highest score corresponds to the model's predicted digit.

```
Parameters: 64 × 10 weights + 10 biases = 650
```

### 9. 📊 Softmax
Converts the 10 raw logits into a **probability distribution** that sums to 1.0. Each value represents the model's confidence for that digit class:

$$P(y = k) = \frac{e^{z_k}}{\sum_{j=0}^{9} e^{z_j}}$$

The implementation subtracts the maximum logit before exponentiating for **numerical stability**.

---

### 🔁 Backpropagation & Training

The network is trained using **mini-batch stochastic gradient descent** with **cross-entropy loss**:

$$\mathcal{L} = -\sum_{k} y_k \log(\hat{y}_k)$$

Gradients are propagated backwards through every layer — softmax, dense, ReLU, flatten, pooling, and convolution — computing `dW`, `db`, and `d_input` for each. Weights are updated with a fixed learning rate of `0.005`.

#### Training Configuration

| Parameter | Value |
|---|---|
| Dataset | MNIST (provided in `backend/dataset/`) |
| Training samples | 10,000 images |
| Epochs | 20 |
| Batch size | 32 |
| Learning rate | 0.005 |
| Weight save frequency | Every 5 epochs (when test accuracy improves) |
| Random seed | 0 |

> **Pre-trained weights are included** in `backend/weights/` — you don't need to retrain from scratch. The model loads them automatically when the server starts.

---

## 🗂️ Project Structure

```
CNN from scratch/
├── backend/
│   ├── api.py              # FastAPI server — wraps the CNN with HTTP endpoints
│   ├── main.py             # The raw CNN: all training + inference logic (NumPy only)
│   ├── requirements.txt    # Python dependencies
│   ├── weights/            # ✅ Pre-trained model weights (included)
│   │   ├── kernels.npy
│   │   ├── conv_biases.npy
│   │   ├── W1.npy
│   │   ├── b1.npy
│   │   ├── W2.npy
│   │   └── b2.npy
│   └── dataset/            # ✅ MNIST dataset binary files (included)
│       ├── train-images.idx3-ubyte
│       ├── train-labels.idx1-ubyte
│       ├── t10k-images.idx3-ubyte
│       └── t10k-labels.idx1-ubyte
│
├── frontend/
│   ├── src/
│   │   ├── components.tsx  # The entire React frontend in one file
│   │   ├── index.css       # Global styles & Tailwind directives
│   │   └── vite-env.d.ts   # Vite type references
│   ├── index.html
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── tsconfig.json
│   └── package.json
│
├── main.ipynb              # Original Jupyter notebook used for development
└── .gitignore
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **CNN / Training** | Python, NumPy |
| **API Server** | FastAPI, Uvicorn |
| **Image Processing** | Pillow |
| **Frontend Framework** | React 18, TypeScript |
| **Build Tool** | Vite |
| **Styling** | Tailwind CSS |
| **Animations** | Framer Motion |
| **State Management** | Zustand |
| **Deployment** | Render (Backend: Web Service, Frontend: Static Site) |

---

## 🚀 Running Locally

### Prerequisites
- **Python 3.11+**
- **Node.js 18+** and **npm**
- **Git**

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/AA-KH/Handwritten-Digits-Classifier.git
cd Handwritten-Digits-Classifier
```

---

### Step 2 — Start the Backend

```bash
# Navigate to the backend folder
cd backend

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Start the API server
uvicorn api:app --host 0.0.0.0 --port 8000
```

You should see:
```
✅ Model weights loaded successfully
   kernels: (16, 3, 3), W1: (2704, 64), W2: (64, 10)
INFO:     Uvicorn running on http://0.0.0.0:8000
```

You can verify the backend is healthy at: **[http://localhost:8000/health](http://localhost:8000/health)**

---

### Step 3 — Start the Frontend

Open a **second terminal** and run:

```bash
# Navigate to the frontend folder
cd frontend

# Install Node dependencies
npm install

# Start the Vite development server
npm run dev
```

You should see:
```
  VITE v5.x.x  ready in 300ms

  ➜  Local:   http://localhost:5173/
```

Open **[http://localhost:5173](http://localhost:5173)** in your browser. The frontend automatically proxies all `/api` requests to the backend on port 8000.

---

### Step 4 — (Optional) Retrain the Model

If you want to retrain the CNN from scratch, run `main.py` from inside the `backend/` directory:

```bash
cd backend
python main.py
```

> ⚠️ **Note:** Training runs 20 epochs with batches of 32 samples on 10,000 training images. The model evaluates on the test set every 5 epochs and saves updated weights to `backend/weights/` only when test accuracy improves. Training is CPU-only and may take some time.

---

## 📡 API Endpoints

Once the backend is running, these endpoints are available:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check — confirms model is loaded |
| `POST` | `/predict` | Accepts a base64 PNG, returns predictions + all layer activations |
| `GET` | `/architecture` | Returns static CNN architecture metadata |

---

## 📝 Notes

- **Pre-trained weights are included** — the model loads them immediately on startup. No training required to use the app.
- **The MNIST dataset is included** in `backend/dataset/` — needed only if you want to retrain.
- The model was trained for **20 epochs** on **10,000 images** with weights saved every **5 epochs** whenever test accuracy improved.
- The CNN is intentionally kept simple (single conv layer) to make every layer's behaviour clearly observable in the visualiser.
- `main.py` is the original, untouched training script. `api.py` wraps it for HTTP access without modifying it.

---

<div align="center">

Built with 🤍 — pure NumPy, no ML frameworks.

</div>
