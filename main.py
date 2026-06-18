import numpy as np

np.random.seed(0)

## FORWARD PASS
#1. CONVOLUTION
def convolve(image, kernel, stride = 1, padding = 0):
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode="constant")
    output_size = ((image.shape[0] - kernel.shape[0])//stride) + 1
    convolve_out = np.zeros((output_size, output_size))

    for i in range(output_size):
        for j in range(output_size):
            row = i * stride
            col = j * stride
            convolve_out[i, j] = np.sum(image[row:row+kernel.shape[0], col:col+kernel.shape[1]] * kernel)
    
    return convolve_out

#2. RECTIFIED LINEAR UNIT
def relu(feature_map):
    relu_out = np.maximum(0, feature_map)

    return relu_out

#3. POOLING - TAKING MAX VALUE FROM NON OVERLAPPING PATCHES
def max_pool(relu_out, pool_size = 2, stride = 2):
    output_size = ((relu_out.shape[0] - pool_size) // stride) + 1
    pool_out = np.zeros((output_size, output_size))

    for i in range(output_size):
        for j in range(output_size):
            row = i * stride
            col = j * stride

            patch = relu_out[row:row+pool_size,col:col+pool_size]
            pool_out[i, j] = np.max(patch)

    return pool_out

#4. FLATTEN
def flatten(pool_out):
    flattened = pool_out.reshape(-1)

    return flattened

#5. DENSE LAYER - AGGREGATOR
def dense_forward(flattened, W, b):
    scores = np.dot(flattened, W) + b

    return scores

#6. SOFTMAX - CALC PROBABILITIES
def softmax(scores):
    scores = scores - np.max(scores)
    exp_values = np.exp(scores)
    probabilities =  exp_values / np.sum(exp_values)

    return probabilities

#7. CROSS ENTROPY - CALC LOSS
def cross_entropy(y_true, probabilities):
    probabilities = np.clip(probabilities, 1e-8, 1-1e-8)
    loss = -np.sum(y_true * np.log(probabilities))

    return loss

##BACKPROPAGATION

#1. CROSS ENTROPY AND SOFTMAX BACKWARD - CALC d_scores
def cross_entropy_backward(probabilities, y_true):
    score_grad = probabilities - y_true

    return score_grad

#2. DENSE BACKWARD - CALC CHANGE IN WEIGHTS, INPUTS AND BIASES
def dense_backward(flattened, W, score_grad):
    dW = np.outer(flattened, score_grad)
    db = score_grad
    flattened_grad = np.dot(score_grad, W.T)

    return dW, db, flattened_grad

#3. FLATTEN BACKWARD
def flatten_backward(flattened_grad, pool_shape):
    pool_out_grad = flattened_grad.reshape(pool_shape)

    return pool_out_grad

#4. POOLING BACKWARD
def max_pool_backward(relu_out, pool_out_grad, pool_size = 2, stride = 2):
    relu_out_grad = np.zeros_like(relu_out, dtype=float)
    output_size = pool_out_grad.shape[0]

    for i in range(output_size):
        for j in range(output_size):
                row = i * stride
                col = j * stride

                patch = relu_out[row:row+pool_size,col:col+pool_size]
                mask = (patch == np.max(patch))

                relu_out_grad[row:row+pool_size,col:col+pool_size] += mask * pool_out_grad[i, j]
    
    return relu_out_grad

#5. RELU BACKWARD
def relu_backward(relu_out_grad, convolve_out):
    conv_out_grad = relu_out_grad * (convolve_out > 0)

    return conv_out_grad

#6. CONVOLUTION BACKWARD
def convolve_backward(image, kernel, conv_out_grad, stride = 1, padding = 0):
    if padding > 0:
        padded_image = np.pad(image,((padding, padding),(padding, padding)),mode="constant")
    else:
        padded_image = image
    dKernel = np.zeros_like(kernel, dtype=float)
    dInput = np.zeros_like(image, dtype=float)

    output_size = conv_out_grad.shape[0]

    for i in range(output_size):
        for j in range(output_size):
            row = i * stride
            col = j * stride
            patch = padded_image[row:row+kernel.shape[0], col:col+kernel.shape[1]]
            dKernel += (patch * conv_out_grad[i, j])
            dInput[row:row+kernel.shape[0], col:col+kernel.shape[1]] += (kernel * conv_out_grad[i, j])
            if padding > 0:
                dInput = dInput[padding:-padding,padding:-padding]

    return dKernel, dInput

def forward(image, kernel, W, b, y_true):
    conv_out = convolve(image, kernel, stride=1, padding=0)
    relu_out = relu(conv_out)
    pool_out = max_pool(relu_out, pool_size=2, stride=2)
    flattened = flatten(pool_out)
    scores = dense_forward(flattened, W, b)
    y_pred = softmax(scores)
    loss = cross_entropy(y_true, y_pred)

    cache = {
    "image": image,
    "conv_out": conv_out,
    "relu_out": relu_out,
    "pool_out": pool_out,
    "flattened": flattened,
    "scores": scores
    }

    return loss, y_pred, cache

def backward(y_pred, y_true, cache, kernel, W):
    score_grad = cross_entropy_backward(y_pred, y_true)
    dW, db, flattened_grad = dense_backward(cache["flattened"], W, score_grad)
    pool_out_grad = flatten_backward(flattened_grad, cache["pool_out"].shape)
    relu_out_grad = max_pool_backward(cache["relu_out"], pool_out_grad, pool_size=2, stride=2)
    conv_out_grad = relu_backward(relu_out_grad, cache["conv_out"])
    dKernel, dInput = convolve_backward(cache["image"], kernel, conv_out_grad, stride=1, padding=0)

    return dKernel, dW, db

def update(kernel, W, b, dKernel, dW, db, lr):
    W  = W - lr * dW
    b = b - lr * db
    kernel = kernel - lr * dKernel
    
    return kernel, W, b

image = np.array([
    [1,1,1,0,0],
    [0,1,1,1,0],
    [0,0,1,1,1],
    [1,1,0,0,1],
    [1,0,1,0,1]
])

y_true = np.array([1,0])

vertical_line = np.array([
    [1,0,0,1,0],
    [1,0,0,1,0],
    [1,0,0,1,0],
    [1,0,0,1,0],
    [1,0,0,1,0]
])

vertical_label = np.array([1,0])

horizontal_line = np.array([
    [1,1,1,1,1],
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0],
    [1,1,1,1,1]
])

horizontal_label = np.array([0,1])

vertical_line_2 = np.array([
    [1,0,0,1,0],
    [1,0,1,1,0],
    [1,0,0,1,0],
    [1,1,0,1,0],
    [1,0,0,1,0]
])

horizontal_line_2 = np.array([
    [1,1,1,1,1],
    [0,0,1,0,0],
    [0,0,0,0,0],
    [0,1,0,0,0],
    [1,1,1,1,1]
])

kernel = np.random.randn(2,2) * 0.01
W = np.random.randn(4,2) * 0.01
b = np.zeros(2)

training_images = [vertical_line, vertical_line_2, horizontal_line, horizontal_line_2]
training_labels = [vertical_label, horizontal_label]

for epoch in range(5000):
    total_loss = 0
    for image, label in zip(training_images, training_labels):
        loss, y_pred, cache = forward(image, kernel, W, b, label)
        dKernel, dW, db = backward(y_pred, label, cache, kernel, W)
        kernel, W, b = update(kernel, W, b, dKernel, dW, db, 0.01)
        total_loss += loss
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: "f"{total_loss:.4f}")