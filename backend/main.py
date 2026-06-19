import numpy as np
import struct
from scipy.ndimage import rotate

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
        images = images.astype(np.float32) / 255.0

    return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels

np.random.seed(0)

## FORWARD PASS
#1. CONVOLUTION
def convolve(image, kernel, stride = 1, padding = 0):
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode="constant")
    output_size = ((image.shape[0] - kernel.shape[0])//stride) + 1
    convolve_out = np.zeros((output_size, output_size), dtype=np.float32)

    for i in range(output_size):
        for j in range(output_size):
            row = i * stride
            col = j * stride
            convolve_out[i, j] = np.sum(image[row:row+kernel.shape[0], col:col+kernel.shape[1]] * kernel)
    
    return convolve_out

# CONVOLUTION FOR MULTIPLE KERNELS
def conv_layer_forward(image, kernels, conv_biases, stride = 1, padding = 0):
    feature_maps = []
    for kernel,bias in zip(kernels, conv_biases):
        feature_map = convolve(image, kernel, stride, padding) + bias
        feature_maps.append(feature_map)

    return np.array(feature_maps)

#2. RECTIFIED LINEAR UNIT
def relu(feature_maps):
    relu_out = np.maximum(0, feature_maps)

    return relu_out

#3. POOLING - TAKING MAX VALUE FROM NON OVERLAPPING PATCHES
def max_pool(relu_out, pool_size = 2, stride = 2):
    output_size = ((relu_out.shape[0] - pool_size) // stride) + 1
    pool_out = np.zeros((output_size, output_size), dtype=np.float32)

    for i in range(output_size):
        for j in range(output_size):
            row = i * stride
            col = j * stride

            patch = relu_out[row:row+pool_size,col:col+pool_size]
            pool_out[i, j] = np.max(patch)

    return pool_out

# POOLING FOR MULTIPLE FEATURE MAPS AKA OUTPUTS FROM 
def pool_layer_forward(feature_maps, pool_size = 2, stride = 2):
    pooled_maps = []
    for feature in feature_maps:
        pooled_maps.append(max_pool(feature, pool_size, stride))

    return np.array(pooled_maps)

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
    relu_out_grad = np.zeros_like(relu_out, dtype=np.float32)
    output_size = pool_out_grad.shape[0]

    for i in range(output_size):
        for j in range(output_size):
                row = i * stride
                col = j * stride

                patch = relu_out[row:row+pool_size,col:col+pool_size]
                mask = (patch == np.max(patch))

                relu_out_grad[row:row+pool_size,col:col+pool_size] += mask * pool_out_grad[i, j]
    
    return relu_out_grad

## MULTI KERNEL BACKWARD POOLING
def pool_layer_backward(relu_out, pooled_maps_grad, pool_size = 2, stride = 2):
    relu_maps_grad = []
    
    for output, pooled_map_grad in zip(relu_out, pooled_maps_grad):
        relu_maps_grad.append(max_pool_backward(output, pooled_map_grad, pool_size, stride))

    return np.array(relu_maps_grad)

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
    dKernel = np.zeros_like(kernel, dtype=np.float32)
    dInput = np.zeros_like(padded_image, dtype=np.float32)

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

#MULTI KERNEL BACKWARD CONVOLUTION
def conv_layer_backward(image, kernels, conv_biases, feature_maps_grad, stride = 1, padding = 0):
    kernels_grad = np.zeros_like(kernels, dtype=np.float32)
    conv_biases_grad = np.zeros_like(conv_biases, dtype=np.float32)
    image_grad = np.zeros_like(image, dtype=np.float32)

    for kernel_index in range(len(kernels)):
        kernel_grad, single_image_grad = (convolve_backward(image, kernels[kernel_index], feature_maps_grad[kernel_index], stride, padding))

        kernels_grad[kernel_index] = kernel_grad
        conv_biases_grad[kernel_index] = np.sum(feature_maps_grad[kernel_index])
        image_grad += (single_image_grad)

    return kernels_grad, conv_biases_grad, image_grad

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
    "hidden_relu": hidden_relu
    }

    return y_pred, cache

def backward(y_pred, y_true, cache, kernels, conv_biases, W1, W2):
    score_grad = cross_entropy_backward(y_pred, y_true)
    dW2, db2, hidden_relu_grad = dense_backward(cache["hidden_relu"], W2, score_grad)
    hidden_score_grad = relu_backward(hidden_relu_grad, cache["hidden_scores"])
    dW1, db1, flattened_grad = dense_backward(cache["flattened"], W1, hidden_score_grad)
    pooled_maps_grad = flatten_backward(flattened_grad, cache["pooled_maps"].shape)
    relu_out_grad = pool_layer_backward(cache["relu_out"], pooled_maps_grad, pool_size=2, stride=2)
    feature_maps_grad = relu_backward(relu_out_grad, cache["feature_maps"])
    kernels_grad, conv_biases_grad, image_grad = conv_layer_backward(cache["image"], kernels, conv_biases, feature_maps_grad, stride=1, padding=0)

    return kernels_grad, conv_biases_grad, dW1, db1, dW2, db2

def update(kernels, W1, W2, b1, b2, kernels_grad, dW1, dW2, db1, db2, lr, conv_biases, conv_biases_grad):
    W1  = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2  = W2 - lr * dW2
    b2 = b2 - lr * db2
    kernels = kernels - lr * kernels_grad
    conv_biases -= lr * conv_biases_grad
    
    return kernels, conv_biases, W1, b1, W2, b2

def one_hot(label):
    vector = np.zeros(10, dtype=np.float32)
    vector[label] = 1

    return vector

def augment(image):

    if np.random.random() < 0.5:
        return image

    shift_y = np.random.randint(-2, 3)
    shift_x = np.random.randint(-2, 3)

    image = np.roll(image, shift_y, axis=0)
    image = np.roll(image, shift_x, axis=1)

    angle = np.random.uniform(-15, 15)

    image = rotate(
        image,
        angle,
        reshape=False,
        mode="constant",
        cval=0
    )

    return image

image = np.random.randn(28,28)
kernels= np.random.randn(16,3,3).astype(np.float32) * np.sqrt(2/9)
W1 = np.random.randn(2704, 64).astype(np.float32) * np.sqrt(2/2704)
b1 = np.zeros(64, dtype=np.float32)
W2 = np.random.randn(64, 10).astype(np.float32) * np.sqrt(2/64)
b2 = np.zeros(10, dtype=np.float32)
conv_biases = np.zeros(16, dtype=np.float32)
batch_size = 32

train_images = load_mnist_images("dataset/train-images.idx3-ubyte")
train_labels = load_mnist_labels("dataset/train-labels.idx1-ubyte")
test_images = load_mnist_images("dataset/t10k-images.idx3-ubyte")
test_labels = load_mnist_labels("dataset/t10k-labels.idx1-ubyte")
train_labels_one_hot = np.eye(10, dtype=np.float32)[train_labels]

train_images = train_images[:10000]
train_labels = train_labels[:10000]
train_labels_one_hot = train_labels_one_hot[:10000]

best_accuracy = 0

for epoch in range(20):

    total_loss = 0
    correct = 0

    indices = np.random.permutation(len(train_images))

    train_images = train_images[indices]
    train_labels = train_labels[indices]
    train_labels_one_hot = train_labels_one_hot[indices]

    for batch_start in range(0,len(train_images),batch_size):

        batch_images = train_images[batch_start:batch_start+batch_size]
        batch_labels = train_labels[batch_start:batch_start+batch_size]
        batch_one_hot = train_labels_one_hot[batch_start:batch_start+batch_size]

        kernels_grad_sum = np.zeros_like(kernels, dtype=np.float32)
        conv_biases_grad_sum = np.zeros_like(conv_biases, dtype=np.float32)
        dW1_sum = np.zeros_like(W1, dtype=np.float32)
        db1_sum = np.zeros_like(b1, dtype=np.float32)
        dW2_sum = np.zeros_like(W2, dtype=np.float32)
        db2_sum = np.zeros_like(b2, dtype=np.float32)

        batch_loss = 0

        for image, y_true, label in zip(batch_images, batch_one_hot, batch_labels):

            image = augment(image)

            y_pred, cache = forward(image, kernels, conv_biases, W1, b1, W2, b2)
            loss = cross_entropy(y_true, y_pred)
            kernels_grad, conv_biases_grad, dW1, db1, dW2, db2 = backward(y_pred, y_true, cache, kernels, conv_biases, W1, W2)

            kernels_grad_sum += kernels_grad
            conv_biases_grad_sum += conv_biases_grad

            dW1_sum += dW1
            db1_sum += db1

            dW2_sum += dW2
            db2_sum += db2

            prediction = np.argmax(y_pred)
            if prediction == label:
                correct += 1

            batch_loss += loss
        
        actual_batch_size = len(batch_images)

        kernels_grad_sum /= actual_batch_size
        conv_biases_grad_sum /= actual_batch_size

        dW1_sum /= actual_batch_size
        db1_sum /= actual_batch_size

        dW2_sum /= actual_batch_size
        db2_sum /= actual_batch_size

        kernels, conv_biases, W1, b1, W2, b2 = update(kernels, W1, W2, b1, b2, kernels_grad_sum, dW1_sum, dW2_sum, db1_sum, db2_sum, 0.005, conv_biases, conv_biases_grad_sum)

        total_loss += batch_loss / actual_batch_size

    accuracy = correct / len(train_images)

    print(f"Epoch {epoch}: " f"Loss={total_loss:.4f} " f"Accuracy={accuracy:.4f}")

    if epoch % 5 == 0:

        correct = 0

        for image, label in zip(test_images, test_labels):

            y_pred, _ = forward(image,kernels, conv_biases, W1, b1, W2, b2)

            prediction = np.argmax(y_pred)
            if prediction == label:
                correct += 1

        test_accuracy = (correct / len(test_images))

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy

            np.save("weights/kernels.npy", kernels)
            np.save("weights/conv_biases.npy", conv_biases)
            np.save("weights/W1.npy", W1)
            np.save("weights/b1.npy", b1)
            np.save("weights/W2.npy", W2)
            np.save("weights/b2.npy", b2)

            print(
                f"Saved model "
                f"({test_accuracy:.4f})"
            )

        print(f"Test Accuracy: " f"{test_accuracy:.4f}")