import gzip
import numpy as np
import os
import matplotlib.pyplot as plt

def load_data(directory: str, choice: str) -> tuple:
    file_map = {
        "training": ("emnist-digits-train-images-idx3-ubyte.gz", "emnist-digits-train-labels-idx1-ubyte.gz"),
        "test": ("emnist-digits-test-images-idx3-ubyte.gz", "emnist-digits-test-labels-idx1-ubyte.gz")
    }
    
    img_file, lbl_file = file_map[choice]
    img_path = os.path.join(directory, img_file)
    lbl_path = os.path.join(directory, lbl_file)

    # 1. Load ALL Images
    with gzip.open(img_path, 'rb') as f:
        # We read the whole buffer to ensure nothing is left behind
        data = f.read()
        # The IDX header is 16 bytes. We load EVERYTHING after that.
        raw_images = np.frombuffer(data, np.uint8, offset=16)
        
        # Reshape to 28x28 to fix the EMNIST transposition quirk
        images = raw_images.reshape(-1, 28, 28)
        # Fix orientation (Swap axis 1 and 2)
        images = np.transpose(images, (0, 2, 1)) 
        # Flatten for your Trainer's .T operation
        images = images.reshape(-1, 784).astype(np.float32) / 255.0

    # 2. Load ALL Labels
    with gzip.open(lbl_path, 'rb') as f:
        data = f.read()
        # The IDX header for labels is 8 bytes
        raw_labels = np.frombuffer(data, np.uint8, offset=8)
        
        # One-Hot Encode for 10 digits
        num_samples = raw_labels.shape[0]
        one_hot_labels = np.zeros((num_samples, 10))
        one_hot_labels[np.arange(num_samples), raw_labels] = 1

    # 3. GLOBAL SHUFFLE (Crucial for EMNIST)
    # This prevents the model from seeing 28,000 zeros, then 28,000 ones, etc.
    indices = np.random.permutation(num_samples)
    
    return images[indices], one_hot_labels[indices]


def display(image: np.ndarray, title: np.ndarray) -> None:
    reshaped = image.reshape(28,28)
    plt.imshow(reshaped, cmap="gray")
    plt.colorbar()
    plt.axis("off")
    plt.title(np.argmax(title))
    plt.pause(3)
    plt.close("all")
# Verify the count
# images, labels = load_data("gzip", "training")
# print(f"Successfully loaded {images.shape[0]} images.") 
# This should print 240,000 for training or 40,000 for test (Digits split).