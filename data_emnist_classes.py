import os
import numpy as np
import gzip
import matplotlib.pyplot as plt

def load_data(directory: str, choice: str) -> tuple:
    file_map = {
        "training": ("emnist-byclass-train-images-idx3-ubyte.gz", "emnist-byclass-train-labels-idx1-ubyte.gz"),
        "test": ("emnist-byclass-test-images-idx3-ubyte.gz", "emnist-byclass-test-labels-idx1-ubyte.gz")
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
        one_hot_labels = np.zeros((num_samples, 62))
        one_hot_labels[np.arange(num_samples), raw_labels] = 1

    # 3. GLOBAL SHUFFLE (Crucial for EMNIST)
    # This prevents the model from seeing 28,000 zeros, then 28,000 ones, etc.
    indices = np.random.permutation(num_samples)
    
    return images[indices], one_hot_labels[indices]

def get_label(label: np.ndarray) -> str:
    emnist_labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    return emnist_labels[np.argmax(label)]


def main():
    images, labels = load_data("gzip", "training")
    display(images[0], get_label(labels[0]))


def display(image: np.ndarray, title: str) -> None:
    reshaped = image.reshape(28,28)
    plt.imshow(reshaped, cmap="gray")
    plt.colorbar()
    plt.axis("off")
    plt.title(title)
    plt.pause(3)
    plt.close("all")
    

if __name__ == "__main__":
    main()