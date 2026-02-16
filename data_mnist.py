import numpy as np

def load_data(filename: str, choice: str) -> tuple:
    with np.load(filename) as data:
        match choice:
            case "training":
                training_images = data["training_images"]
                training_labels = data["training_labels"]
                return training_images, training_labels

            case "test":
                test_images = data["test_images"]
                test_labels = data["test_labels"]
                return test_images, test_labels

            case "validation":
                validation_images = data["validation_images"]
                validation_labels = data["validation_labels"]
                return validation_images, validation_labels
