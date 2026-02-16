import numpy as np
from data import load_data
from loss import Loss

class Trainer():
    def fit(network: object, loss: str, data_filename: str, epochs: int, learning_rate: float, batch: int = 32) :
        images, labels = load_data(data_filename, "training")
        
        num_samples = images.shape[0]
        input_dim = images.shape[1]
        label_dim = labels.shape[1]

        images_ready = images.reshape(num_samples, input_dim).T
        labels_ready = labels.reshape(num_samples, label_dim).T

        match loss:
            case "mse": loss_tool = Loss.MSE()
            case "mae": loss_tool = Loss.MAE()
            case "bce": loss_tool = Loss.BCE()
            case "cce": loss_tool = Loss.CCE()

        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            images_shuffled = images_ready[:, indices]
            labels_shuffled = labels_ready[:, indices]
            
            total_loss = 0
            for i in range(0, num_samples, batch):
                batch_x = images_shuffled[:, i : i + batch]
                batch_y = labels_shuffled[:, i : i + batch]
                
                batch_loss = network.train(batch_x, batch_y, learning_rate, loss_tool)
                total_loss += batch_loss
            
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / (num_samples / batch):.4f}")

    def test(network: object, loss: str, data_filename: str):
        images, labels = load_data(data_filename, "test")

        num_samples = images.shape[0]
        input_dim = images.shape[1]
        label_dim = labels.shape[1]
        
        images_ready = images.reshape(num_samples, input_dim).T
        labels_ready = labels.reshape(num_samples, label_dim).T

        match loss:
            case "mse": loss_tool = Loss.MSE()
            case "mae": loss_tool = Loss.MAE()
            case "bce": loss_tool = Loss.BCE()
            case "cce": loss_tool = Loss.CCE()

        y_pred = network.forward_pass(images_ready)
        
        score = loss_tool.calc(y_pred, labels_ready)
        
        predictions = np.argmax(y_pred, axis=0)
        truth = np.argmax(labels_ready, axis=0)
        accuracy = np.mean(predictions == truth)

        print(f"Loss: {score:.4f}")
        print(f"Accuracy: {accuracy * 100:.2f}%")