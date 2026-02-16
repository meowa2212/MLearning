import numpy as np
import pickle

class Layer():
    def __init__(self, input_count: int, node_count: int, activation: str):
        if activation not in ["ReLu", "tanh", "sigmoid", "leakyReLu", "softmax"]:
            raise ValueError(f"chosen activation {activation} not known [ReLu, tanh, sigmoid, leakyReLu, softmax]")
        self.activation = activation
        if input_count <= 0:
            raise ValueError(f"chosen input count {input_count} is not allowed (int, >0)")
        self.input_count = input_count
        if node_count <= 0:
            raise ValueError(f"chosen node_count {node_count} is not allowed (int, >0)")
        self.node_count = node_count

        match self.activation:
            case "ReLu" | "leakyReLu":
                self.weights = np.random.randn(node_count, input_count) * np.sqrt(2/input_count)
            case "sigmoid" | "tanh":
                self.weights = np.random.randn(node_count, input_count) * np.sqrt(1/input_count)
            case "softmax":
                self.weights = np.random.randn(node_count, input_count) * np.sqrt(2.0 / (input_count + node_count))

        self.biases = np.zeros((node_count, 1))
        
        self.last_input = None
        self.last_output = None
        self.last_activated_output = None

    def calculate(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.shape[0] != self.input_count:
            raise ValueError(f"incorrect number of inputs:{inputs.shape[0]}, inputs:{inputs}")
        
        self.last_input = inputs
        self.last_output = np.dot(self.weights, inputs) + self.biases
        
        match self.activation:
            case "ReLu":
                self.last_activated_output = np.maximum(self.last_output, 0)
            case "tanh":
                self.last_activated_output = np.tanh(self.last_output)
            case "sigmoid":
                self.last_activated_output = 1 / (1 + np.exp(-self.last_output))
            case "leakyReLu":
                self.last_activated_output = np.where(self.last_output > 0, self.last_output, self.last_output * 0.01)
            case "softmax":
                exp_output = np.exp(self.last_output - np.max(self.last_output, axis=0, keepdims=True)) 
                self.last_activated_output = exp_output / np.sum(exp_output, axis=0, keepdims=True)
        
        return self.last_activated_output
    
    def backward(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        match self.activation:
            case "sigmoid":
                delta = output_error * (self.last_activated_output * (1 - self.last_activated_output))
            case "tanh":
                delta = output_error * (1 - self.last_activated_output**2)
            case "ReLu":
                delta = output_error * (self.last_output > 0)
            case "leakyReLu":
                d_relu = np.ones_like(self.last_output)
                d_relu[self.last_output <= 0] = 0.01
                delta = output_error * d_relu
            case "softmax":
                delta = output_error
            case _:
                delta = output_error

        batch_size = self.last_input.shape[1]
        
        dw = np.dot(delta, self.last_input.T) / batch_size
        db = np.sum(delta, axis=1, keepdims=True) / batch_size

        prev_error = np.dot(self.weights.T, delta)

        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db

        return prev_error

class Network():
    def __init__(self):
        self.layers = []

    def add_layer(self, input_count: int, node_count: int, activation: str):
        layer = Layer(input_count, node_count, activation)
        self.layers.append(layer)
    
    def forward_pass(self, inputs: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            inputs = layer.calculate(inputs)
        return inputs
    
    def train(self, x: np.ndarray, y_true: np.ndarray, learning_rate: float, loss_tool: object) -> float:
        y_pred = self.forward_pass(x)
        
        gradient = loss_tool.deriv(y_pred, y_true)

        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)

        return loss_tool.calc(y_pred, y_true)

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self.layers, f)

    def load(self, filename: str):
        with open(filename, 'rb') as f:
            self.layers = pickle.load(f)