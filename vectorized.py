'''
TO DO: Implement Loss Functions
-------------------------------------------------------------------------
1. Mean Squared Error (MSE) - For Regression (Standard)
   - Loss: np.mean((y_pred - y_true)**2)
   - Gradient: 2 * (y_pred - y_true) / y_true.size

2. Mean Absolute Error (MAE) - For Regression (Robust to Outliers)
   - Loss: np.mean(np.abs(y_pred - y_true))
   - Gradient: np.sign(y_pred - y_true) / y_true.size

3. Binary Cross-Entropy (BCE) - For Binary Classification (use with Sigmoid)
   - Note: Clip y_pred to (1e-15, 1 - 1e-15) to avoid log(0) errors.
   - Loss: -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
   - Gradient: (y_pred - y_true) / (y_pred * (1 - y_pred))

4. Categorical Cross-Entropy (CCE) - For Multi-class (use with Softmax)
   - Loss: -np.sum(y_true * np.log(y_pred + 1e-15))
   - Gradient Shortcut: If paired with Softmax output, the chain rule 
     simplifies the layer's error to: (y_pred - y_true)

5. Backpropagation Implementation
   - Store 'self.last_input' and 'self.last_z' during Forward pass.
   - Use 'np.dot(delta, self.last_input.T)' for weight gradients (dW).
   - Use 'np.dot(self.weights.T, delta)' to pass error to previous layer.
-------------------------------------------------------------------------
'''


import numpy as np

class Layer():
    def __init__(self, node_count: int, activation: str, input_count: int):
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
            case "ReLu":
                self.weights = np.random.randn(node_count, input_count) * np.sqrt(2/input_count)
            case "leakyReLu":
                self.weights = np.random.randn(node_count, input_count) * np.sqrt(2/input_count)
            case "sigmoid":
                self.weights = np.random.randn(node_count, input_count) * np.sqrt(1/input_count)
            case "tanh":
                self.weights = np.random.randn(node_count, input_count) * np.sqrt(1/input_count)
            case "softmax":
                self.weights = np.random.randn(node_count, input_count) * np.sqrt(2.0 / (input_count + node_count))

        self.biases = np.zeros((node_count, 1))

    def calculate(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.shape[0] != self.input_count:
            raise ValueError(f"incorrect number of inputs:{inputs.shape[0]}, inputs:{inputs}")
        output = np.dot(self.weights, inputs) + self.biases
        match self.activation:
            case "ReLu":
                return np.maximum(output, 0)
            case "tanh":
                return np.tanh(output)
            case "sigmoid":
                return 1 / (1 + np.exp(-output))
            case "leakyReLu":
                return np.where(output > 0, output, output*0.01)
            case "softmax":
                exp_output = np.exp(output - np.max(output)) 
                return exp_output / np.sum(exp_output, axis=0)

# layers_struct = [(3, "tanh"), (2, "relu"), (3, "sigm")]

class Network():
    def __init__(self, input_count: int, layers_struct: list):
        self.input_count = input_count
        self.layers = [Layer(layers_struct[0][0], layers_struct[0][1], input_count),]
        for i in range(1, len(layers_struct)):
            self.layers.append(Layer(layers_struct[i][0], layers_struct[i][1], layers_struct[i-1][0]))
    
    def forward_pass(self, inputs: list) -> list:
        for layer in self.layers:
            inputs = layer.calculate(inputs)
        return inputs
    
    def present_network(self):
        for x in self.layers:
            print(f"Layer, a: {x.activation}")
            [print(f"w: {x.weights[i]}, b: {x.biases[i]}")for i in range(x.node_count)]

def prepare_input(inputs: list) -> np.ndarray:
    arraylike = []
    for element in inputs:
        arraylike.append([element,])
    return np.array(arraylike)

def main():
    test = Network(8, [(32, "leakyReLu"), (16,"ReLu"), (8, "ReLu"), (4, "softmax")])
    inputs = prepare_input([1, 3.14, 4, -2, 0.5, 1.67, -0.37, 2.8])
    results = test.forward_pass(inputs)
    print(results)
    print(np.sum(results))
    
    # print(test.calculate([1,3]))
    # test2 = Network(2, [(2, "sigmoid"), (3, "sigmoid"), (2, "tanh")])
    # print(test2.calculate([0.5,-2]))
    # print(test2.calculate([1,3]))
    

if __name__ == "__main__":
    main()