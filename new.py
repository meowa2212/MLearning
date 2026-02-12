
from random import uniform, normalvariate
from math import tanh, exp

'''
TO DO:
implement backpropagation
'''


class Layer():
    def __init__(self, neuron_count: int, activation: str, input_count: int):
        self.neuron_count = neuron_count
        self.input_count = input_count

        if activation not in ["ReLu", "tanh", "sigmoid", "leakyReLu"]:
            raise ValueError(f"chosen activation {activation} not known [ReLu, tanh, sigmoid, leakyReLu]")
        self.activation = activation

        self.weights = []
        self.biases = []
        for i in range(self.neuron_count):
            self.weights.append([uniform(0, 1) for x in range(self.input_count)])
            self.biases.append(uniform(0, 1))

    def calculate(self, inputs: list) -> list:
        if len(inputs) != self.input_count:
            raise ValueError("Invalid amount of input signals to layer")

        outputs = [[inputs[i]*self.weights[node][i] for i in range(self.input_count)] for node in range(self.neuron_count)]
        outputs = [sum([x for x in outputs[i]])+self.biases[i] for i in range(len(outputs))]
        match self.activation:
            case "ReLu":
                return [max(0, x) for x in outputs]
            case "tanh":
                return [tanh(x) for x in outputs]
            case "sigmoid":
                return [1 / (1 + exp(-x)) for x in outputs]
            case "leakyReLu":
                return [x if x>=0 else 0.01*x for x in outputs]

# layers_struct = [(3, "tanh"), (2, "relu"), (3, "sigm")]

class Network():
    def __init__(self, input_count: int, layers_struct: list):
        self.input_count = input_count
        if input_count <= 0:
            raise ValueError(f"Wrong connection count: {input_count}!")
        self.layers = [Layer(layers_struct[0][0], layers_struct[0][1], input_count),]
        for i in range(1, len(layers_struct)):
            self.layers.append(Layer(layers_struct[i][0], layers_struct[i][1], layers_struct[i-1][0]))
    
    def calculate(self, inputs: list) -> list:
        if len(inputs) != self.input_count:
            raise ValueError("Invalid amount of input signals to network")
        for layer in self.layers:
            inputs = layer.calculate(inputs)
        return inputs

def main():
    test = Network(2, [(2, "ReLu")])
    [[print((x.weights[i], x.biases[i]))for i in range(x.neuron_count)] for x in test.layers]
    print(test.calculate([0.5,-2]))
    # print(test.calculate([1,3]))

    # test2 = Network(2, [(2, "sigmoid"), (3, "sigmoid"), (2, "tanh")])
    # print(test2.calculate([0.5,-2]))
    # print(test2.calculate([1,3]))
    

if __name__ == "__main__":
    main()