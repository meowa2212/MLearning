from random import uniform, normalvariate
from math import tanh, exp

'''
TO DO:
implement backpropagation

or

continue with evolutionary
'''

class Neuron():
    def __init__(self, input_count: int):
        self.input_count = input_count
        self.weights = [0 for x in range(self.input_count)]
        self.bias = 0
    
    def calculate(self, inputs: list) -> float:
        value = 0
        for i in range(self.input_count):
            value += inputs[i]*self.weights[i]
        return value + self.bias
    
    def randomize(self, mutation: float, draft: float):
        self.weights = [x*normalvariate(1, mutation) + normalvariate(0, mutation) if uniform(0,1) <= draft else x for x in self.weights ]
        if uniform(0,1) < draft:
            self.bias = self.bias*normalvariate(1, mutation) + normalvariate(0, mutation)


class Layer():
    def __init__(self, neuron_count: int, activation: str, input_count: int):
        self.neuron_count = neuron_count
        self.input_count = input_count

        if activation not in ["ReLu", "tanh", "sigmoid", "leakyReLu"]:
            raise ValueError(f"chosen activation {activation} not known [ReLu, tanh, sigmoid, leakyReLu]")
        self.activation = activation

        self.neurons = []
        for i in range(self.neuron_count):
            neuron = Neuron(input_count)
            self.neurons.append(neuron)
    
    def randomize(self, mutation: float, draft: float):
        [x.randomize(mutation, draft) for x in self.neurons]

    def calculate(self, inputs: list) -> list:
        if len(inputs) != self.input_count:
            raise ValueError("Invalid amount of input signals to layer")
        outputs = [x.calculate(inputs) for x in self.neurons]
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
        
    def randomize(self, mutation: float, draft: float):
        [x.randomize(mutation, draft) for x in self.layers]
    
    def calculate(self, inputs: list) -> list:
        if len(inputs) != self.input_count:
            raise ValueError("Invalid amount of input signals to network")
        for layer in self.layers:
            inputs = layer.calculate(inputs)
        return inputs

def main():
    test = Network(2, [(2, "sigm"), (3, "sigm"), (2, "tanh")])

    test.randomize(0.3, 0.5)
    print(test.calculate([0.5,-2]))
    print(test.calculate([1,3]))

    test.randomize(0.3, 0.5)
    print(test.calculate([0.5,-2]))
    print(test.calculate([1,3]))
    

if __name__ == "__main__":
    main()