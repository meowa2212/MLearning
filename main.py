from network import Network
from trainer import Trainer
from testing import run_realtime_prediction

def main():
    network = Network()
    # network.add_layer(784, 256, "leakyReLu")
    # network.add_layer(256, 256, "leakyReLu")
    # network.add_layer(256, 128, "ReLu")
    # network.add_layer(128, 10, "softmax")
    # Trainer.fit(network, "cce", "gzip", 500, 0.05, 128)
    # Trainer.test(network, "cce", "gzip")
    # if input("save?: ") == "y":
    #     network.save("digit.pkl")
    network.load("digit.pkl")
    run_realtime_prediction(network)
    
if __name__ == "__main__":
    main()