from network import Network
from trainer import Trainer
from testing import run_realtime_prediction, plot_loss

def main():
    network = Network()
    network.add_layer(784, 256, "leakyReLu")
    network.add_layer(256, 128, "leakyReLu")
    network.add_layer(128, 10, "softmax")
    Trainer.fit(network, "cce", "mnist.npz", 50, 0.01, 64)
    Trainer.test(network, "cce", "mnist.npz")
    if input("save?: ") == "y":
        network.save("digit.pkl")
    network.load("digit.pkl")
    Trainer.test(network, "cce", "mnist.npz")
    run_realtime_prediction(network)
    
if __name__ == "__main__":
    main()