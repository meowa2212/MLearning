from network import Network
from trainer import Trainer
from testing import run_realtime_prediction
from data_emnist_numbers import load_data

def main():
    network = Network()
    # network.add_layer(784, 256, "leakyReLu")
    # network.add_layer(256, 256, "leakyReLu")
    # network.add_layer(256, 256, "leakyReLu")
    # network.add_layer(256, 128, "ReLu")
    # network.add_layer(128, 10, "softmax")

    # x_train, y_train = load_data("gzip", "training")
    x_test, y_test = load_data("gzip", "test")
    print(x_test.shape)
    # print(x_train.shape)
    # for i in range(15):
    #     Trainer.fit(network, "cce", (x_train, y_train), 10, 0.05, 128)
    #     score = Trainer.test(network, "cce", (x_test, y_test))
    #     print(score)
    #     # network.save(f"models/digits/digits_e{(i+1)*10}_acc{score:.5f}.pkl")
    #     # print(f"saved at: {(i+1)*10} epochs, loss: {score}")
        
    network.load("models/digits/digits_e800_acc0.04110.pkl")
    run_realtime_prediction(network)
    print(Trainer.test(network, "cce", (x_test, y_test)))
    
if __name__ == "__main__":
    main()
