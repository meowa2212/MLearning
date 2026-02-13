class Loss():
    class MSE():
        def calc():
            print("calculate mse")
        def deriv():
            print("derivative mse")
    class BCE():
        def calc():
            print("calculate bce")
        def deriv():
            print("derivative bce")


def main():
    Loss.MSE.calc()
    Loss.BCE.deriv()

if __name__ == "__main__":
    main()