def read_training(filename="train-images.idx3-ubyte"):

    file = open(filename, encoding="rb")

    with open('train-labels-idx1-ubyte', 'rb') as f:
        bytes = f.read(8)
        print(bytes)


if __name__ == "__main__":
    read_training()
