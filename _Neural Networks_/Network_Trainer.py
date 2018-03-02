# --------Hopping-------#
import os
import sys
from Net_Approach import *

sys.path.append('../_Core Functions_')
# ----CUSTOM CLASSES-----#
import Extractor
# ---SUPPORT LIBRARIES---#
from PIL import Image
import numpy as np

network = None

def train_images(stop_at=2):
    FOLDER_NAME = "/_Neural Networks_"
    os.chdir('..')
    root = os.getcwd()

    global network
    network = generate_net([28*28, 28, 10], np.zeros((28*28), dtype=int))

    for x in range(10):
        os.chdir(root +"/train_images_sorted/" + str(x))
        i = 0
        for filename in os.listdir(os.getcwd()):
            i += 1
            image = Extractor.getImage(filename)
            matrix = Extractor.ImageToMatrix(image)
            #print(matrix)
            data = np.asarray(matrix).flatten()
            #print(data)
            #exit(0)
            data = scale_data(data)
            run_net(data, x)
            if i == stop_at:
                break
            #for m in range(10):
            #    print(network.layers[-1][m].value)
        print("Digit " + str(x) + " is complete")


def scale_data(grayscale):
    """
    grayscale 255 is -1, 0 is + 1
    """
    r = np.zeros((len(grayscale)), dtype=float)
    for i in range(len(grayscale)):
        r[i] = grayscale[i] / 255
    #print(grayscale)
    #print(r)
    #exit(0)
    return r


def run_net(scaled_grayscale, correct_answer):

    global network

    network.set_inputs(scaled_grayscale)
    network = propagate(network)

    correct_array = np.zeros((10,), dtype=int)
    correct_array[correct_answer] = 1
    #print(correct_array)
    network = output_error(network, correct_array)

    network = hidden_error(network)
    network = improve_bias(network)
    network = improve_weights(network)

    #show_net(network,1,2)

def run_test(scaled_grayscale):
    global network

    network.set_inputs(scaled_grayscale)
    network = propagate(network)
    maximum = network.layers[-1][0].value
    index = 0
    count = 0
    for node in network.layers[-1]:
        #print(node.value)
        if node.value > maximum:
            maximum = node.value
            index = count
        count += 1
    #print("_____________")
    print(index)
    #print("_____________")

if __name__ == "__main__":
    train_images(1)
    print("done training")
    #print(os.getcwd())
    os.chdir('..')
    os.chdir('..')
    #print(os.getcwd())
    os.chdir(os.getcwd()+'/test_images')
    #print(os.getcwd())
    stop = 0
    for filename in os.listdir(os.getcwd()):
        if stop == 20:
            break
        image = Extractor.getImage(filename)
        matrix = Extractor.ImageToMatrix(image)
        data = np.asarray(matrix).flatten()
        data = scale_data(data)
        run_test(data)
        stop += 1
    
