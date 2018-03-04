# --------Hopping-------#
import os
import sys
import pickle
from pathlib import Path
from Net_Approach import *

sys.path.append('../_Core Functions_')
# ----CUSTOM CLASSES-----#
import Extractor
# ---SUPPORT LIBRARIES---#
from PIL import Image
import numpy as np

network = None
batches = 0

def train_images(size,num):
    count = 0
    error_layer = [0,0,0,0,0,0,0,0,0,0]
    os.chdir('..')
    root = os.getcwd()
    answer_array = read_answers()

    global batches
    iteration = 0

    global network
    if network is None:
        network = generate_net([28*28,20,20, 10], np.zeros((28*28), dtype=int))
    os.chdir(root +"/train_images")
    for filename in os.listdir(os.getcwd()):
        #Read values from test image
        image = Extractor.getImage(filename)
        matrix = Extractor.ImageToMatrix(image)
        data = np.asarray(matrix).flatten()
        data = scale_data(data)
        #Transform an answer like 3 into [0,0,0,1,0,0,0,0,0,0]
        correct_array = np.zeros((10,), dtype=int)
        correct_array[int(answer_array[count])] = 1
        #Propagate input forward
        network = run_net(network,data)
        #Add up all the errors for each output node for size number of iterations
        for i in range(len(error_layer)):
            error_layer[i] += (network.layers[-1][i].value-correct_array[i])*network.layers[-1][i].compute_error()

        # If we finished a batch
        if count%size == 0 and count!=0:
            total_error = 0
            #Sets the averaged error per output node
            for i in range(len(network.layers[-1])):
                total_error+=error_layer[i]
                #Set the output error to the average error of that node for entire batch
                network.layers[-1][i].error = error_layer[i]/size
            print("Total Error: "+str(total_error)+" batch:"+str(batches + 1))

            #Improve the net based on the output error
            network = improve_net(network)

            #Reinitialize values for next batch
            count = 0
            batches += 1
            iteration += 1
            error_layer = [0,0,0,0,0,0,0,0,0,0]

            if iteration >= num:
                return network
        else:
            count+=1

    return network

def read_answers():
    """
    Creates an array full of all the answers for test dataset
    """
    answer_array = []
    answers = open("mnist-train-labels.txt", "r")
    
    index = 0
    for line in answers:
        answer_array.append(int(line.strip()))
    return answer_array

def scale_data(grayscale):
    """
    grayscale 255 is -1, 0 is + 1
    """
    r = np.zeros((len(grayscale)), dtype=float)
    for i in range(len(grayscale)):
        r[i] = grayscale[i] / 255
    return r

def run_net(network, scaled_grayscale):
    network.set_inputs(scaled_grayscale)
    network = propagate(network)
    return network

def improve_net(network):
    network = hidden_error(network)
    network = improve_bias(network,0.5)
    network = improve_weights(network,0.5)
    return network

def run_test(network,scaled_grayscale):
    network.set_inputs(scaled_grayscale)
    network = propagate(network)
    maximum = network.layers[-1][0].value
    index = 0
    count = 0
    for node in network.layers[-1]:
        if node.value > maximum:
            maximum = node.value
            index = count
        count += 1
    print(index)

if __name__ == "__main__":
    if Path("network.p").is_file():
        (network,batches) = pickle.load(open("network.p", "rb"))
    network = train_images(100,5)
    print("done training")

    os.chdir('..')
    os.chdir(os.getcwd() + "/_Neural Networks_")
    pickle.dump((network,batches), open("network.p", "wb"))

    # os.chdir('..')
    # os.chdir(os.getcwd()+'/test_images')
    # for filename in os.listdir(os.getcwd()):
    #     image = Extractor.getImage(filename)
    #     matrix = Extractor.ImageToMatrix(image)
    #     data = np.asarray(matrix).flatten()
    #     data = scale_data(data)
    #     run_test(network,data)
    
