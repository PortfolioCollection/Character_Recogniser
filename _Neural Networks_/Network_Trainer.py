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
    global network
    if network is None:
        builder = Builder(np.array([28*28,20,20,10]))
        network = builder.generate_net()
    os.chdir(root +"/train_images")
    for b in range(num):
        for s in range(size):
            #Read values from random test image
            img_num = random.randint(1,60000)
            num_zeros = 5-len(str(img_num))
            zeros = '0'*num_zeros
            filename = zeros+str(img_num)+".tif"
            image = Extractor.getImage(filename)
            matrix = Extractor.ImageToMatrix(image)
            data = np.asarray(matrix).flatten()
            data = scale_data(data)
            network.set_inputs(data)
            #Transform an answer like 3 into [0,0,0,1,0,0,0,0,0,0]
            correct_array = np.zeros((10,), dtype=int)
            correct_array[int(answer_array[img_num-1])] = 1
            #Propagate input forward
            network = propagate(network)
            #Add up all the errors for each output node for size number of iterations
            network, total_error = output_error(network,correct_array)
            for i in range(len(error_layer)):
                error_layer[i] += network.error_layers[-1][i]
            
        total_error = 0
        #Sets the averaged error per output node
        for i in range(len(error_layer)):
            total_error+=error_layer[i]
            #Set the output error to the average error of that node for entire batch
            network.error_layers[-1][i] = error_layer[i]
        print("Total Error: "+str(total_error)+" batch:"+str(batches + 1))
        #Improve the net based on the output error
        network = improve_net(network,1,size)

        #Reinitialize values for next batch
        batches += 1
        error_layer = [0,0,0,0,0,0,0,0,0,0]
        #print(img_num)
        #network.show(3,3)


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

def improve_net(network,learning_rate,size):
    network = hidden_error(network)
    network = improve_bias(network,learning_rate,size)
    network = improve_weights(network,learning_rate,size)
    return network

def run_test(network,scaled_grayscale):
    network.set_inputs(scaled_grayscale)
    network = propagate(network)
    maximum = network.node_layers[-1][0]
    index = 0
    count = 0
    for node in network.node_layers[-1]:
        if node > maximum:
            maximum = node
            index = count
        count += 1
    return index

if __name__ == "__main__":
    reset = False
    train = True
    test = True
    stop_at = 1000
    
    if reset:
        if os.path.isfile("network.p"):
            os.remove("network.p")
    
    if Path("network.p").is_file():
        (network,batches) = pickle.load(open("network.p", "rb"))

    if train:
        network = train_images(100,20)
        print("done training")
        os.chdir('..')
        os.chdir(os.getcwd() + "/_Neural Networks_")
        pickle.dump((network,batches), open("network.p", "wb"))

    if test:
        correct = 0
        total = 0
        percent = 0
        os.chdir('..')
        answer_array = []
        answers = open("mnist-test-labels.txt", "r")
        index = 0
        for line in answers:
            answer_array.append(int(line.strip()))
        os.chdir(os.getcwd()+'/test_images')
        for filename in os.listdir(os.getcwd()):
            image = Extractor.getImage(filename)
            matrix = Extractor.ImageToMatrix(image)
            data = np.asarray(matrix).flatten()
            data = scale_data(data)
            #print(answer_array[total])
            if run_test(network,data) == answer_array[total]:
                correct+=1
            total+=1
            if total %((10000)/100) == 0:
                percent+=1
                print(str(percent)+"%")
                
            
            #network.show(2,2)
            if total == stop_at:
                break
        print(str(correct)+"/"+str(total))
        
            
    
