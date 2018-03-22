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

def train_images(size,num,learning_rate,layers):
    """
    Trains the neural net for num batches of size number of images in each
    """
    count = 0
    os.chdir('..')
    root = os.getcwd()
    answer_array = read_answers()
    average_error = 0
    global batches
    global network
    if network is None:
        builder = Builder(np.array(layers))
        network = builder.generate_net()
    os.chdir(root +"/train_images")
    for b in range(num):
        for s in range(size):   #one loop is a batch
            network, img_num = load_image(network,s)
            #Transform an answer like 3 into [0,0,0,1,0,0,0,0,0,0]
            correct_array = np.zeros((10,), dtype=int)
            correct_array[int(answer_array[img_num-1])] = 1
            #print(correct_array)
            #Propagate input forward
            network = propagate(network)
            #Add up all the errors for each output node for size number of iterations
            network = output_error(network,correct_array)
            #average_error += total_error
            network = hidden_error(network)
        
        #print(average_error/size)
        #print(correct_array)
        #network.show(3,3)
        #print(network.weight_errors[0][0])
        ##print(network.error_layers)
        network = improve_bias(network,learning_rate,size)
        network = improve_weights(network,learning_rate,size)
        network.delta_layers = [np.zeros(b.shape) for b in network.delta_layers]
        network.weight_errors = [np.zeros(w.shape) for w in network.weight_errors]
        average_error = 0
        batches += 1
        #print(network.error_layers[-1])

    return network

def read_answers():
    """
    Creates an array full of all the answers for test dataset
    """
    answer_array = []
    answers = open("mnist-train-labels.txt", "r")
    #puts all the answers in an array
    index = 0
    for line in answers:
        answer_array.append(int(line.strip()))
    return answer_array

def load_image(network,s):
    """
    Generates a random number between 1 and 60000 and set's the network's input
    """
    img_num = random.randint(1,60000)
    num_zeros = 5-len(str(img_num))
    zeros = '0'*num_zeros
    filename = zeros+str(img_num)+".tif"
    #img_num = 11534
    #filename = str(img_num)+".tif"
    
    #print(filename)
    image = Extractor.getImage(filename)
    matrix = Extractor.ImageToMatrix(image)
    data = np.asarray(matrix).flatten()
    """
    count = 0
    for item in data:
        if count>=28:
            count = 0
            print()
        print(str(item)+(3-len(str(item)))*" "+" ",end="")
        count+=1
    print()
    print()
    """
    data = scale_data(data)
    #print()
    """
    count = 0
    for item in data:
        if count>=28:
            count = 0
            print()
        print(str(round(item,1))+(3-len(str(item)))*" "+" ",end="")
        count+=1
    print()
    print()
    """
    #exit()
    
    network.set_inputs(data)
    return (network,img_num)

def scale_data(grayscale):
    """
    grayscale 255 is 0, 0 is +1
    """
    r = np.zeros((len(grayscale)), dtype=float)
    for i in range(len(grayscale)):
        result = 1 - (grayscale[i] / 255)
        r[i] = result
        """
        if result > 0.1:
            r[i] = 1
        else:
            r[i] = 0
        """
    return r.tolist()

def improve_net(network,learning_rate,size):
    """
    Runs 3 parts of the back propagation algorithm
    """
    network = hidden_error(network)
    network = improve_bias(network,learning_rate,size)
    network = improve_weights(network,learning_rate,size)
    return network

def run_test(network,scaled_grayscale):
    """
    Goes through all the outputs of the network
    and determines the best prediction
    """
    network.set_inputs(scaled_grayscale)
    network = propagate(network)
    maximum = network.activation_layers[-1][0]
    index = 0
    count = 0
    for node in network.activation_layers[-1]:
        if node > maximum:
            maximum = node
            index = count
        count += 1
    return index

if __name__ == "__main__":
    reset = False
    train = False
    test = True
    stop_at = 10000
    size = 100
    num = 100
    learning_rate = 3
    
    if reset:
        if os.path.isfile("network.p"):
            os.remove("network.p")
    
    if Path("network.p").is_file():
        (network,batches) = pickle.load(open("network.p", "rb"))

    if train:
        network = train_images(size,num,learning_rate,[28*28,30,10])
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
        
            
    
