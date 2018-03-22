import math
import random
import Visualizer
import numpy as np

class Neaural_Net:
    """
    A neural net representation with input, hidden and output layers
    """
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes  #format of the net
        self.bias_layers = []           #array of bias in the net
        self.weight_layers = []         #array of weights in the net
        self.z_layers = []              #array of inputs into a node
        self.activation_layers = []     #array of values in the net

        self.delta_layers = []          #array of errors on a node
        self.weight_errors = []         #array of errors on a connection

    def set_inputs(self,inputs):
        """
        Sets an input's value
        """
        self.activation_layers[0] = np.array(inputs)
        
    def add_node_layer(self, values, biases):
        """
        Sets an input's value, bias and error
        """
        self.activation_layers.append(values)
        self.bias_layers.append(biases)
        self.z_layers.append(np.zeros(len(values), dtype=float))
        self.delta_layers.append(np.zeros(len(values), dtype=float))

    def add_connection_layer(self, data):
        """
        Creates an array of weights
        """
        self.weight_layers.append(data)
        self.weight_errors.append(np.zeros(data.shape))

    def compute_value(self, l):
        """
        Calculating sigmoid function sum on a node
        """
        self.z_layers[l] = np.dot(self.weight_layers[l-1],self.activation_layers[l-1])
        self.activation_layers[l] = sigmoid(self.z_layers[l]+self.bias_layers[l])
            
        
    def show(self,start,end):
        """
        Runs the visualizer on the net

        [0,1,2,3,4,5,6,7]

        net.show(3,5) shows layers 3,4,5
        
        """
        layer_sizes = []
        nodes = []
        for layer in self.activation_layers[start:end+1]:
            layer_sizes.append(len(layer))
            for node in layer:
                nodes.append(round(node,3))
        weights = []
        for layer in self.weight_layers[start:end]:
            layer_t = layer.transpose()
            for output_node in layer_t:
                for connection in output_node:
                    weights.append(connection)
        Visualizer.draw_neural_net(0.1, 1, 0, 1, layer_sizes, nodes, weights)
    

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))



