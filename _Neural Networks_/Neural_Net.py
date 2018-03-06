import math
import random
import Visualizer
import numpy as np

class Neaural_Net:
    """
    A neural net representation with input, hidden and output layers
    """
    def __init__(self, num_layers):
        self.layer_sizes = []
        self.node_layers = []
        self.bias_layers = []
        self.connection_layers = []
        self.z_layers = []
        self.error_layers = []

    def set_inputs(self,inputs):
        self.node_layers[0] = np.array(inputs)
        
    def add_node_layer(self, values, biases, layer_num):
        self.node_layers.append(values)
        self.bias_layers.append(biases)
        self.z_layers.append(np.zeros(len(values)))
        self.error_layers.append(np.zeros(len(values)))

    def add_connection_layer(self, data, layer_num):
        self.connection_layers.append(data)

    def compute_value(self, layer, node):
        z = 0
        for c in range(len(self.connection_layers[layer-1][node])):
            z += self.node_layers[layer-1][c]*self.connection_layers[layer-1][node][c]
        z+=self.bias_layers[layer][node]
        self.z_layers[layer][node] = z
        self.node_layers[layer][node] = sigmoid(z)

    
    def __str__(self):
        return str(self.node_layers)

    def show(self,start,end):
        """
        Runs the visualizer on the net

        [0,1,2,3,4,5,6,7]

        net.show(3,5) shows layers 3,4,5
        
        """
        layer_sizes = []
        nodes = []
        for layer in self.node_layers[start:end+1]:
            layer_sizes.append(len(layer))
            for node in layer:
                nodes.append(node)
        weights = []
        for layer in self.connection_layers[start:end]:
            layer_t = layer.transpose()
            for output_node in layer_t:
                for connection in output_node:
                    weights.append(connection)
        Visualizer.draw_neural_net(0.1, 1, 0, 1, layer_sizes, nodes, weights)
    

def sigmoid(z):
    return 1/(1+math.exp(-z))

def derivative_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

