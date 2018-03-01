import math
import random
import Visualizer

class Neaural_Net:
    """
    A neural net representation with input, hidden and output layers
    """
    def __init__(self):
        self.layers = []
        self.count = 0
        
    def add_layer(self):
        self.layers.append([])

    def add_input_node(self,value):
        self.layers[0].append(Sigmoid_Node(self.count,None,value))
        self.count+=1

    def add_hidden_node(self,layer):
        """
        Adds a hidden node to the specified layer
        """
        global count
        self.layers[layer].append(Sigmoid_Node(self.count,random.uniform(-1, 1)))
        self.count+=1

    def add_output_node(self):
        global count
        self.layers[len(self.layers)-1].append(Node(self.count))
        self.count+=1

    def connect(self,node1,node2,weight):
        """
        Makes a wire between two given nodes with a given weight
        """
        connection = Wire(node1,node2,weight)
        node1.connections[1].append(connection)
        node2.connections[0].append(connection)

    def set_inputs(self,inputs):
        """
        Sets the inputs values for the input layes
        """
        count = 0
        for node in self.layers[0]:
            node.value = inputs[count]
            count+=1

    def __str__(self):
        return str(self.layers)

class Node():
    """
    A simple node with a value and an error
    """
    def __init__(self, index, value = 0):
        self.index = index
        self.value = value
        self.error = 0
        self.connections = [[],[]] #[[inputs],[outputs]]

    def compute_value(self,wires):
        self.value = 0
        for wire in wires:
            self.value += wire.back_node.value * wire.weight

    def __str__(self):
        return "Value: "+str(self.value)+"   Index :"+str(self.index)+"   Connections: "+str(self.connections)

class Sigmoid_Node(Node):
    """
    A sigmoid node with an aditional bias
    """
    def __init__(self,index,bias=0,value=0):
        Node.__init__(self,index,value)
        self.bias = bias
        self.z = None

    def compute_value(self,wires):
        self.z = 0
        for wire in wires:
            self.z += wire.back_node.value * wire.weight
        self.z += self.bias
        self.value = self.sigmoid(self.z)

    def compute_error(self):
        return self.derivative_sigmoid(self.z)

    def sigmoid(self,z):
        return 1/(1+math.exp(-z))

    def derivative_sigmoid(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def __str__(self):
        return "Bias: " + str(self.bias) + " " + Node.__str__(self)

class Wire():
    """
    A connection between two nodes
    """
    def __init__(self, back_node, front_node, weight = 0):
        self.weight = weight
        self.back_node = back_node
        self.front_node = front_node

    def set_weight(self,weight):
        self.weight = weight

    def __str__(self):
        return "Back Node: "+str(self.back_node.index)+" Front Node: "+str(self.front_node.index)+" Weight: "+str(self.weight)
        
    
