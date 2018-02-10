import math
import random
import Visualizer

class Neaural_Net:
    
    def __init__(self):
        self.layers = []
        self.count = 0
        
    def add_layer(self):
        self.layers.append([])

    def add_input_node(self,value):
        self.layers[0].append(Sigmoid_Node(self.count,None,value))
        self.count+=1

    def add_hidden_node(self,layer):
        global count
        self.layers[layer].append(Sigmoid_Node(self.count,random.uniform(-1, 1)))
        self.count+=1

    def add_output_node(self):
        global count
        self.layers[len(self.layers)-1].append(Sigmoid_Node(self.count,random.uniform(-1, 1)))
        self.count+=1

    def connect(self,node1,node2,weight):
        connection = Wire(node1,node2,weight)
        node1.connections[1].append(connection)
        node2.connections[0].append(connection)

    def __str__(self):
        return str(self.layers)

class Node():
    def __init__(self, index, value = 0):
        self.index = index
        self.value = value
        self.error = 0
        self.connections = [[],[]] #[[inputs],[outputs]]


    def compute_value(self,wires):
        for wire in wires:
            self.value += wire.back_node.value * wire.weight
        self.value = round(self.value, 2)

    def __str__(self):
        return "Value: "+str(self.value)+"   Index :"+str(self.index)+"   Connections: "+str(self.connections)

class Sigmoid_Node(Node):
    def __init__(self,index,bias=0,value=0):
        Node.__init__(self,index,value)
        self.bias = bias
        self.z = None

    def compute_value(self,wires):
        self.z = 0
        for wire in wires:
            self.z += wire.back_node.value * wire.weight
        self.z += self.bias
        #print(self.z)
        return self.sigmoid(self.z)

    def sigmoid(self,z):
        return 1/(1+math.exp(-self.z))

    def __str__(self):
        return "Bias: " + str(self.bias) + " " + Node.__str__(self)

class Wire():
    def __init__(self, back_node, front_node, weight = 0):
        self.weight = weight
        self.back_node = back_node
        self.front_node = front_node

    def set_weight(self,weight):
        self.weight = weight

    def __str__(self):
        return "Back Node: "+str(self.back_node.index)+" Front Node: "+str(self.front_node.index)+" Weight: "+str(self.weight)
        
    
