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
        self.layers[0].append(node(self.count,value))
        self.count+=1

    def add_hidden_node(self,layer):
        global count
        self.layers[layer].append(node(self.count))
        self.count+=1

    def add_output_node(self):
        global count
        self.layers[len(self.layers)-1].append(node(self.count))
        self.count+=1

    def connect(self,node1,node2,weight):
        connection = Wire(node1,node2,weight)
        node1.connections[0].append(connection)
        node2.connections[1].append(connection)

    def __str__(self):
        return self.layers

class node():
    def __init__(self, index, value = 0):
        self.index = index
        self.value = value
        self.connections = [[],[]]

    def __str__(self):
        return "Value: "+str(self.value)+"   Index :"+str(self.index)+"   Connections: "+str(self.connections)

class Wire():
    def __init__(self, back_node, front_node, weight = 0):
        self.weight = weight
        self.back_node = back_node
        self.front_node = front_node

    def set_weight(self,weight):
        self.weight = weight

    def __str__(self):
        return "Back Node: "+str(back_node.index)+" Front Node: "+str(front_node.index)+" Weight: "+str(weight)
        
    
