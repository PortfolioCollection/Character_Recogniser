import math
import random
import Visualizer


class Node:
    def __init__(self, value, index, inputs, in_weights, outputs = [], out_weights = []):
        self.value = value
        self.index = index
        self.inputs = inputs
        self.outputs = outputs
        self.in_weights = in_weights
        self.out_weights = out_weights

    def set_outputs(self,outputs,out_weights):
        self.outputs = outputs
        self.out_weights = out_weights

    def __str__(self):
        return "Value: "+str(self.value)+"   Index :"+str(self.index)+"   Outputs: "+str(self.outputs)


class Neaural_Net:

    def __init__(self):
        self.layers = []
        



def generate_net(inpt, answer):
    a = Node(inpt[0],1, [], []);
    b = Node(inpt[1],2, [], []);
    c = Node(inpt[2],3, [], []);

    start_nodes = [a,b,c]


    h1 = Node(0,4, [a,b,c], [0.5,0.4,0.9]);
    h2 = Node(0,5, [a,b,c], [0.5,0.4,0.9]);
    h3 = Node(0,6, [a,b,c], [0.5,0.4,0.9]);


    a.set_outputs([h1,h2],[0.5,0.5,0.5]);
    b.set_outputs([h1,h2],[0.4,0.4,0.4]);
    c.set_outputs([h1,h2],[0.9,0.9,0.9]);

    y = Node(0,7, [h1,h2], [0.9,0.9,0.9]);

    h1.set_outputs([y],[0.9]);
    h2.set_outputs([y],[0.9]);
    h3.set_outputs([y],[0.9]);

    net = ([a,b,c],[h1,h2,h3],[y])
    #snapshot(net)

    return propagate(net,inpt, answer)

def improve_net(net,inpt,answer):

    net = propagate(net,inpt,answer)
    
    LEARNING_RATE = 1/10
    ([a,b,c],[h1,h2,h3],[y],error) = net
    
    for element in [a,b,c]:
        delta = LEARNING_RATE*element.value*error
        for i in range(len(element.out_weights)):
            element.out_weights[i] += delta
    return net

def propagate(net,inpt,answer):

    ([a,b,c],[h1,h2,h3],[y]) = net

    a.value = inpt[0]
    b.value = inpt[1]
    c.value = inpt[2]

    start_nodes = [a,b,c]
    
    while len(start_nodes[0].outputs) != 0:
        for i in range(len(start_nodes)):
            for j in range(len(start_nodes[i].outputs)):
                start_nodes[i].outputs[j].value += start_nodes[i].value*start_nodes[i].out_weights[j]
                #print((start_nodes[i].outputs[j].index,start_nodes[i].outputs[j].value))
        start_nodes = start_nodes[0].outputs
    return([a,b,c],[h1,h2,h3],[y],answer - y.value)

def snapshot(net):
    ([a,b,c],[h1,h2,h3],[y]) = net
    node_text = [a.value,b.value,c.value,
             h1.value,h2.value,h3.value,
             y.value]
    weights = [a.out_weights[0],a.out_weights[1],a.out_weights[2],
           b.out_weights[0],b.out_weights[1],b.out_weights[2],
           c.out_weights[0],c.out_weights[1],c.out_weights[2],
           h1.out_weights[0],h2.out_weights[0],h3.out_weights[0], 
           ]
    Visualizer.draw_neural_net(0.1, 1, 0, 1, [3,3,1], node_text, weights)


net = generate_net([1,2,3],13)[:3]
#snapshot(net)
for i in range(11):
    a = random.randint(1,10)
    b = random.randint(1,10)
    c = random.randint(1,10)
    answer = a*a + b*b + c*c -1
    net = improve_net(net,[a,b,c],answer)[:3]
    #snapshot(net)
snapshot(net)





