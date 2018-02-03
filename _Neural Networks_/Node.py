import math



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


a = Node(1,1, [], []);
b = Node(2,2, [], []);
c = Node(3,3, [], []);

start_nodes = [a,b,c]


h1 = Node(0,4, [a,b,c], [0.5,0.4,0.9]);
h2 = Node(0,5, [a,b,c], [0.5,0.4,0.9]);


a.set_outputs([h1,h2],[0.5,0.5]);
b.set_outputs([h1,h2],[0.4,0.4]);
c.set_outputs([h1,h2],[0.9,0.9]);


x = Node(0,6, [h1,h2], [0.8,0.8]);
y = Node(0,7, [h1,h2], [0.9,0.9]);
z = Node(0,8, [h1,h2], [0.4,0.4]);

h1.set_outputs([x,y,z],[0.8,0.9,0.4]);
h2.set_outputs([x,y,z],[0.8,0.9,0.4]);

while len(start_nodes[0].outputs) != 0:
    #print(start_nodes[0].outputs[0])
    for i in range(len(start_nodes)):
        for j in range(len(start_nodes[i].outputs)):
            start_nodes[i].outputs[j].value += start_nodes[i].value*start_nodes[i].out_weights[j]
            #print(start_nodes[i].outputs[j].value)
        #print("------")
    #print(start_nodes[0].outputs[0].value)
    start_nodes = start_nodes[0].outputs
    #print(start_nodes[0])

print(x.value)
print(y.value)
print(z.value)
