from Neural_Net import*
import random

class Builder:
    def __init__(self, layer_sizes, inputs):
        self.net = Neaural_Net()
        self.layer_sizes = layer_sizes
        self.inputs = inputs
        self.count = 0
        length = len(layer_sizes)
        self.net.add_layer()
        for i in range(layer_sizes[0]):
            self.net.add_input_node(inputs[i])
        for h in range(1,length-1):
            self.net.add_layer()
            for n in range(layer_sizes[h]):
                self.net.add_hidden_node(h)
            self.connect_layers(layer_sizes,h-1, h)
        self.net.add_layer()
        for o in range(layer_sizes[length-1]):
            self.net.add_output_node()
        self.connect_layers(layer_sizes,length-2, length-1)

    def connect_layers(self,layer_sizes,prev, nxt):
        for i in range(layer_sizes[prev]):
            for j in range(layer_sizes[nxt]):
                self.net.connect(self.net.layers[prev][i], self.net.layers[nxt][j],random.randint(0,10)/10)
                self.count+=1
                
    def get_net(self):
        return self.net

def collect_connections_weights(net):
    connections = []
    for layer in net.layers:
        seg = []
        for node in layer:
            #print(node)
            for connection in node.connections[1]:      #output connections
                seg.append(connection.weight)
                #print(connection)
        if seg:
            connections.append(seg)
    return connections

def mass_set_connections(net,array):
    count = 0
    for layer in net.layers:
        for node in layer:
            for connection in node.connections[1]:
                connection.weight = array[count]
                count+=1
    return net        


def show_net(net):
    layer_sizes = []
    nodes = []
    for layer in net.layers:
        layer_sizes.append(len(layer))
        for node in layer:
            nodes.append(node.value)
    weights = []
    for layer in net.layers:
        for node in layer:
            for connection in node.connections[1]:
                weights.append(connection.weight)
    Visualizer.draw_neural_net(0.1, 1, 0, 1, layer_sizes, nodes, weights)


if __name__ == "__main__":
    builder = Builder([3,2,3],list(range(12)))
    net = builder.net
    print(collect_connections_weights(net))
    net = mass_set_connections(net,[1,21,54,32,42,12,43,62,75,85,23,95])
    show_net(net)

