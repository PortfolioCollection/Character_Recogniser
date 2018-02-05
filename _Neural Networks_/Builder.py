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
                self.net.connect(self.net.layers[prev][i], self.net.layers[nxt][j],random.randint(0,10))
                self.count+=1


    def show_net(self):
        nodes = []
        for layer in self.net.layers:
            for node in layer:
                nodes.append(node.value)
        weights = []
        for i in range(len(self.net.layers)-1):
            for node in self.net.layers[i]:
                for connection in node.connections[0]:
                    weights.append(connection.weight)
        Visualizer.draw_neural_net(0.1, 1, 0, 1, self.layer_sizes, nodes, weights)

    def get_net(self):
        return self.net

if __name__ == "__main__":
    builder = Builder([3,5,6],list(range(14)))
    builder.show_net()





