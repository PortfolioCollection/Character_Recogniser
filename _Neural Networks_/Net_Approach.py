from Builder import*
from Neural_Net import*

def generate_net(structure, inputs):
    builder = Builder(structure, inputs)
    net = builder.get_net()
    return net

def propagate(net):
    for i in range(len(net.layers)-1):
        for j in range(len(net.layers[i])):
            for n in range(len(net.layers[i+1])):
                node_value = net.layers[i][j].value
                connection_weight = net.layers[i][j].connections[0][n].weight
                print(node_value)
                print(connection_weight)
                net.layers[i+1][n].value += node_value*connection_weight
    return net
            

if __name__ == "__main__":
    net = generate_net([3,2,3],list(range(14)))
    net = propagate(net)
    show_net(net)
