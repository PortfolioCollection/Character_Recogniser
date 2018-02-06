from Builder import*
from Neural_Net import*

def generate_net(structure, inputs):
    builder = Builder(structure, inputs)
    net = builder.get_net()
    return net

def propagate(net):
    for layer in net.layers:
        for node in layer:
            for connection in node.connections[1]:      #output connections
                connection.front_node.value += connection.back_node.value * connection.weight
                connection.front_node.value = round(connection.front_node.value, 2)
                #print((connection.back_node.index,connection.front_node.index,connection.weight,connection.front_node.value))
        
    return net
            

if __name__ == "__main__":
    net = generate_net([3,2,3],list(range(12)))
    net = propagate(net)
    show_net(net)
