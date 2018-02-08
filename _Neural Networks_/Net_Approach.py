from Builder import*
from Neural_Net import*

def generate_net(structure, inputs):
    builder = Builder(structure, inputs)
    net = builder.get_net()
    return net

def propagate(net):
    for layer in net.layers[1:]:
        for node in layer:
            node.compute_value(node.connections[0])
            #print((connection.back_node.index,connection.front_node.index,connection.weight,connection.front_node.value))
        
    return net

def back_propagate(net):
    for layer in net.layers[::-1]:
        for node in layer:
            for connection in node.connections[1]:      #output connections
                connection.back_node.value += connection.front_node.value * connection.weight
                connection.back_node.value = round(connection.back_node.value, 2)            
    return net
    
if __name__ == "__main__":
    net = generate_net([4,7,10,2,4],list(range(1,5)))
    net = propagate(net)
    show_net(net)
