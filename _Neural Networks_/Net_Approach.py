from Builder import*
from Neural_Net import*
import math

def generate_net(structure, inputs):
    """
    Generates a neural net with the given parameters
    """
    builder = Builder(structure, inputs)
    net = builder.get_net()
    return net

def propagate(net):
    """
    Does a single forward propagation for all nodes
    """
    for layer in net.layers[1:]:
        for node in layer:
            node.compute_value(node.connections[0])
    return net

def output_error(net,answers):
    """
    Calculates the error on all the output neurons by the cost function
    """
    output_layer = net.layers[-1]
    count = 0
    total_error = 0
    for node in output_layer:
        """
        δerror =∂C/∂a * σ′(z).
        """
        node.error = derivative_cost(node.value, answers[count])*node.comput_error()
        total_error += cost_funcion(node.value, answers[count])
        count+=1
    print("Total Error: "+str(total_error))
    return net

def hidden_error(net):
    """
    Calculates the error on all the output neurons by the cost function
    """
    reverse_layers = net.layers[::-1]
    for layer in reverse_layers[1:-1]:
        for node in layer:
            """
            δerror = (w_L+1)(δerror_L+1) * σ′(z).
            """
            error_weight = 0
            for connection in node.connections[0]:
                error_weight+= connection.weight * connection.front_node.error
            node.error = error_weight * node.compute_error()
    return net

def improve_bias(net,learning_rate):
    """
    Chenges the bias of a node based on the errors of the next layer
    """
    for layer in net.layers[1:]:
        for node in layer:
            for connection in node.connections[0]:
                node.bias -= learning_rate*node.error
    return net

def improve_weights(net,learning_rate):
    """
    Chenges the weights of a node based on the errors of the next layer
    """
    for layer in net.layers[1:]:
        for node in layer:
            for connection in node.connections[0]:
                connection.weight = connection.weight - learning_rate*(connection.back_node.value*connection.front_node.error)
    return net

def cost_funcion(prediction, answer):
    return (1/2)*(prediction - answer)**2
    

def derivative_cost(prediction, answer):
    return  prediction - answer
    
    
