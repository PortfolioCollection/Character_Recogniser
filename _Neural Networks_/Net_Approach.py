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
    for node in output_layer:
        """
        δerror =∂C/∂a * σ′(z).
        """
        node.error = derivative_cost(node.value, answers[count])
        count+=1
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
            for connection in node.connections[1]:
                error_weight+= connection.weight * connection.front_node.error
            node.error = error_weight * node.compute_error()
    return net

def improve_bias(net):
    """
    Chenges the bias of a node based on the errors of the next layer
    """
    LEARNING_RATE = 0.05
    for layer in net.layers[1:-1]:
        for node in layer:
            for connection in node.connections[1]: ## What is this
                node.bias -= LEARNING_RATE*node.error
    return net

def improve_weights(net):
    """
    Chenges the weights of a node based on the errors of the next layer
    """
    LEARNING_RATE = 0.05
    for layer in net.layers[1:-1]:
        for node in layer:
            for connection in node.connections[1]:
                connection.weight = connection.weight - LEARNING_RATE*(connection.back_node.value*connection.front_node.error)

    return net

def cost_funcion(prediction, answer):
    return (1/2)*(prediction - answer)**2
    

def derivative_cost(prediction, answer):
    return  prediction - answer
    
    
if __name__ == "__main__":
    net = generate_net([4,8,10],[69,69,69,69])
    for k in range(1):
        for i in range(5000):
            w = random.randint(-1,1)
            x = random.randint(-1,1)
            y = random.randint(-1,1)
            z = random.randint(-1,1)
            net.set_inputs([2,2,2,2])
            net = propagate(net)
            output_error(net,[1,0,0,0,0,0,0,0,0,0])
            hidden_error(net)
            improve_bias(net)
            improve_weights(net)
        print((net.layers[-1][1].value))
    show_net(net)
    
    
