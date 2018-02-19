from Builder import*
from Neural_Net import*
import math

def generate_net(structure, inputs):
    builder = Builder(structure, inputs)
    net = builder.get_net()
    return net

def propagate(net):
    for layer in net.layers[1:]:
        for node in layer:
            node.compute_value(node.connections[0])
            #print(node.index)
            #print((connection.back_node.index,connection.front_node.index,connection.weight,connection.front_node.value))
    return net

def output_error(net,answers):
    output_layer = net.layers[-1]
    count = 0
    for node in output_layer:
        #print(node)
        node.error = derivative_cost(node.value, answers[count])*node.compute_error(node.connections[0])
        count+=1

def hidden_error(net):
    reverse_layers = net.layers[::-1]
    for layer in reverse_layers[1:-1]:
        for node in layer:
            #print(node)
            error_weight = 0
            for connection in node.connections[1]:
                #print("Front Weight: "+str(connection.front_node.error))
                error_weight+= connection.weight * connection.front_node.error
            node.error = error_weight * node.compute_error(node.connections[0]) 
            #print("Error Weight: "+str(node.error))
            #print(node.error)

def improve_bias():
    LEARNING_RATE = 0.05
    for layer in net.layers[1:-1]:
        for node in layer:
            node.bias = node.bias - LEARNING_RATE*node.error

def improve_weights():
    LEARNING_RATE = 0.05
    for layer in net.layers[1:-1]:
        for node in layer:
            for connection in node.connections[1]:
                connection.weight = connection.weight - LEARNING_RATE*(connection.back_node.value*connection.front_node.error)
    

def cost_funcion(prediction, answer):
    return (1/2)*(prediction - answer)**2
    

def derivative_cost(prediction, answer):
    return  prediction - answer
    
    
if __name__ == "__main__":
    net = generate_net([3,10,1],[0,0,0])
    for k in range(10):
        for i in range(100000):
            x = random.randint(-1,1)
            y = random.randint(-1,1)
            z = random.randint(-1,1)
            net.set_inputs([x,y,z])
            net = propagate(net)
            output_error(net,[x*y*z])
            hidden_error(net)
            improve_bias()
            improve_weights()
        x = 0.2
        y = 0.25
        z = 0.24
        net.set_inputs([x,y,z])
        net = propagate(net)
        print((net.layers[-1][0].value))
    show_net(net)
    
    
