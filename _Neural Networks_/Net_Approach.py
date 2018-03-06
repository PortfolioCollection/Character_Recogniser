from Builder import*

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
    for l in range(1,len(net.node_layers)):         #for layer in layers
        for n in range(len(net.node_layers[l])):    #for node in layer
            net.compute_value(l,n)
    return net

def output_error(net,answers):
    """
    Calculates the error on all the output neurons by the cost function
    """
    total_error = 0
    output_error = np.zeros(len(answers))
    derivative_error = np.zeros(len(answers))
    for i in range(len(output_error)):
        output_error[i] = derivative_cost(net.node_layers[-1][i], answers[i])
        derivative_error[i] = derivative_sigmoid(net.z_layers[-1][i])
        total_error += cost_funcion(net.node_layers[-1][i], answers[i])
    net.error_layers[-1] = np.multiply(output_error,derivative_error)
    #print("Total Error: "+str(total_error))
    return (net,total_error)

def hidden_error(net):
    """
    Calculates the error on all the output neurons by the cost function
    """
    
    for r in range(len(net.node_layers)-2,0,-1):
        weighting = np.dot(net.connection_layers[r].transpose(),net.error_layers[r+1])
        derivative_weigth = np.zeros(len(net.node_layers[r]))
        for i in range(len(net.node_layers[r])):
            derivative_weigth[i] = derivative_sigmoid(net.z_layers[r][i])
        net.error_layers[r] = np.multiply(weighting,derivative_weigth)
    return net

def improve_bias(net,learning_rate,size):
    """
    Chenges the bias of a node based on the errors of the next layer
    """
    for l in range(len(net.bias_layers)):
        for b in range(len(net.bias_layers[l])):
            net.bias_layers[l][b] -= (learning_rate/size)*net.error_layers[l][b]
            #print("Bias [{0},{1}]: {2}".format(l,b,net.bias_layers[l][b]))
    #print("---------------------")
    return net

def improve_weights(net,learning_rate,size):
    """
    Chenges the weights of a node based on the errors of the next layer
    """
    for l in range(len(net.connection_layers)):
        for n in range(len(net.connection_layers[l])):
            for w in range(len(net.connection_layers[l][n])):
                error = net.node_layers[l][w]*net.error_layers[l+1][n]
                net.connection_layers[l][n][w] -= (learning_rate/size)* error
                #print("Weight [{0},{1},{2}]: {3}".format(l,n,w,net.connection_layers[l][n][w]))
    #print("---------------------")
    return net

def cost_funcion(prediction, answer):
    return (1/2)*(prediction - answer)**2
    

def derivative_cost(prediction, answer):
    return  prediction - answer
    
if __name__ == "__main__":
    builder = Builder([3,2,3])
    net = builder.generate_net()
    net.set_inputs(list(range(3)))
    for i in range(30):
        net = propagate(net)
        net, total_error = output_error(net,np.array([1,1,1]))
        net = hidden_error(net)
        net = improve_bias(net,0.5,1)
        net = improve_weights(net,0.5,1)
        net.show(0,2)
    
    
