from Builder import*

def propagate(net):
    """
    Does a single forward propagation for all nodes
    """
    for l in range(1,len(net.activation_layers)):         #for layer in layers
        net.compute_value(l)
    return net

def output_error(net,answers):
    delta = derivative_cost(net.activation_layers[-1],answers)*sigmoid_prime(net.z_layers[-1])
    #print(derivative_cost(net.activation_layers[-1],answers))
    net.delta_layers[-1] = delta
    t_delta = np.expand_dims(delta, axis=1)
    t_activation_layers = np.expand_dims(net.activation_layers[-2], axis=1)
    #print(delta)
    #print(net.activation_layers[-2].transpose())
    net.weight_errors[-1] = np.dot(t_delta,t_activation_layers.transpose())
    return net

def hidden_error(net):
    #print(len(net.layer_sizes))
    for l in range(2,len(net.layer_sizes)):
        sp = sigmoid_prime(net.z_layers[-l])
        send_back = np.dot(net.weight_layers[-l+1].transpose(),net.delta_layers[-l+1])
        net.delta_layers[-l] = np.multiply(send_back,sp)
        t_delta = np.expand_dims(net.delta_layers[-l], axis=1)
        t_activation_layers = np.expand_dims(net.activation_layers[-l-1], axis=1)
        net.weight_errors[-l] += np.dot(t_delta, t_activation_layers.transpose())
    return net

def improve_bias(net,learning_rate,size):
    net.bias_layers = [b-(learning_rate/size)*nb
                       for b, nb in zip(net.bias_layers, net.delta_layers)]
    return net

def improve_weights(net,learning_rate,size):
    net.weight_layers = [w-(learning_rate/size)*nw
                        for w, nw in zip(net.weight_layers, net.weight_errors)]
    return net

def cost_funcion(prediction, answer):
    return (1/2)(answer-prediction)**2

def derivative_cost(prediction, answer):
    return (prediction-answer)
    
if __name__ == "__main__":
    layers = [3,3,3]
    builder = Builder(layers)
    net = builder.generate_net()

    num = 300
    size = 100
    
    for i in range(num):
        for j in range(size):
            lst = [0,0,0]
            lst[0] = random.randint(0,1)
            lst[1] = random.randint(0,1)
            lst[2] = random.randint(0,1)
            net.set_inputs(lst)
            net = propagate(net)
            net = output_error(net,np.array(lst))
            net = hidden_error(net)
        net = improve_bias(net,1,size)
        net = improve_weights(net,1,size)
        net.delta_layers = [np.zeros(b.shape) for b in net.delta_layers]
        net.weight_errors = [np.zeros(w.shape) for w in net.weight_errors]
    lst = [1,0,0]
    net.set_inputs(lst)
    net = propagate(net)
    #print(net.delta_layers)
    #print(net.weight_errors)
    #print(net.weight_layers)
    #print(net.activation_layers)
    net.show(0,len(layers)-1)
    
