import numpy as np

from process import Processor

class NeuralNetwork:
    def __init__(self, nn_arch, seed = 99):
        '''
        nn_arch - array of dictionaries that represents a layer
        '''
        np.random.seed(99)
        self.nn_arch = nn_arch
        self.param_vals = {}

        for idx, layer in enumerate(nn_arch):
            layer_idx = idx+1
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            self.param_vals['W' + str(layer_idx)] = np.random.randn(\
                layer_output_size, layer_input_size) * 0.1
            self.param_vals['b' + str(layer_idx)] = np.random.randn(\
                layer_output_size, 1) * 0.1
    def sigmoid(self, Z):
        return 1.0 / (1 + np.exp(-Z))
    def sigmoid_back(self, dA, Z):
        sig = self.sigmoid(Z)
        return dA * sig * (1-sig)
    def relu(self, Z):
        return np.maximum(0,Z)
    def relu_back(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0;
        return dZ;
    
    def single_layer_forward_prop(self, A_prev, W_curr, b_curr, activation="relu"):
        Z_curr = np.dot(W_curr, A_prev) + b_curr

        if activation == "relu":
            return self.relu(Z_curr), Z_curr
        elif activation == "sigmoid":
            return self.sigmoid(Z_curr), Z_curr
        else:
            raise Exception("Non-supported activation function")
    
    def full_forward_prop(self, X):
        memory = {}
        A_curr = X

        for idx, layer in enumerate(self.nn_arch):
            layer_idx = idx+1
            A_prev = A_curr
            
            activ_function_curr = layer["activation"]
            W_curr = self.param_vals["W"+str(layer_idx)]
            b_curr = self.param_vals["b"+str(layer_idx)]
            A_curr, Z_curr = self.single_layer_forward_prop(A_prev, W_curr, b_curr, activ_function_curr)

            memory["A"+str(idx)] = A_prev
            memory["Z"+str(layer_idx)] = Z_curr
        
        return A_curr, memory

    def get_cost_value(Y_hat, Y):
        m = Y_hat.shape[1]
        cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
        return np.squeeze(cost)

    
    def single_layer_backward_propagation(self, dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
        m = A_prev.shape[1]
        
        if activation == "relu":
            dZ_curr = self.relu_back(dA_curr, Z_curr)
        elif activation == "sigmoid":
            dZ_curr = self.sigmoid_back(dA_curr, Z_curr)
        else:
            raise Exception('Non-supported activation function')

        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr

    def full_backward_propagation(self, Y_hat, Y, memory):
        grads_values = {}
        m = Y.shape[1]
        Y = Y.reshape(Y_hat.shape)
    
        dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));
        
        for layer_idx_prev, layer in reversed(list(enumerate(self.nn_arch))):
            layer_idx_curr = layer_idx_prev + 1
            activ_function_curr = layer["activation"]
            
            dA_curr = dA_prev
            
            A_prev = memory["A" + str(layer_idx_prev)]
            Z_curr = memory["Z" + str(layer_idx_curr)]
            W_curr = self.param_vals["W" + str(layer_idx_curr)]
            b_curr = self.param_vals["b" + str(layer_idx_curr)]
            
            dA_prev, dW_curr, db_curr = self.single_layer_backward_propagation(
                dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
            
            grads_values["dW" + str(layer_idx_curr)] = dW_curr
            grads_values["db" + str(layer_idx_curr)] = db_curr
        
        return grads_values

    def update(self, grads_values, learning_rate):
        for layer_idx, layer in enumerate(self.nn_arch):
            self.param_vals["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]        
            self.param_vals["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]
    
    def train(self, X, Y, epochs, learning_rate):
        cost_history = []
        accuracy_history = []
        
        for i in range(epochs):
            Y_hat, cashe = self.full_forward_prop(X)
            cost = self.get_cost_value(Y_hat, Y)
            cost_history.append(cost)
            #accuracy = get_accuracy_value(Y_hat, Y)
            #accuracy_history.append(accuracy)
            
            grads_values = self.full_backward_propagation(Y_hat, Y, cashe)
            self.update(grads_values, learning_rate)
            
        return self.param_vals, cost_history

processor = Processor([])
matrix, annotations = processor.get_matrix_with_annotations()
training = np.array(matrix)
y = np.array(annotations)

n = len(matrix[0])


nn_architecture = [
    {"input_dim": n, "output_dim": n, "activation": "relu"},
    {"input_dim": n, "output_dim": int(n/2), "activation": "relu"},
    {"input_dim": int(n/2), "output_dim": int(n/2), "activation": "relu"},
    {"input_dim": int(n/2), "output_dim": 1, "activation": "sigmoid"},
]

nn = NeuralNetwork(nn_architecture)
nn.train(training, y, 1000, 0.01)