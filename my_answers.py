import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, (self.input_nodes, self.hidden_nodes))
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        # To do: sigmoid function
        self.activation_function = lambda x : 1.0 / (1.0 + (np.exp(-x))) 

    def train(self, features, targets):
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        # TODO: Hidden layer
        hidden_inputs = np.dot (X, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # TODO: Output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        # TODO: Output error
        error = y - final_outputs
        
        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(self.weights_hidden_to_output, error)
        
        # TODO: Backpropagated error terms
        output_error_term = error
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
        
        # TODO: Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X[:, None]
        
        # TODO: Weight step (hidden to output)
        delta_weights_h_o += output_error_term * hidden_outputs[:, None]
                
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        self.weights_hidden_to_output += (self.lr * delta_weights_h_o) / n_records # updates hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += (self.lr * delta_weights_i_h) / n_records # updates input-to-hidden weights with gradient descent step

    def run(self, features):
        # TODO: Hidden layer
        hidden_inputs = np.dot (features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # TODO: Output layer
        final_inputs = np.dot (hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs
        return final_outputs


#########################################################
# Hyperparameters
##########################################################
iterations = 8182
learning_rate = .512
hidden_nodes = 32
output_nodes = 1