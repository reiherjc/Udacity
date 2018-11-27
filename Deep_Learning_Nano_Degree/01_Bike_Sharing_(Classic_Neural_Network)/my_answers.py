import numpy as np

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### Set self.activation_function to your implemented sigmoid function ####
        self.activation_function = lambda x : 1.0/(1.0+np.exp(-x))  # Replace 0 with your sigmoid calculation.
        
        # Next line contains definition of the derivative of the activation fuction
        self.Dactivation_function = lambda x : self.activation_function(x) * (1.0- self.activation_function(x))
       
     
    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
            Arguments
            ---------
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            # forward pass function below
            final_outputs, hidden_outputs = self.forward_pass_train(X)  
            # backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Arguments
            ---------
            X: features batch
        '''

        # Hidden layer
        hidden_inputs =  X.dot(self.weights_input_to_hidden)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # Output layer
        final_inputs = hidden_outputs.dot(self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer, Activation function here is identity
        
        return final_outputs, hidden_outputs

    
    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
        '''
                
        # Output error
        error = y - final_outputs # Output layer error is the difference between desired target and actual output.
       
        # Calculate the hidden layer's contribution to the error
        hidden_error = self.weights_hidden_to_output * self.Dactivation_function(X.dot(self.weights_input_to_hidden)[:,None])

        # Backpropagated error terms - Replace these values with your calculations.
        output_error_term = error * np.ones(error.shape)  # 1 is the derivative of the activation function f(x)=x of the output node
        hidden_error_term = hidden_error.dot(output_error_term)
        
        # Weight step (input to hidden
        delta_weights_i_h += X[:,None]* hidden_error_term 
        
        # Weight step (hidden to output)
        delta_weights_h_o += hidden_outputs[:,None] * output_error_term
        
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        # update hidden-to-output weights with gradient descent step
        self.weights_hidden_to_output += self.lr*(1.0/n_records) *delta_weights_h_o
        
        # update input-to-hidden weights with gradient descent step
        self.weights_input_to_hidden  += self.lr*(1.0/n_records) *delta_weights_i_h   

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        # Hidden layer - Replace these values with your calculations.
        hidden_inputs =  features.dot(self.weights_input_to_hidden)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # Output layer - Replace these values with your calculations.
        final_inputs = hidden_outputs.dot(self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer
        
        return final_outputs
        
#########################################################
# Set hyperparameters here
##########################################################
iterations = 3000
learning_rate = 0.7
hidden_nodes = 10
output_nodes = 1