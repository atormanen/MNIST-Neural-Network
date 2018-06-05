import numpy
import scipy.special
import pickle
import random
from timeit import default_timer as timer
import matplotlib
import matplotlib.pyplot
#%matplotlib inline

# neural network class definition
class neuralNetwork:

    #initialize the neural network
    def __init__(self, inputnodes, hiddennodes, hiddennodesone, outputnodes, learningrate, wih, wh, who):
        #set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #link weight matricies, wih and who
        #wheights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        #w11 w21
        #w12 w22 etc
        self.wih = wih
        self.wh = wh
        self.who = who

        #learning rate
        self.lr = learningrate

        #activation funcion is the sigmoid function
        self.activation_function = lambda x:scipy.special.expit(x)
        self.inverse_activation_function = lambda x: scipy.special.logit(x)
        pass

    #query the neural network
    def query(self,inputs_list):
        #convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        #calculate the signals emergin from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        #calcualte the signals emergin from the final output layer
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
    
    def backquery(self, targets_list):
        # transpose the targets list to a vertical array
        final_outputs = numpy.array(targets_list, ndmin=2).T
        
        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        # calculate the signal out of the hidden layer
        hidden_one_outputs = numpy.dot(self.who.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_one_outputs -= numpy.min(hidden_one_outputs)
        hidden_one_outputs /= numpy.max(hidden_one_outputs)
        hidden_one_outputs *= 0.98
        hidden_one_outputs += 0.01


        hidden_one_inputs = self.inverse_activation_function(hidden_one_outputs)

        # calculate the signal out of the hidden layer
        hidden_zero_outputs = numpy.dot(self.wh.T, hidden_one_inputs)
        # scale them back to 0.01 to .99
        hidden_zero_outputs -= numpy.min(hidden_zero_outputs)
        hidden_zero_outputs /= numpy.max(hidden_zero_outputs)
        hidden_zero_outputs *= 0.98
        hidden_zero_outputs += 0.01
        
        # calculate the signal into the hidden layer
        hidden_inputs = self.inverse_activation_function(hidden_zero_outputs)
        
        # calculate the signal out of the input layer
        inputs = numpy.dot(self.wih.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        
        return inputs

       
def main():

    pickle_in = open("network", "rb") 
    weights = pickle.load(pickle_in)
    
    inputnodes = 784
    hiddennodes = 500
    hiddennodesone =300
    outputnodes = 10
    learningrate = 0.01
    network = neuralNetwork(inputnodes, hiddennodes, hiddennodesone, outputnodes, learningrate, weights[0], weights[1], weights[2])
    show = input("Enter a number: ")
    show = int(show)
    inp = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
    inp[show] = 0.99
    image_data = network.backquery(inp)
    matplotlib.pyplot.imshow(image_data.reshape(28,28), cmap='Greys', interpolation='None')
    matplotlib.pyplot.show()

if __name__ == '__main__': main()
