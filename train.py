import numpy
import scipy.special
import pickle
import random
from timeit import default_timer as timer
import matplotlib.pyplot
#%matplotlib inline

# neural network class definition
class neuralNetwork:

    #initialize the neural network
    def __init__(self, inputnodes, hiddennodes, hiddennodesone, outputnodes, learningrate, wih, wh, who):
        #set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.hnodesone = hiddennodesone
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
        pass

    #train the neural network
    #@vectorize(["float32(float32, float32)"], target='cuda')
    def train(self, inputs_list, targets_list):
        #convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        #calcualte signals into hidden layer zero
        hidden_zero_inputs = numpy.dot(self.wih, inputs)
        #calculate the signals emergin from hidden layer
        hidden_zero_out = self.activation_function(hidden_zero_inputs)

        #calfulate signals into hidden layer one
        hidden_one = numpy.dot(self.wh, hidden_zero_out)
        #calculate the signals emerging from the hidden layer
        hidden_one_out = self.activation_function(hidden_one)

        #calcualte signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_one_out)
        #calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        #output layer error is the (target - actual)
        output_errors = targets - final_outputs
        #hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_one_errors = numpy.dot(self.who.T, output_errors)
        hidden_zero_errors = numpy.dot(self.wh.T, hidden_one_errors)


        #update the weights for the links between the hidden and outpu layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 -  final_outputs)), numpy.transpose(hidden_one_out))

        self.wh += self.lr * numpy.dot((hidden_one_errors * hidden_one_out * (1.0 - hidden_one_out)), numpy.transpose(hidden_zero_out))

        #update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_zero_errors * hidden_zero_out * (1.0 - hidden_zero_out)), numpy.transpose(inputs))
        pass


    def save(self):
        list = [self.wih,self.wh,self.who]
        pickle_out = open("network", "wb")
        pickle.dump(list, pickle_out)
        pickle_out.close()

    def randList(self, min, max, length):
        randList = [random.uniform(min,max)]
        length = length - 1
        for n in range(0,length):
            randList.append(random.uniform(min,max))
        return randList



def main():
    pickle_in = open("network", "rb")
    weights = pickle.load(pickle_in)

    # Open and save the training data file
    train_data_file = open("mnist_train.csv",'r')
    train_data_list = train_data_file.readlines()
    train_data_file.close()



    inputnodes = 784
    hiddennodes = 500
    hiddennodesone =300
    outputnodes = 10
    learningrate = 0.001
    network = neuralNetwork(inputnodes, hiddennodes, hiddennodesone, outputnodes, learningrate, weights[0], weights[1], weights[2])



    # Test how long the network will take to compute values
    start = timer()

    # Train the network with a given data set n number of times
    for n in range(0,165):
        for record in train_data_list:
            all_values = record.split(',')
            scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            target = numpy.zeros(outputnodes) + 0.01
            target[int(all_values[0])] = 0.99
            network.train(scaled_input,target)
    network.save()

    time = timer() - start
    print("Time: ",time)



if __name__ == '__main__': main()
