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


    #query the neural network
    def query(self,inputs_list):
        #convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        #calculate the signals emergin from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signals into middle hidden layer
        hidden_middle = numpy.dot(self.wh, hidden_outputs)
        #calculate the signals emerging from the hidden layer
        hidden_middle_out = self.activation_function(hidden_middle)

        #calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_middle_out)

        #calcualte the signals emergin from the final output layer
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

    
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


    # Open and save the test data file
    test_data_file = open("mnist_test.csv",'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    inputnodes = 784
    hiddennodes = 500
    hiddennodesone =300
    outputnodes = 10
    learningrate = 0.01
    network = neuralNetwork(inputnodes, hiddennodes, hiddennodesone, outputnodes, learningrate, weights[0], weights[1], weights[2])

    # Create a scorecard to keep track of good and bad results
    scorecard = []
    hit = 0
    miss = 0

    # Test how long the network will take to compute values
    start = timer()

            
    # Query the network with a test case that is not from the training list
    for record in test_data_list:
        all_values = record.split(',')
        scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        n = network.query(scaled_input)
        correct_label = all_values[0]
        label = numpy.argmax(n) 
        if (int(label) == int(correct_label)):
            hit = hit + 1
        else:
            miss = miss + 1
            print("Value: ",correct_label,"   Network Value: ",label)
        
    time = timer() - start
    print("\nHit: ",hit)
    print("Miss: ",miss)
    acc = (hit/(hit+miss))*100
    print("Accuracy: ",acc)    
    print("Time: ",time)

    pause = input("Press enter:")
       

if __name__ == '__main__': main()
