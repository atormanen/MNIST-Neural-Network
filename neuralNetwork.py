#   Asher Tormanen
#   Last updated 6/10/2018


import numpy
import scipy.special
import pickle
import random
from timeit import default_timer as timer
import matplotlib.pyplot
import os.path
from os import path

# neural network class definition
class neuralNetwork:

    #initialize the neural network
    def __init__(self, inputNodes, hiddenNodes, hiddenNodesOne, outputNodes, learningRate):
        #set number of nodes in each input, hidden, output layer

        self.inputLayer = inputNodes
        self.hiddenLayerOne = hiddenNodes
        self.hiddenLayerTwo = hiddenNodesOne
        self.outputLayer = outputNodes

        #link weight matricies,  and who
        #wheights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        #w11 w21
        #w12 w22 etc
        self.weightsOne = numpy.random.normal(0.0, pow(self.hiddenLayerOne, -0.5),(self.hiddenLayerOne, self.inputLayer))
        self.weightsTwo = numpy.random.normal(0.0, pow(self.hiddenLayerTwo, -0.5),(self.hiddenLayerTwo, self.hiddenLayerOne))
        self.weightsThree = numpy.random.normal(0.0, pow(self.outputLayer, -0.5), (self.outputLayer, self.hiddenLayerTwo))

        #learning rate
        self.lr = learningRate

        #activation funcion is the sigmoid function
        self.activation_function = lambda x:scipy.special.expit(x)
        pass

    def networkWithWeights(self,weightsOne, weightsTwo, weightsThree):

        self.weightsOne = weightsOne
        self.weightsTwo = weightsTwo
        self.weightsThree = weightsThree
        pass

    #train the neural network
    def train(self, inputs_list, targets_list):
        #convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        #calcualte signals going into hiddenLayerOne
        hidden_one_inputs = numpy.dot(self.weightsOne, inputs)
        #apply activation function to nodes in hiddenLayerOne
        hidden_one_out = self.activation_function(hidden_one_inputs)

        #calfulate signals going into hiddenLayerTwo
        hidden_two = numpy.dot(self.weightsTwo, hidden_one_out)
        #apply activation function to nodes in hiddenLayerTwo
        hidden_two_out = self.activation_function(hidden_two)

        #calcualte signals going into outputLayer
        final_inputs = numpy.dot(self.weightsThree, hidden_two_out)
        #apply activation function to nodes in outputLayer
        final_outputs = self.activation_function(final_inputs)

        #output layer error is the (target - actual)
        output_errors = targets - final_outputs
        #hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_two_errors = numpy.dot(self.weightsThree.T, output_errors)
        hidden_one_errors = numpy.dot(self.weightsTwo.T, hidden_two_errors)


        #update the weights in between all of the layers
        self.weightsThree += self.lr * numpy.dot((output_errors * final_outputs * (1.0 -  final_outputs)), numpy.transpose(hidden_two_out))
        self.weightsTwo += self.lr * numpy.dot((hidden_two_errors * hidden_two_out * (1.0 - hidden_two_out)), numpy.transpose(hidden_one_out))
        self.weightsOne += self.lr * numpy.dot((hidden_one_errors * hidden_one_out * (1.0 - hidden_one_out)), numpy.transpose(inputs))

        pass

    #query the neural network
    def query(self,inputs_list):
        #convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        #calculate signals going into hiddenLayerOne
        hidden_inputs = numpy.dot(self.weightsOne, inputs)
        #apply activation function to hiddenLayerOne
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signals going into hiddenLayerTwo
        hidden_middle = numpy.dot(self.weightsTwo, hidden_outputs)
        #apply activation function to hiddenLayerTwo
        hidden_middle_out = self.activation_function(hidden_middle)

        #calculate signals going into outputLayer
        final_inputs = numpy.dot(self.weightsThree, hidden_middle_out)
        #apply activation function to outputLayer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


    def save(self):
        list = [self.weightsOne,self.weightsTwo,self.weightsThree]
        pickle_out = open("weightsList", "wb")
        pickle.dump(list, pickle_out)
        pickle_out.close()

    def randList(self, min, max, length):
        randList = [random.uniform(min,max)]
        length = length - 1
        for n in range(0,length):
            randList.append(random.uniform(min,max))
        return randList



def main():


    # Open and save the training data file
    train_data_file = open("mnist_train.csv",'r')
    train_data_list = train_data_file.readlines()
    train_data_file.close()

    # Open and save the test data file
    test_data_file = open("mnist_test.csv",'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    inputNodes = 784
    hiddenNodes = 80
    hiddenNodesOne =15
    outputNodes = 10
    learningRate = 0.025

    network = neuralNetwork(inputNodes, hiddenNodes, hiddenNodesOne, outputNodes, learningRate)

    if(path.exists("weightsList")):
        print("weightsList.py exists")
        pickleIn = open("weightsList", "rb")
        weights = pickle.load(pickleIn)
        network.networkWithWeights(weights[0], weights[1], weights[2])
        print(weights[0])
    else:
        print("weightsList.py does not exist")


    # Create a scorecard to keep track of good and bad results
    scorecard = []
    hit = 0
    miss = 0

    # Test how long the network will take to compute values
    start = timer()

    # Train the network with a given data set n number of times
    for n in range(0,10):
        for record in train_data_list:
            all_values = record.split(',')
            scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            target = numpy.zeros(outputNodes) + 0.01
            target[int(all_values[0])] = 0.99
            network.train(scaled_input,target)
    network.save()

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
            #print("Value: ",correct_label,"   Network Value: ",label)

    time = timer() - start
    print("\nHit: ",hit)
    print("Miss: ",miss)
    acc = (hit/(hit+miss))*100
    print("Accuracy: ",acc)
    print("Time: ",time)

    wait = input("PRESS ENTER TO CONTINUE.")

if __name__ == '__main__': main()
