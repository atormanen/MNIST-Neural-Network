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
    def __init__(self, inputnodes, hiddennodes, hiddennodesone, outputnodes, learningrate):
        #set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.hnodesone = hiddennodesone
        self.onodes = outputnodes

        #link weight matricies, wih and who
        #wheights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        #w11 w21
        #w12 w22 etc
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5),(self.hnodes, self.inodes))
        self.wh = numpy.random.normal(0.0, pow(self.hnodesone, -0.5),(self.hnodesone, self.hnodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodesone))

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
    
    # Open and save the training data file
    train_data_file = open("mnist_train.csv",'r')
    train_data_list = train_data_file.readlines()
    train_data_file.close()

    # Open and save the test data file
    test_data_file = open("mnist_test.csv",'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    
##    all_values = data_list[0].split(',')
##    image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
##    matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
##    scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    inputnodes = 784
    hiddennodes = 500
    hiddennodesone =300
    outputnodes = 10
    learningrate = 0.1
    network = neuralNetwork(inputnodes, hiddennodes, hiddennodesone, outputnodes, learningrate)

    # Create a scorecard to keep track of good and bad results
    scorecard = []
    hit = 0
    miss = 0

    # Test how long the network will take to compute values
    start = timer()

    # Train the network with a given data set n number of times
    for n in range(0,1):   
        for record in train_data_list:
            all_values = record.split(',')
            scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            target = numpy.zeros(outputnodes) + 0.01
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
            print("Value: ",correct_label,"   Network Value: ",label)
        
    time = timer() - start
    print("\nHit: ",hit)
    print("Miss: ",miss)
    acc = (hit/(hit+miss))*100
    print("Accuracy: ",acc)    
    print("Time: ",time)
    
       

if __name__ == '__main__': main()
