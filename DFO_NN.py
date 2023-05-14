import numpy as np
from DFO import DFO
import random 
#from keras.losses import CategoricalCrossentropy
import torch.nn as nn
import torch
from sklearn.metrics import f1_score
random.seed(10)

'''
the purpose of this code is to find a way to use disperse flies optimization to train our neural network
==> find the best set of weights for our neural network. 
'''

class NeuralNetwork : 
    '''
    this class implements an artificial neural network that is trained using the Dispersive flies optimization
    '''
    def __init__(self,X, Y, NN_structure, batch_size,num_flies, DFO_bounds,init_weights = None,epochs = 10 ,ratio = 0.8, delta = 0.001, max_iter = 100):

        # Our data and labels  : 
        self.x = X
        self.y = Y

        # number of epochs :
        self.epochs= epochs

        # ratio of our data :
        self.ratio = ratio

        # training and testing data
        self.X_train, self.X_test, self.Y_train , self.Y_test = self.train_test_split()


        # our training batch size: 
        self.batch_size = batch_size

        # number of unique classes of our labels
        self.num_classes = len(np.unique(self.y))
        #self.num_classes = len(list(set(map(tuple, self.y))))

        # a dictionnary of the structure of our neural network with the form :  Structure  = {0 : (10, 'Relu'), 1 : (10, 'Relu'), 2 : (3, 'Softmax')} 
        self.NN_structure = NN_structure

        # DFO Hyperparameters : 
        self.delta , self.max_iter = delta, max_iter 

        # number of flies in our dispersive flies optimization
        self.num_flies= num_flies

        # enviroenemnt bounds : 
        self.bounds = DFO_bounds

        # a dictionnary of the activation functions used. 
        self.activation_functions = {
            'Relu' : self.Relu, 
            'Softmax': self.Softmax, 
            'Sigmoid' : self.Sigmoid,
            'Tanh' : self.Tanh,
            'Leaky Relu': self.leaky_relu
        } 

        # best final_weights : 
        self.best_weights = init_weights


    def Relu(self, x):
        return np.maximum(0, x)
    def Softmax(self, x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    def Sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def Tanh(self, x):
        return np.tanh(x)    
    def leaky_relu(self, x, alpha=0.01):
        return np.maximum(alpha * x, x)

    
    def train_test_split(self):  
        '''
        function to split our training data w 
        #ratio : % of data that goes to the training set
        '''  
        n = len(self.x)
        n_samples = int(np.around(n*self.ratio, 0))
        n_data = np.arange(n)
        np.random.shuffle(n_data)
        idx_train = np.random.choice(n_data, n_samples, replace=False)
        idx_test = list(set(n_data) - set(idx_train))    
        X_train , X_test , Y_train, Y_test = self.x[idx_train, ], self.x[idx_test, ], self.y[idx_train, ], self.y[idx_test, ]
        return X_train , X_test , Y_train, Y_test 
    
    

    def loss_function(self, predicted_result, expected_result):
        '''
        loss function of the neural network : categorical loss entropy
        '''

        pred_tensor, true_tensor= torch.tensor(predicted_result, dtype=torch.float), torch.tensor(expected_result, dtype=torch.long)

        loss = nn.CrossEntropyLoss()
        #print(pred_tensor, pred_tensor.size())
        #print(true_tensor, true_tensor.size())

        fit = loss(pred_tensor, true_tensor)
        #loss = CategoricalCrossentropy(predicted_result, expected_result).numpy()
        return fit.item()

        
    def fitness_function(self,X,Y,weights):
        '''
        function to evaluate each fly weights
        '''
        # each weights represent a fly from our environement.             

        vect , label =  X, Y 
        layer_idx = 0
        
        for weight in weights:
            layer = self.NN_structure[layer_idx] # extract the tuple with the structure of the layer
            res = np.dot(vect,weight)
            vect = self.activation_functions[layer[1]](res)
            layer_idx+=1

        #print(vect.shape)
        #print('goood')
        #print([np.sum(item, axis=0) for item in vect])

        # vect : the probabilities of the classes
        # the fitness of the given weight :
        fitness = self.loss_function(vect,label)
        return fitness


    def predict(self, x):
        '''
        function to predict the for a given input 
        '''


        vect =  x
        layer_idx = 0
        
        for weight in self.best_weights:
            layer = self.NN_structure[layer_idx] # extract the tuple with the structure of the layer
            res = np.dot(vect,weight)
            vect = self.activation_functions[layer[1]](res)
            layer_idx+=1
        print(vect)

        predicted = np.argmax(vect, axis = 1)
        
        # the one hot encoded labels
        s = [[1 if item == idx else 0 for item in range(len(np.unique(predicted)))] for idx in predicted]

        return predicted


    def train(self):
        '''
        this method trains our neural network using DFO 

        '''        
        training_data = list(zip(self.X_train.tolist() , self.Y_train.tolist()))
        
        #print(training_data)
        #training_data = [list(item) for item in training_data]
        for epoch in range(self.epochs) :
            print("epoch num ",epoch," :") 
            batches = [training_data[i:i + self.batch_size] for i in range(0, len(training_data), self.batch_size)]
            batches = random.sample(batches, len(batches))
            input_size = self.x.shape[1]
            DFO_w = None
            for batch in batches : 
                #X_t , Y_t = [item[0] for item in batch], [item[1] for item in batch]
                X_t, Y_t = [item[0] for item in batch], [item[1] for item in batch]
                #print(X_t)
                DFO_env = DFO(X_t, Y_t,self.num_flies, self.fitness_function, self.NN_structure, input_size, self.bounds,DFO_w,self.delta, self.max_iter)
                self.best_weights, DFO_w = DFO_env.train()
        
        
        
        # now we predict the test  
        y_pred = self.predict(self.X_test)

        #print(y_pred)
        #print(self.Y_test)
        print(self.Y_test)
        test_labels = list(map(int, self.Y_test))
        print(test_labels)
        print(y_pred)
        '''
        true_positives = len(np.where((y_pred == test_labels) & (y_pred == 1) )[0])
        false_positives = len(np.where((y_pred != test_labels) & (y_pred == 1) )[0])
        false_negatives = len(np.where((y_pred != test_labels) & (y_pred == 0) )[0])

        print(true_positives, false_positives, false_negatives)

        f1_score = (true_positives) / ((true_positives + 1)/(2*(false_positives + false_negatives)))'''

        F1_score = f1_score(test_labels, y_pred, average= 'weighted')

        print("*" * 100)
        print("the f1 score of our neural network trained using DFO is : ", F1_score)
        
        return self.best_weights












