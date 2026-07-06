import numpy as np
import math
import matplotlib.pyplot as plt
#BOTH FUNCTIONS ONLY FOR VERTICAL MATRICES
def Sigmoid(X):
    S = np.zeros(shape=X.shape)
    for i in range(0,len(X)):
        x = X[i][0]
        S[i] = 1/(1+math.exp(-x))
    return S
def SigmoidDer(X):
    S = np.zeros(shape=X.shape)
    S = Sigmoid(X)*(1-Sigmoid(X))
    return S 
def ReLU(X):
    S = np.zeros(shape=X.shape)
    for i in range(0,len(X)):
        x = X[i][0]
        if x <= 0:
            S[i] = 0
        else:
            S[i] = x
    return S 
def ReLUDer(X):
    S = np.zeros(shape=X.shape)
    for i in range(0,len(X)):
        x = X[i][0]
        if x <= 0:
            S[i] = 0
        else:
            S[i] = 1
    return S 
def LeakyReLU(X, alpha):
    S = np.zeros(shape=X.shape)
    for i in range(0,len(X)):
        x = X[i][0]
        if x <= 0:
            S[i] = alpha * x
        else:
            S[i] = x
    return S 
def LeakyReLUDer(X, alpha):
    S = np.zeros(shape=X.shape)
    for i in range(0,len(X)):
        x = X[i][0]
        if x <= 0:
            S[i] = alpha
        else:
            S[i] = 1
    return S 
def Softmax(X):
    S = np.zeros(shape=X.shape)
    c = 0
    for i in range(0,len(X)):
        x = X[i][0]
        c = c + math.exp(x)
    for i in range(0,len(X)):
        x = X[i][0]
        S[i] = math.exp(x)/c
   # print(X, ':', S)
    return S 
def SoftmaxDer(X):
    S = np.zeros(shape=X.shape)
    S = Softmax(X)*(np.ones(shape=X.shape)- Softmax(X))
    return S 
class DenseLayer:
    def __init__(self, input_size, output_size, activation, optional_alpha = 0):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size,input_size)
        self.biases = np.random.randn(output_size,1)
        self.activation = activation
        self.optional_alpha = optional_alpha
    def Forward(self, inputs):
        self.remInputs = inputs
        '''
        if self.activation == 'relu' or self.activation == 'leaky_relu':
            mean = np.sum(inputs)/len(inputs)
            sd = max(np.std(inputs),1)
            inputs = (inputs - mean)/sd
            '''
        self.S = self.weights @ inputs + self.biases
        match self.activation:
            case 'sigmoid':
                self.Y = Sigmoid(self.S)
            case 'relu':
                self.Y = ReLU(self.S)
            case 'leaky_relu':
                self.Y = LeakyReLU(self.S, self.optional_alpha)
            case 'softmax':
                self.Y = Softmax(self.S)
        return self.Y
    def Backward(self, last_delta, last_weights, is_last_layer):
        if is_last_layer:
            match self.activation:
                case 'sigmoid':
                    derv = SigmoidDer(self.S)
                case 'relu':
                    derv = ReLUDer(self.S)
                case 'leaky_relu':
                    derv = LeakyReLUDer(self.S, self.optional_alpha)
                case 'softmax':
                    derv = SoftmaxDer(self.S)
            self.delta = derv * last_delta
        else:
            match self.activation:
                case 'sigmoid':
                    derv = SigmoidDer(self.S)
                case 'relu':
                    derv = ReLUDer(self.S)
                    
                case 'leaky_relu':
                    derv = LeakyReLUDer(self.S, self.optional_alpha)
                case 'softmax':
                    derv = SoftmaxDer(self.S)
            wage_l_delta_prod = last_weights.T @ last_delta
            self.delta = derv * wage_l_delta_prod
            #print(self.delta)
    def Learn(self, learning_rate):
        if self.delta is None:
            raise Exception("THERE IS NO DELTA! Throw Backward before Learn")
        else:
            deltaWeight = self.remInputs @ self.delta.T
            wT = self.weights.T - learning_rate*deltaWeight
            self.weights = wT.T
            self.biases = self.biases - learning_rate*self.delta
class LayerNormalization:
    def __init__(self, input_size, output_size, activation, optional_alpha = 0):
        self.input_size = input_size
        self.output_size = output_size
        if output_size != input_size:
            raise Exception('Output size must be the same as Input size in LayerNormalization layer!')
        self.gammas = np.random.randn(input_size,1)
        self.biases = np.random.randn(input_size,1)
    def Forward(self, inputs):
        self.remInputs = inputs
        self.mean = np.sum(inputs)/self.input_size
        self.var = ((inputs - self.mean)*(inputs - self.mean))/self.input_size
        self.epsilon = 0.00001
        self.outputs = (inputs - self.mean)/math.sqrt(self.var + self.epsilon)
       
        return self.outputs
    def Backward(self, last_delta, last_weights):
        gamma_squared_vec = self.gammas*(self.remInputs-self.mean)*(self.remInputs-self.mean)
        other_side_of_der = (1 - 1/self.input_size)*(2/self.input_size)*(-1/(2*(math.sqrt(self.var + self.epsilon)**3)))
        y_on_x_der = gamma_squared_vec * other_side_of_der
        self.delta = last_delta * y_on_x_der
        #TUTAJ SKONCZYLEM
    def Learn(self, learning_rate):
        if self.delta is None:
            raise Exception("THERE IS NO DELTA! Throw Backward before Learn")
        else:
            deltaWeight = self.remInputs @ self.delta.T
            wT = self.weights.T - learning_rate*deltaWeight
            self.weights = wT.T
            self.biases = self.biases - learning_rate*self.delta
class Model:
    def __init__(self, layers):
        self.input_size = layers[0].input_size
        self.size = len(layers)
        self.layers = layers
        self.error = 0
    def Forward(self, inputs):
        '''Inputs->outputs'''
        input_data = np.expand_dims(np.array(inputs),0)
        if len(input_data[0]) != self.input_size:
            raise Exception("Wrong input size!")
        x = input_data.T
        for i in range(0, self.size):
            x = self.layers[i].Forward(x) 
        return x
    def Backward(self, inputs, real_values):
        '''Calculating deltas and error'''
        #first delta - the 1D matrix containing all first derivtives of target function via model-output values
        predictions = self.Forward(inputs)
    
        #Error-----------------------------------

        #SquaredError
        #er_delta = 2*(predictions - real_values.T) 
        
        self.error = math.sqrt(np.sum((predictions - real_values.T)*(predictions - real_values.T)))

        #Categorical cross-entropy WITH SOFTMAX ON LAST LAYER
        if(self.layers[-1].activation != 'softmax'):
            raise Exception('You are using cat. cross-entropy with no softmax on the last layer. USE SOFTMAX PLEASE')
        er_delta = predictions - real_values.T

        self.layers[self.size-1].Backward(er_delta, None, is_last_layer = True)
        delta = self.layers[self.size-1].delta
        for i in range(self.size-2, -1,-1):
            self.layers[i].Backward(delta, self.layers[i+1].weights, is_last_layer = False)
            delta = self.layers[i].delta
    def Learn(self, learning_rate):
        '''Using deltas to learn the net'''
        for i in range(0, self.size):
            self.layers[i].Learn(learning_rate)
    def Train(self, data_base, epochs, learning_rate):
        '''Actual train function'''
        for i in range(epochs):
            cum_error = 0
            for data in data_base:
                input_data = data[0]
                output_data = np.expand_dims(np.array(data[1]),0)
                self.Backward(input_data,output_data)
                self.Learn(learning_rate)
                cum_error = cum_error + self.error
            cum_error = cum_error / len(data_base)
            print('Epoch nr. ' + str(i) + ' Error: ' + str(round(cum_error,4)))


#Sigmoid wiec wyjscia sa od [0,1]
