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
    def __init__(self, size, activation = 'linear', optional_alpha = 0):
        self.output_size = size
        self.activation = activation
        self.optional_alpha = optional_alpha
        self.isTrainable = True
        
    def Compile(self, input_size):
        self.input_size = input_size
        self.weights = np.random.randn(self.output_size,self.input_size)
        self.biases = np.random.randn(self.output_size,1)

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
            case 'linear':
                self.Y = self.S
            case 'sigmoid':
                self.Y = Sigmoid(self.S)
            case 'relu':
                self.Y = ReLU(self.S)
            case 'leaky_relu':
                self.Y = LeakyReLU(self.S, self.optional_alpha)
            case 'softmax':
                self.Y = Softmax(self.S)
        return self.Y
    def SetLastDelta_Backward(self, error_delta):
        #This is [n-1] layer
        match self.activation:
            case 'linear':
                derv = np.ones(shape=self.S.shape)
            case 'sigmoid':
                derv = SigmoidDer(self.S)
            case 'relu':
                derv = ReLUDer(self.S)
            case 'leaky_relu':
                derv = LeakyReLUDer(self.S, self.optional_alpha)
            case 'softmax':
                derv = SoftmaxDer(self.S)
        self.delta = derv * error_delta #The delta of the this layer 
    def Backward(self, next_inputs, next_activation, optional_alpha = 0):
        #This is [k] layer. Next_inputs must be S, not Y
        match next_activation:
            case 'linear':
                derv = np.ones(shape=next_inputs.shape)
            case 'sigmoid':
                derv = SigmoidDer(next_inputs)
            case 'relu':
                derv = ReLUDer(next_inputs)                  
            case 'leaky_relu':
                derv = LeakyReLUDer(next_inputs, optional_alpha)
            case 'softmax':
                derv = SoftmaxDer(next_inputs)
        wage_l_delta_prod = self.weights.T @ self.delta
        next_delta = derv * wage_l_delta_prod #The delta of [k-1] layer
        return next_delta
    def Learn(self, learning_rate):
        if self.delta is None:
            raise Exception("THERE IS NO DELTA! Use Backward before Learn")
        else:
            deltaWeight = self.remInputs @ self.delta.T
            wT = self.weights.T - learning_rate*deltaWeight
            self.weights = wT.T
            self.biases = self.biases - learning_rate*self.delta
class LayerNormalization:
    def __init__(self):
        self.isTrainable = True
    def Compile(self, input_size):
        self.size = input_size
        self.output_size = self.size #The same as size, just different names
        self.gammas = np.random.randn(input_size,1)
        self.biases = np.random.randn(input_size,1)
        self.S = None

    def Forward(self, inputs):
        self.remInputs = inputs
        self.mean = np.sum(inputs)/self.size
        self.var = (np.sum((inputs - self.mean)*(inputs - self.mean)))/self.size
        self.epsilon = 0.00001
        self.outputs = (inputs - self.mean)/math.sqrt(self.var + self.epsilon)
        self.S = self.outputs
       
        return self.outputs
    def SetLastDelta_Backward(self, error_delta):
        self.delta = error_delta 
    def Backward(self):
        '''
        gamma_squared_vec = self.gammas*(self.remInputs-self.mean)*(self.remInputs-self.mean)
        other_side_of_der = (1 - 1/self.size)*(2/self.size)*(-1/(2*(math.sqrt(self.var + self.epsilon)**3)))
        y_on_x_der = gamma_squared_vec * other_side_of_der
        next_delta = self.delta * y_on_x_der
        '''
        #s = sqrt(var + epsilon), scalar
        reversed_s = 1/math.sqrt(self.var + self.epsilon)
        gradient_of_error_with_respect_to_normalized_x = self.delta * self.gammas
        m_of_g_of_e_to_n_x = np.mean(gradient_of_error_with_respect_to_normalized_x) #mean of the above gradient
        m_of_g_of_multi = np.mean(gradient_of_error_with_respect_to_normalized_x * self.outputs) #It's self-explanatory
        next_delta = reversed_s*(gradient_of_error_with_respect_to_normalized_x - m_of_g_of_e_to_n_x - self.outputs * m_of_g_of_multi)
        return next_delta
    def Learn(self, learning_rate):
        if self.delta is None:
            raise Exception("THERE IS NO DELTA! Use Backward before Learn")
        else:
            delta_gammas = self.delta * (self.remInputs - self.mean)/math.sqrt(self.var + self.epsilon)
            self.gammas = self.gammas - learning_rate*delta_gammas
            self.biases = self.biases - learning_rate*self.delta
class Sigmoid_L:
    def __init__(self):
        self.isTrainable = False
    def Compile(self, input_size):
        self.size = input_size
        self.output_size = self.size #The same as size, just different names
        self.S = None
    def Forward(self, inputs):
        self.remInputs = inputs
        self.outputs = Sigmoid(inputs)
        self.S = self.outputs
        #print(self.outputs)
        return self.outputs
    def SetLastDelta_Backward(self, error_delta):
        self.delta = error_delta 
    def Backward(self):
        next_delta = self.delta * SigmoidDer(self.remInputs)
        return next_delta
class ReLU_L:
    def __init__(self):
        self.isTrainable = False
    def Compile(self, input_size):
        self.size = input_size
        self.output_size = self.size #The same as size, just different names
        self.S = None
    def Forward(self, inputs):
        self.remInputs = inputs
        self.outputs = ReLU(inputs)
        self.S = self.outputs
        #print(self.outputs)
        return self.outputs
    def SetLastDelta_Backward(self, error_delta):
        self.delta = error_delta 
    def Backward(self):
        next_delta = self.delta * ReLUDer(self.remInputs)
        return next_delta
class LeakyReLU_L:
    def __init__(self, alpha):
        self.isTrainable = False
        self.alpha = alpha
    def Compile(self, input_size):
        self.size = input_size
        self.output_size = self.size #The same as size, just different names
        self.S = None
    def Forward(self, inputs):
        self.remInputs = inputs
        self.outputs = LeakyReLU(inputs, self.alpha)
        self.S = self.outputs
        #print(self.outputs)
        return self.outputs
    def SetLastDelta_Backward(self, error_delta):
        self.delta = error_delta 
    def Backward(self):
        next_delta = self.delta * LeakyReLUDer(self.remInputs, self.alpha)
        return next_delta
class Model:
    def __init__(self, input_size, layers):
        self.input_size = input_size
        self.size = len(layers)
        self.layers = layers
        self.error = 0
        self.isCompiled = False
        self.layer_outputSizes = []
    def Compile(self):
        print(self.layer_outputSizes)
        for i in range(0, self.size):
            if i == 0:
                self.layers[i].Compile(self.input_size)
            else:
                self.layers[i].Compile(self.layers[i-1].output_size)
        self.layer_outputSizes = [l.output_size for l in self.layers]
        self.isCompiled = True
    def Forward(self, inputs):
        '''Inputs->outputs'''

        if(not self.isCompiled):
            raise Exception('Compile the model before using it!')
        input_data = np.expand_dims(np.array(inputs),0)
        if len(input_data[0]) != self.input_size:
            raise Exception("Wrong input size!")
        x = input_data.T
        for i in range(0, self.size):
            x = self.layers[i].Forward(x) 
        return x
    def Backward(self, inputs, real_values):
        '''Calculating deltas and error'''

        if(not self.isCompiled):
            raise Exception('Compile the model before using it!')

        predictions = self.Forward(inputs)
    
        #Error---------------------------------------

        #SquaredError
        #er_delta = 2*(predictions - real_values.T) 
        
        self.error = math.sqrt(np.sum((predictions - real_values.T)*(predictions - real_values.T)))
        
        #Categorical cross-entropy WITH SOFTMAX ON LAST LAYER
        if(self.layers[-1].activation != 'softmax'):
            raise Exception('You are using cat. cross-entropy with no softmax on the last layer. USE SOFTMAX PLEASE')
        er_delta = predictions - real_values.T #Error delta
        
        #Calculating deltas of layers----------------

        self.layers[self.size-1].SetLastDelta_Backward(er_delta)
        for i in range(self.size-1, 0,-1):
            if isinstance(self.layers[i], DenseLayer):
                try:
                    self.layers[i-1].delta = self.layers[i].Backward(self.layers[i-1].S, self.layers[i-1].activation, self.layers[i-1].optional_alpha)
                except AttributeError:
                    try:
                        self.layers[i-1].delta = self.layers[i].Backward(self.layers[i-1].S, 'linear', self.layers[i-1].optional_alpha)
                    except AttributeError:
                        self.layers[i-1].delta = self.layers[i].Backward(self.layers[i-1].S, 'linear', 0)
            if isinstance(self.layers[i], LayerNormalization):
                self.layers[i-1].delta = self.layers[i].Backward()
            if isinstance(self.layers[i], Sigmoid_L):
                self.layers[i-1].delta = self.layers[i].Backward()
            if isinstance(self.layers[i], ReLU_L):
                self.layers[i-1].delta = self.layers[i].Backward()
            if isinstance(self.layers[i], LeakyReLU_L):
                self.layers[i-1].delta = self.layers[i].Backward()
    def Learn(self, learning_rate):
        '''Using deltas to learn the net'''
        if(not self.isCompiled):
            raise Exception('Compile the model before using it!')
        for i in range(0, self.size):
            if self.layers[i].isTrainable:
                self.layers[i].Learn(learning_rate)
    def Train(self, data_base, epochs, learning_rate):
        '''Actual train function'''
        if(not self.isCompiled):
            raise Exception('Compile the model before using it!')
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
