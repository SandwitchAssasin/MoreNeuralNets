import numpy as np
import math
import matplotlib.pyplot as plt
import copy
#CAN USE CuPy FOR INSTANT GPU SUPPORT!!!!!!
#BOTH FUNCTIONS ONLY FOR VERTICAL MATRICES
def Sigmoid(X):
    shap = X.shape
    lX = X.flatten()
    S = np.zeros(shape=lX.shape)
    for i in range(0,len(lX)):
        x = lX[i]
        if x >= 0:
            S[i] = 1/(1+math.exp(-x))
        else:
            S[i] = math.exp(x)/(1+math.exp(x))
    S = S.reshape(shap)
    return S
def SigmoidDer(X):
    shap = X.shape
    lX = X.flatten()
    S = np.zeros(shape=lX.shape)
    S = Sigmoid(X)*(1-Sigmoid(X))
    return S 
def ReLU(X):
    shap = X.shape
    lX = X.flatten()
    S = np.zeros(shape=lX.shape)
    for i in range(0,len(lX)):
        x = lX[i]
        if x <= 0:
            S[i] = 0
        else:
            S[i] = x
    S = S.reshape(shap)
    return S
def ReLUDer(X):
    shap = X.shape
    lX = X.flatten()
    S = np.zeros(shape=lX.shape)
    for i in range(0,len(X)):
        x = lX[i]
        if x <= 0:
            S[i] = 0
        else:
            S[i] = 1
    S = S.reshape(shap)
    return S
def LeakyReLU(X, alpha):
    shap = X.shape
    lX = X.flatten()
    S = np.zeros(shape=lX.shape)
    for i in range(0,len(lX)):
        x = lX[i]
        if x <= 0:
            S[i] = alpha*x
        else:
            S[i] = x
    S = S.reshape(shap)
    return S
def LeakyReLUDer(X, alpha):
    shap = X.shape
    lX = X.flatten()
    S = np.zeros(shape=lX.shape)
    for i in range(0,len(lX)):
        x = lX[i]
        if x <= 0:
            S[i] = alpha
        else:
            S[i] = 1
    S = S.reshape(shap)
    return S
def Softmax(X):
    S = np.zeros(shape=X.shape)
    dims = X.shape
    for v in range(dims[0]):
        c = 0
        for i in range(dims[1]):
            x = X[v,i]
            c = c + math.exp(x)
        for i in range(dims[1]):
            x = X[v,i]
            S[v,i] = math.exp(x)/c
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
        self.batch_size = 0
        
    def Compile(self, input_size):
        self.input_size = input_size
        self.weights = np.random.randn(self.output_size,self.input_size)
        self.biases = np.random.randn(1,self.output_size)

    def Forward(self, inputs):
        '''inputs->outputs'''
        self.remInputs = inputs
        self.batch_size = inputs.shape[0]
        self.S = inputs @ self.weights.T + np.tile(self.biases,(self.batch_size,1))
        match self.activation:
            case 'linear':
                self.Y = self.S.copy()
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
        '''Sets the delta into the layer (when it is first in model)'''
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
    def Backward(self, next_S, next_activation, optional_alpha = 0):
        '''Calculates the delta for the layer in back'''
        #This is [k] layer. Next_inputs must be S, not Y
        match next_activation:
            case 'linear':
                derv = np.ones(shape=next_S.shape)
            case 'sigmoid':
                derv = SigmoidDer(next_S)
            case 'relu':
                derv = ReLUDer(next_S)                  
            case 'leaky_relu':
                derv = LeakyReLUDer(next_S, optional_alpha)
            case 'softmax':
                derv = SoftmaxDer(next_S)
        wage_l_delta_prod =  self.delta @ self.weights
        next_delta = derv * wage_l_delta_prod #The delta of [k-1] layer
        return next_delta
    def Learn(self, learning_rate):
        if self.delta is None:
            raise Exception("THERE IS NO DELTA! Use Backward before Learn")
        else:
            #We sum both the delta for biases and the delta for weights in the whole batch
            summed_delta = self.delta.sum(axis=0, keepdims=True)
            deltaWeight = np.zeros(shape=(self.input_size,self.delta.shape[1]))
            for i in range(self.batch_size):
                #This loop calculates the weight_delta matrices for all inputs in batch multiplied by all deltas in batch
                #We expand dims since those are vectors, and we need matrices
                deltaWeight = deltaWeight + np.expand_dims(self.remInputs[i,:],axis=0).T@np.expand_dims(self.delta[i,:],axis=0)
            #Updating weights and biases 
            wT = self.weights.T - learning_rate*deltaWeight
            self.weights = wT.T
            self.biases = self.biases - learning_rate*summed_delta
            #print(self.delta)
class LayerNormalization:
    def __init__(self):
        self.isTrainable = True
    def Compile(self, input_size):
        self.size = input_size
        self.output_size = self.size #The same as size, just different names
        self.gammas = np.random.randn(1,self.size)
        self.biases = np.random.randn(1,self.size)
        self.S = None

    def Forward(self, inputs):
        self.remInputs = inputs
        self.means = np.expand_dims(np.sum(inputs,axis=1)/self.size,axis=1) #(batch_size,1)
        self.vars = (np.sum((inputs - self.means)*(inputs - self.means),axis=1))/self.size #(batch_size,1)
        
        self.epsilon = 0.00001
        #Leaky_relu ma problem na pewno
        self.normalized_x = (inputs - self.means)/np.expand_dims(np.sqrt(self.vars + self.epsilon),axis=1)
        self.outputs = self.gammas * self.normalized_x + self.biases
        self.S = self.outputs
       
        return self.outputs
    def SetLastDelta_Backward(self, error_delta):
        self.delta = error_delta 
    def Backward(self):
        reversed_s = np.expand_dims(1/np.sqrt(self.vars + self.epsilon),axis=1)
        gradient_of_error_with_respect_to_normalized_x = self.delta * np.tile(self.gammas,(self.batch_size,1))
        
        m_of_g_of_e_to_n_x = np.mean(gradient_of_error_with_respect_to_normalized_x,axis=1,keepdims=True) #mean of the above gradient
        m_of_g_of_multi = np.mean(gradient_of_error_with_respect_to_normalized_x * self.normalized_x,axis=1,keepdims=True) #It's self-explanatory
        next_delta = reversed_s*(gradient_of_error_with_respect_to_normalized_x - m_of_g_of_e_to_n_x - self.normalized_x * m_of_g_of_multi)
        return next_delta
    def Learn(self, learning_rate):
        if self.delta is None:
            raise Exception("THERE IS NO DELTA! Use Backward before Learn")
        else:
            delta_gammas = self.delta * self.outputs
            #print('L',self.delta)
            summed_delta = np.sum(self.delta,axis=0,keepdims=True)
            summed_delta_gammas = np.sum(delta_gammas,axis=0,keepdims=True)
            #print('H',summed_delta_gammas)
            self.gammas = self.gammas - learning_rate*summed_delta_gammas
            self.biases = self.biases - learning_rate*summed_delta
class BatchNormalization:
    #Trzeba dokonczyc
    def __init__(self):
        self.isTrainable = True
        self.isTraining = True
    def Compile(self, input_size):
        self.size = input_size
        self.output_size = self.size #The same as size, just different names
        self.gammas = np.ones(shape=(self.size,))
        self.biases = np.zeros(shape=(self.size,))

        self.running_means = np.zeros(shape=(1,self.size))
        self.running_vars = np.ones(shape=(1,self.size))
        self.momentum = 0.1
        self.S = None

    def Forward(self, inputs):
        self.remInputs = inputs
        if self.isTraining:
            self.means = inputs.mean(axis=0) #(size)
            self.vars = ((inputs - self.means)**2).mean(axis=0) #(size)
            #print(self.vars.shape)
        
            self.epsilon = 0.00001
            #Leaky_relu ma problem na pewno
            self.normalized_x = (inputs - self.means)/np.sqrt(self.vars + self.epsilon)
            self.outputs = self.gammas * self.normalized_x + self.biases
            self.S = self.outputs

            self.running_means = (1-self.momentum)*self.running_means + self.momentum * self.means
            self.running_vars = (1-self.momentum)*self.running_vars + self.momentum * self.vars
            #print(np.mean(self.outputs))
        else:
            self.normalized_x = (inputs - self.running_means)/np.sqrt(self.running_vars + self.epsilon)
            self.outputs = self.gammas * self.normalized_x + self.biases
            self.S = self.outputs
        return self.outputs
    def SetLastDelta_Backward(self, error_delta):
        self.delta = error_delta 
    def Backward(self):

        self.gradient_error_normalized_x = self.delta * self.gammas #(batch_size, features) rosnie strasznie mocno
        #print('E',self.gammas)
        #print('begin ',np.sum(self.gradient_error_normalized_x*(self.remInputs - self.means),axis=0),' then ',(-1/2)*np.float_power((self.vars + self.epsilon),-3/2))
        self.grad_error_var = np.sum(self.gradient_error_normalized_x*(self.remInputs - self.means)*(-1/2)*np.float_power((self.vars + self.epsilon),-3/2),axis=0)
        self.grad_error_mean = np.sum(self.gradient_error_normalized_x*(-1/np.sqrt(self.vars+self.epsilon)),axis=0) + self.grad_error_var * (np.sum((-2)*(self.remInputs - self.means),axis=0)/self.batch_size)
        self.next_delta = self.gradient_error_normalized_x * (1/np.sqrt(self.vars + self.epsilon)) + self.grad_error_var * (2*(self.remInputs - self.means)/self.batch_size) + self.grad_error_mean * (1/self.batch_size)
        #print('E',self.gammas)
        return self.next_delta
    def Learn(self, learning_rate):
        if self.delta is None:
            raise Exception("THERE IS NO DELTA! Use Backward before Learn")
        else:
            #Tu moze jakis blad
            summed_delta_gammas = np.sum(self.delta * self.normalized_x,axis=0)
            summed_delta = np.sum(self.delta,axis=0)
            #print(self.delta)
            #gammas idzie do 0
            self.gammas = self.gammas - learning_rate*summed_delta_gammas
            self.biases = self.biases - learning_rate*summed_delta
            #print(self.delta)
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
        return self.outputs
    def SetLastDelta_Backward(self, error_delta):
        self.delta = error_delta 
    def Backward(self):
        next_delta = self.delta * LeakyReLUDer(self.remInputs, self.alpha)
        return next_delta
class Model:
    def __init__(self, input_size, layers, batch_size = 1):
        self.input_size = input_size
        self.size = len(layers)
        self.layers = layers
        self.error = 0
        self.isCompiled = False
        self.layer_outputSizes = []
        self.batch_size = batch_size    
    def Compile(self,batch_size):
        '''Compiles the whole model'''
        for i in range(0, self.size):
            self.layers[i].batch_size = batch_size
        for i in range(0, self.size):
            if i == 0:
                self.layers[i].Compile(self.input_size)
            else:
                self.layers[i].Compile(self.layers[i-1].output_size)
        for i in range(0, self.size):
            if isinstance(self.layers[i], BatchNormalization):
                self.layers[i].batch_size = self.batch_size
        self.layer_outputSizes = [l.output_size for l in self.layers]
        self.isCompiled = True
    def Forward(self, inputs):
        '''Inputs->outputs'''
        if(not self.isCompiled):
            raise Exception('Compile the model before using it!')
        #For inputs that are vectors, we make them into matrices (n,1)
        input_data = np.array(inputs)
        if(len(input_data.shape) == 1):
            input_data = np.expand_dims(input_data,axis=0)

        if input_data.shape[1] != self.input_size:
            raise Exception("Wrong input size!")
        x = input_data
        for i in range(0, self.size):
            x = self.layers[i].Forward(x) 
        return x
    def Backward(self, batch):
        '''Calculating deltas and error'''

        if(not self.isCompiled):
            raise Exception('Compile the model before using it!')

        self.error = 0
        er_delta = np.zeros((self.batch_size,self.layer_outputSizes[-1]))
        batch_inputs = []
        batch_real_values = []
        for data in batch:
            batch_inputs.append(data[0])
            batch_real_values.append(data[1])

        batch_inputs_matrix = np.array(batch_inputs)
        batch_real_values_matrix = np.array(batch_real_values)

        predictions_batch = self.Forward(batch_inputs_matrix)
        er = np.sum((predictions_batch - batch_real_values_matrix)*(predictions_batch - batch_real_values_matrix))
        self.error = er

        #Error---------------------------------------
        
        #SquaredError
        er_delta = 2*(predictions_batch - batch_real_values_matrix) #Error delta is (output_size, batch_size)
        
        er_delta = er_delta
        '''
        #Categorical cross-entropy WITH SOFTMAX ON LAST LAYER
        if(self.layers[-1].activation != 'softmax'):
            raise Exception('You are using cat. cross-entropy with no softmax on the last layer. USE SOFTMAX PLEASE')
        er_delta = (predictions_batch - batch_real_values_matrix).sum(axis=1) #Error delta
        er_delta = np.expand_dims(er_delta,1)
        
        er_delta = er_delta / self.batch_size
        '''
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
            if isinstance(self.layers[i], BatchNormalization):
                self.layers[i-1].delta = self.layers[i].Backward()
            if isinstance(self.layers[i], Sigmoid_L):
                self.layers[i-1].delta = self.layers[i].Backward()
            if isinstance(self.layers[i], ReLU_L):
                self.layers[i-1].delta = self.layers[i].Backward()
            if isinstance(self.layers[i], LeakyReLU_L):
                self.layers[i-1].delta = self.layers[i].Backward()
            #print(self.layers[i-1].delta)
    def Learn(self, learning_rate):
        '''Using deltas to learn the net'''
        if(not self.isCompiled):
            raise Exception('Compile the model before using it!')
        for i in range(0, self.size):
            if self.layers[i].isTrainable:
                self.layers[i].Learn(learning_rate)
    def Train(self, data_base, epochs, learning_rate, batch_size = 1):
        '''Actual train function'''
        self.Compile(batch_size)
        self.batch_size = batch_size
        for ep in range(0,epochs):
            cum_error = 0
            batches = []
            restOfDataB = len(data_base)
            counter = 0
            batch = []
            for data in data_base:
                batch.append(data)
                counter = counter + 1
                restOfDataB = restOfDataB - 1
                if counter >= self.batch_size or restOfDataB <= 0:
                    counter = 0
                    batches.append(batch.copy())
                    batch = []
            for batch in batches:
                for i in range(0, self.size):
                    self.layers[i].batch_size = len(batch)
                self.Backward(batch)
                self.Learn(learning_rate)
                cum_error = cum_error + self.error
            cum_error = cum_error / len(data_base)
            print('Epoch nr. ' + str(ep) + ' Error: ' + str(round(cum_error,4)))
        #Training ends
        for i in range(0, self.size):
            if isinstance(self.layers[i], BatchNormalization):
                self.layers[i].isTraining = False


#Sigmoid wiec wyjscia sa od [0,1]
