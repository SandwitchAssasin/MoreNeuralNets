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
    S = Sigmoid(X)*(1-Sigmoid(X))
    return S 
def ReLU(X):
    S = np.maximum(0,X)
    return S
def ReLUDer(X):
    S = X.copy()
    S[S <= 0] = 0
    S[S > 0] = 1
    return S
def LeakyReLU(X, alpha):
    S = np.maximum(alpha*X,X)
    return S
def LeakyReLUDer(X, alpha):
    S = X.copy()
    S[S <= 0] = alpha
    S[S > 0] = 1
    return S
def Softmax(X):
    S = np.zeros(shape=X.shape)
    dims = X.shape
    z = np.exp(X)
    z_ = np.sum(z,axis=1)
    S = (z.T/z_).T #Simply dividing each batch by its z_[i]
    return S 

class Dense:
    def __init__(self, size, activation = 'linear', optional_alpha = 0):
        self.output_size = size
        self.activation = activation
        self.optional_alpha = optional_alpha
        self.isTrainable = True
        self.batch_size = 0
        
    def Compile(self, input_size):
        #Dense layer only accepts 1 ranked tensors (input_size does not count batch_size)
        if len(input_size.shape) != 1:
            raise Exception("Wrong input shape for Dense!")
        self.input_size = input_size
        self.weights = np.random.randn(self.output_size,self.input_size)
        #self.biases = np.random.randn(1,self.output_size)
        #For ReLU use below
        self.biases = np.zeros((1,self.output_size))

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
                derv = np.ones(shape=self.S.shape)
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
                derv = np.ones(shape=next_S.shape)
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
class Conv2D:
    '''

    kernels = 10
    channels = 3

    input_size = 5
    kernel_size = 4
    padding = 1
    strides = 3

    output_size = math.floor((input_size + 2*padding - kernel_size)/strides) + 1

    #input_size - rozmiar wejscia bez padding
    #wpierw robimy template duzej macierzy dense
    templ_kernel = np.full((kernel_size,kernel_size),'',dtype='U10') 

    templ_modified_kernel = np.full((output_size*output_size,(input_size+2*padding)*(input_size+2*padding)),'',dtype='U10') 
    for c in range(kernel_size):
        for w in range(kernel_size):
            templ_kernel[c,w] = 'w' + str(c * kernel_size + w)
    for i in range(output_size):
        for j in range(output_size):
            templ_small_kernel = np.full((input_size+2*padding,input_size+2*padding),'',dtype='U10')
            templ_small_kernel[strides*i:kernel_size + strides*i,strides*j:kernel_size + strides*j] = templ_kernel
            print(templ_small_kernel)
            templ_small_kernel = templ_small_kernel.flatten()
            templ_modified_kernel[i*output_size+j,:] = templ_small_kernel

    kernel = np.random.randint(0,2,(kernel_size,kernel_size))

    curr_mod_kernel = templ_modified_kernel.copy()
    for c in range(kernel_size):
        for w in range(kernel_size):
            curr_mod_kernel[curr_mod_kernel == 'w' + str(c * kernel_size + w)] = str(kernel[c,w])
    curr_mod_kernel[curr_mod_kernel == ''] = str(0)
    curr_mod_kernel = curr_mod_kernel.astype(float)
    print(curr_mod_kernel)

    '''
    def __init__(self, num_of_kernels, kernel_size, padding, strides, activation = 'linear', optional_alpha = 0):
        self.kernel_size = kernel_size
        self.num_of_kernels = num_of_kernels
        self.padding = padding
        self.strides = strides
        self.activation = activation
        self.optional_alpha = optional_alpha
        self.isTrainable = True
        self.batch_size = 0
        
    def Compile(self, input_shape):
        #Conv2D layer only accepts 3 ranked tensors (input_size does not count batch_size)
        if len(input_shape.shape) != 3:
            raise Exception("Wrong input shape for Dense!")
        self.input_size = input_shape.shape[0]
        self.channels = input_shape.shape[2]
        self.output_size = math.floor((self.input_size + 2*self.padding - self.kernel_size)/self.strides) + 1

        #input_size - size of input - width of an input image
        #first we do a template of modified kernel so it will be like weight matrix for dense
        #templ_kernel - a kernel with strings like 'w00', 'w01' etc.
        #templ_small_kernel - a input-sized matrix of zeroes with one templ_kernel on itself (as in convolutions)
        #templ_modified_kernel - a dense-like weight matrix with strings, as it is just a template
        #templates are done, so the calculations can be done much faster (I think so)
        templ_kernel = np.full((self.kernel_size,self.kernel_size),'',dtype='U10') 

        self.templ_modified_kernel = np.full((self.output_size*self.output_size,(self.input_size+2*self.padding)*(self.input_size+2*self.padding)),'',dtype='U10') 
        for c in range(self.kernel_size):
            for w in range(self.kernel_size):
                templ_kernel[c,w] = 'w' + str(c * self.kernel_size + w)
        for i in range(self.output_size):
            for j in range(self.output_size):
                templ_small_kernel = np.full((self.input_size+2*self.padding,self.input_size+2*self.padding),'',dtype='U10')
                templ_small_kernel[self.strides*i:self.kernel_size + self.strides*i,self.strides*j:self.kernel_size + self.strides*j] = templ_kernel
                print(templ_small_kernel)
                templ_small_kernel = templ_small_kernel.flatten()
                self.templ_modified_kernel[i*self.output_size+j,:] = templ_small_kernel

        self.kernels = np.randn((self.num_of_kernels,self.channels,self.kernel_size,self.kernel_size))
        self.biases = np.zeros(self.output_size)

    def Forward(self, inputs):
        '''inputs->outputs'''
        self.remInputs = inputs
        self.batch_size = inputs.shape[0]
        for img_num in range(self.batch_size):
            output_img_total = np.zeros(self.kernel_size,self.kernel_size,self.num_of_kernels)
            for k in range(self.num_of_kernels):
                for c in range(self.channels):
                    tmp_img_1d = inputs[img_num,:,:,c]
                    tmp_img_1d = tmp_img_1d.flatten()
                    curr_kernel_1_channel = self.kernels[]

                    curr_mod_kernel = self.templ_modified_kernel.copy()
                    for c in range(self.kernel_size):
                        for w in range(self.kernel_size):
                            curr_mod_kernel[curr_mod_kernel == 'w' + str(c * self.kernel_size + w)] = str(curr_kernel_1_channel[c,w])
                    curr_mod_kernel[curr_mod_kernel == ''] = str(0)
                    curr_mod_kernel = curr_mod_kernel.astype(float)
                output_img_1d = 
                output_img = np.reshape(output_img_1d,shape=(self.output_size,self.output_size))
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
                derv = np.ones(shape=self.S.shape)
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
                derv = np.ones(shape=next_S.shape)
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
'''
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
'''
class BatchNormalization:
    #Jest ok
    def __init__(self):
        self.isTrainable = True
        self.isTraining = True
    def Compile(self, input_size):
        self.size = input_size
        self.output_size = self.size #The same as size, just different names
        self.gammas = np.ones(shape=(self.size,))
        self.biases = np.zeros(shape=(self.size,))
        self.iter = 0

        self.running_means = np.zeros(shape=(self.size,))
        self.running_vars = np.ones(shape=(self.size,))
        self.momentum = 0.1


        self.S = None

    def Forward(self, inputs):
        self.remInputs = inputs
        if self.isTraining:

            self.means = inputs.mean(axis=0) #(size)
            self.vars = ((inputs - self.means)**2).mean(axis=0) #(size)
        
            self.epsilon = 0.00001
            self.normalized_x = (inputs - self.means)/np.sqrt(self.vars + self.epsilon)
            self.outputs = self.gammas * self.normalized_x + self.biases
            self.S = self.outputs

            if self.iter != 0:
                self.running_means = (1-self.momentum)*self.running_means + self.momentum * self.means
                self.running_vars = (1-self.momentum)*self.running_vars + self.momentum * self.vars
            else:
                self.running_means = self.means
                self.running_vars = self.vars
            self.iter = self.iter + 1
        else:
            self.normalized_x = (inputs - self.running_means)/np.sqrt(self.running_vars + self.epsilon)
            self.outputs = self.gammas * self.normalized_x + self.biases
            self.S = self.outputs
        return self.outputs
    def SetLastDelta_Backward(self, error_delta):
        self.delta = error_delta 
    def Backward(self):

        self.gradient_error_normalized_x = self.delta * self.gammas #(batch_size, features) 
        self.grad_error_var = np.sum(self.gradient_error_normalized_x*(self.remInputs - self.means)*(-1/2)*np.float_power((self.vars + self.epsilon),-3/2),axis=0)
        self.grad_error_mean = np.sum(self.gradient_error_normalized_x*(-1/np.sqrt(self.vars+self.epsilon)),axis=0) + self.grad_error_var * (np.sum((-2)*(self.remInputs - self.means),axis=0)/self.batch_size)
        self.next_delta = self.gradient_error_normalized_x * (1/np.sqrt(self.vars + self.epsilon)) + self.grad_error_var * (2*(self.remInputs - self.means)/self.batch_size) + self.grad_error_mean * (1/self.batch_size)

        return self.next_delta
    def Learn(self, learning_rate):
        if self.delta is None:
            raise Exception("THERE IS NO DELTA! Use Backward before Learn")
        else:
            summed_delta_gammas = np.sum(self.delta * self.normalized_x,axis=0)
            summed_delta = np.sum(self.delta,axis=0)
            self.gammas = self.gammas - learning_rate*summed_delta_gammas
            self.biases = self.biases - learning_rate*summed_delta
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
        '''
        er_delta = 2*(predictions_batch - batch_real_values_matrix) #Error delta is (output_size, batch_size)
        
        er_delta = er_delta
        '''
        
        #Categorical cross-entropy WITH SOFTMAX ON LAST LAYER
        if(self.layers[-1].activation != 'softmax'):
            raise Exception('You are using cat. cross-entropy with no softmax on the last layer. USE SOFTMAX PLEASE')
        er_delta = predictions_batch - batch_real_values_matrix #Error delta
        
        #Calculating deltas of layers----------------

        self.layers[self.size-1].SetLastDelta_Backward(er_delta)
        for i in range(self.size-1, 0,-1):
            if isinstance(self.layers[i], Dense):
                try:
                    self.layers[i-1].delta = self.layers[i].Backward(self.layers[i-1].S, self.layers[i-1].activation, self.layers[i-1].optional_alpha)
                except AttributeError:
                    try:
                        self.layers[i-1].delta = self.layers[i].Backward(self.layers[i-1].S, 'linear', self.layers[i-1].optional_alpha)
                    except AttributeError:
                        try:
                            self.layers[i-1].delta = self.layers[i].Backward(self.layers[i-1].S, 'linear', 0)
                        except AttributeError:
                            if isinstance(self.layers[i-1], Conv2D):
                                raise Exception('You should not use Dense after Conv2D, use Flatten layer')
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