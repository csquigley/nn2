#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from math import e,log, sqrt
import json
#bug: somewhere the shape is changing which is screwing everything up (solved)


# In[2]:


class Sigmoid:
    def __name__(self):
        return "Sigmoid"
    def forward_calc(self,z):
        return (1/(1+e**-z))
    
    def backward_calc(self,z):
        dzda = ((1-1/(1+e**-z))*(1/(1+e**-z)))
        return dzda
    
class ReLU:
    def __name__(self):
        return "ReLU"
    def forward_calc(self,z):  
        z = np.array([max(i,0) for i in z[0]]).reshape(1,-1).astype(np.longdouble)
        return z
    def backward_calc(self,z):
        z = np.array([1 if x > 0 else 0 for x in z[0]]).reshape(z.shape).astype(np.longdouble)
        return z
class LeakyReLU:
    def __name__(self):
        return "LeakyReLU"
    def forward_calc(self,z):

        z = np.array([x if x > 0 else x * 0.1 for x in z[0]]).reshape(z.shape).astype(np.longdouble)

        return z
            
    def backward_calc(self,z):
        z = np.array([1 if x > 0 else 0.1 for x in z[0]]).reshape(z.shape).astype(np.longdouble)
        return z 
            
class NoActivation:
    def __name__(self):
        return "NoActivation"
class SquaredError: 
    def __init__(self):
        self.y_pred = 0
        self.y_true = 0
    def __name__(self):
        return "SquaredError" 
    def cost(self,y_pred,y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return (self.y_true-self.y_pred)**2
    def backward_calc(self):
        return np.array([2*(self.y_pred-self.y_true)]).astype(np.longdouble)
    
class CrossEntropy:
    def __init__(self):
        self.y_preds = None
        self.y_trues = None
    def __name__(self):
        return "CrossEntropy"
    def cost(self,y_preds,y_trues):
        self.y_preds = np.clip(y_preds.reshape(-1,),1.0e-1599,1.0)
        self.y_trues = y_trues.reshape(-1,)
#         print(y_preds)
#         return -np.log(y_preds[y_trues.argmax()])

#         return -sum((y_trues*np.log(y_preds)))
        return -np.log(self.y_preds[self.y_trues.argmax()])

    def backward_calc(self):
        r = -(1/self.y_preds[self.y_trues.argmax()])
        r = self.y_trues * r
                             
        return r
#         return self.y_trues
    
class SoftMax:
    #this is an activation function
    def __init__(self):
        self.output = None
        self.z = None
    def __name__(self):
        return "SoftMax"
    def forward_calc(self,z):

        self.z = z
        z = z.reshape(-1,)
        z = z - z.max()
        self.output = e**z/sum(e**z)

        return self.output.astype(np.longdouble)
    def backward_calc(self,z):
        #there are two cases
        #when the element you are taking the derivative for is the same as the output position
        #when the element is different
        #the derivative is different for both
        d = np.zeros((len(self.output),(len(self.output))))
        for j in range(0,len(self.output)):
            for i in range(0,len(self.output)):
                if j == i:
                    d[j][i] = self.output[j]*(1-self.output[i]) 
                else:
                    d[j][i] = -self.output[j]*self.output[i]
        d = (z@d).reshape(z.shape).astype(np.longdouble)

        return d
# class LogSoftMax:
#     def __name__(self):
#         return "LogSoftMax"
#     def __init__(self):
#         self.output = None
#         self.z = None
#     def forward_calc(self,z):
# #         z = np.clip(z,0,1)

#         z = z.reshape(-1,)
#         z = z - z.max()
#         self.z = z
#         #this normalizes the output and helps with numerical stability
# #         self.output = np.exp(z) / np.sum(np.exp(z),axis=0)
#         self.output = (e**z/sum(e**z))
# #         print(f"self.output={self.output}")
#         return np.log(self.output)
#     def backward_calc(self,z):
#         #there are two cases
#         #when the element you are taking the derivative for is the same as the output position
#         #when the element is different
#         #the derivative is different for both
#         d = np.zeros((len(self.output),(len(self.output))))
#         for j in range(0,len(self.output)):
#             for i in range(0,len(self.output)):
#                 if j == i:
#                     d[j][i] = 1-self.output[i]
#                 else:
#                     d[j][i] = -self.output[i]
#         d = (z@d).reshape(z.shape).astype(np.longdouble)

# #         print((self.output - z).shape)
# #         print(self.output-z)
# #         return self.output - z
# #         if ((self.output - z) - d).max() < 0.00001:
# #         print(self.output - z)
#         return d
        
        
#a function to decay learning rate over time   
def EDL(time_period,start_lr,decay_rate=0.001):
    return start_lr*e**(-decay_rate*time_period)


class Adam:
    def __name__(self):
        return "Adam"
    def __init__(self,lr=0.0001,beta1=0.9,beta2=.999):
        self.t = 1
        self.m = 0
        self.v = 0
        self.alpha = lr
        self.beta1 = beta1
        self.beta2 = beta2
        
    def adam(self):
        #these will be multiplied by the gradients in the update step
        self.mt = self.beta1*self.beta1*(self.t-1) + (1-self.beta1)
        self.vt = self.beta2*self.beta2*(self.t-1) + (1-self.beta2)
        self.t += 1
        

class Dropout:
    #NOT WORKING
    def __name__(self):
        return 'Dropout'
    def __init__(self,p,inpf,outpf,activation=NoActivation):
        self.inpf = inpf
        self.outpf = outpf
        self.p = p
        self.drop = None
        self.activation = activation()
    def forward_calc(self,X):
        self.drop = np.random.randint(0,2,X.shape)
        X = X * self.drop
        return X
    def backward_calc(self,z):
        return z 
    def update_w(self,lr=None,batch_size=None):
        pass
        


# In[3]:


class Layer:
    
    def __init__(self,inpf,outpf,activation=NoActivation):
        self.inpf = inpf
        self.outpf = outpf
        self.W = ((np.random.randn(outpf,inpf))).astype(np.longdouble)
        self.W = 2*((self.W-self.W.min())/(self.W.max()-self.W.min()))-1
#         lower, upper = -(1.0 /sqrt(outpf)),(1.0/sqrt(outpf))
#         self.W = (lower + np.random.randn(outpf,inpf)*(upper-lower)).astype(np.longdouble)
        self.b = np.ones((1,outpf))
        self.activation = activation()
        self.z = None
        self.X = None
        self.dCdW = np.zeros(self.W.shape,dtype=np.longdouble)
        self.dCdb = np.zeros((1,outpf),dtype=np.longdouble)
        
    def forward_calc(self,X):
        
        self.X = X
        self.z = sum((X * self.W).T) + self.b
#         if self.activation.__name__() == "SoftMax":
#             print(self.z)
#         if max(z) > 1.0e10 or min(z) < 1.0e-10:
#             print("Houstin there's a problem")
        if self.activation.__name__() != "NoActivation":
            a = self.activation.forward_calc(self.z).astype(np.longdouble)
            return a.astype(np.longdouble)
#         if self.activation.__name__() == "SoftMax":
#             print(f"self.z={self.z}")
#             print(f"a={a}")

        return self.z.astype(np.longdouble)
    
    def backward_calc(self,d):
        #if the layer has an activation function, first find the derivative of the activation function
        if self.activation.__name__() != "NoActivation":
            d = (d * self.activation.backward_calc(self.z)).astype(np.longdouble)
        #calculate the derivative of the weights and biases
        self.dCdW += np.clip((d.T * self.X),-10,10)
#         if self.dCdW.max() > 5.0:
#             print(self.dCdW.max())
#         if self.dCdW.min() < 1.0:
#             print(self.dCdW.min())
#         print(f"self.b={self.b}")
        self.dCdb += (d * self.b)
        

#         print(f"self.dCdb{self.dCdb}")
        #calculate the derivative of X to send back to the previous layer
        #it isn't necessary to save dCdX because we don't change that parameter
        dCdX = sum(d.T * self.W)
#         if dCdX.max() > 1.0e25:
#             print("numbers getting crazy large")
#         if dCdX.min() < 1.0e-25:
#             print("numbers getting close to zero")
#             if dCdX.min() < 0:
#                 print("numbers are actually less than zero")
#         print(f"dCdX={dCdX}")
        return dCdX.astype(np.longdouble)
            
    def update_w(self,lr=0.0001,batch_size=10):
        #update the weights and the biases
        self.W += (self.dCdW/batch_size * lr)
#         self.W = (self.W)/(self.W.max()-self.W.min())

        self.b += (self.dCdb/batch_size * lr)
#         if self.W.max() > 1.0:
#             print("one of the weights is over 1")
#             print(self.W.max())
        #reset the derivatives of the weights and the biases to zero
        self.dCdW = np.zeros(self.W.shape,dtype=np.longdouble)
        self.dCdb = np.zeros(self.b.shape,dtype=np.longdouble)


# In[4]:


class Network:
    
    def __init__(self,*layers):
        #initialize network object, the most important thing here is the layers
        self.layers = layers
        self.lr = 0.001
        self.cost = None
        self.X = None
        self.y = None
        self.batch_size = 10
        
    def settings(self,lr=0.001,cost=SquaredError,batch_size=10):
        #this method exists to update parameters (such as the learning rate) while training
        self.lr = lr
        self.cost = cost()
        self.batch_size = batch_size
        
    def connect_data(self,X,y):
        #connect the data separately because it allows us to use different data to train the network later
        self.X = X
        self.y = y
        
    def train(self):
        costs = []
        for n,i in enumerate(self.X):
            #predict y for input x
            for layer in self.layers:
#                 print("doing this")
                i = layer.forward_calc(i)

            #calculate cost
#             print(self.y[n])
            cost = self.cost.cost(i,self.y[n])
            #append cost for statistics, measuring performance
            costs.append(cost)
            #calculate the derivative of the cost function for the given result
            d = self.cost.backward_calc().reshape(1,-1)
            #propagate the derivatives backward through the layers
            for i in range(len(self.layers)-1,-1,-1):
                d = self.layers[i].backward_calc(d)
            #update the weights when n = batch_size
            if (n + 1) % self.batch_size == 0:
                for layer in self.layers:
                    layer.update_w(lr=self.lr,batch_size=self.batch_size) 
        #calculate the average cost for the epoch
        costs = sum(costs)/len(costs)
        return costs
    
    def predict(self,X):
        #predict y without running backpropagation 
        preds = []
        for i in X:
            for layer in self.layers:
                i = layer.forward_calc(i)

#             i = i[0]
            preds.append(i)
        preds = np.array(preds)
        return preds
    
    def save(self,file_path):
        #save the model and the weights as .json  
        #some precision is lost (not noticeable though)
        with open(file_path,'w') as f:
            data = {
                'architecture' : [{'inpf': layer.inpf,'outpf':layer.outpf,'AF':layer.activation.__name__()} for layer in self.layers],
                'weights': [layer.W.astype(np.float32).tolist() for layer in self.layers],
                'bias':[layer.b.astype(np.float32).tolist() for layer in self.layers],
                'cost':self.cost.__name__()
            }
            data=json.dumps(data)
            f.write(data)
            f.close()
               
    def load(self,file_path):
        #load a model and its weights from .json
        with open(file_path) as f:
            data = json.loads(f.read())
            self.layers = [Layer(layer['inpf'],layer['outpf'],eval(layer['AF'])) for layer in data['architecture']]
            self.cost = eval(data['cost'])()
            for n,layer in enumerate(self.layers):
                layer.W = np.array(data['weights'][n])
                layer.b = np.array(data['bias'][n])

    def info(self):
        print("Network Architecture")
        for layer in self.layers:
            print(f"Inputs: {layer.inpf}, Outputs: {layer.outpf}, Activation: {layer.activation.__name__()}")
        if self.cost != None:
            print(f"Cost Function: {self.cost.__name__()}\n")
        else:
            print(f"No Cost Function Connected\n")
        print("Settings")
        print(f"lr={self.lr} batch_size={self.batch_size}")
        #return information about the parameters, number of nodes, etc. 

import pandas as pd

columns = ["sepal.length","sepal.width","petal.length","petal.width","variety"]
data = pd.read_csv("flowers.csv",names=columns)

data = data.drop(index=0)
data.head()
data = data.sample(frac=1)
data['variety'].value_counts()
datay = pd.get_dummies(data['variety'])
dataX = data.drop('variety',axis=1)
dataX.head()


y = datay.to_numpy().astype(float)


X = dataX.to_numpy().astype(np.float32)

X = (X-X.min())/(X.max()-X.min())

cells = 256
net = Network(
    Layer(4,cells,LeakyReLU),
#     Dropout(0.2,128,128),
    Layer(cells,cells,LeakyReLU),
#     Dropout(0.5,128,128),
    Layer(cells,cells,LeakyReLU),
    Layer(cells,cells,LeakyReLU),
#     Dropout(0.5,128,128),
    Layer(cells,3,SoftMax)
)
net.settings(lr=0.00001,cost=CrossEntropy,batch_size=25)


# In[35]:


net.connect_data(X=X,y=y)


# In[36]:


net.info()



epochs = 100
cst = []
cor = 0
for i in range(0,epochs):
    c = net.train()
    cst.append(c)
    #decrease the learning rate exponentially 
    net.settings(cost=CrossEntropy,batch_size=10,lr=EDL(i,0.00001,0.001))
    for n,r in enumerate(X):
        pred = net.predict([r])[0].argmax()
        true = y[n].argmax()
        if pred == true:
            cor += 1
    if (i+1) % 1 == 0:
        print(f"epoch {i+1} | cost {round(c,4)} | accuracy {round(100*cor/len(X),2)}%")
    data = data.sample(frac=1)
    datay = pd.get_dummies(data['variety'])
    dataX = data.drop('variety',axis=1)
    y = datay.to_numpy().astype(np.float64)
    X = dataX.to_numpy().astype(np.float64)
    X = (X-X.min())/(X.max()-X.min())
    net.connect_data(X=X,y=y)
    
    cor = 0
cst = np.array(cst)


n = 0

net.predict([X[n]])

cor = 0

for n,i in enumerate(X):
    pred = net.predict([i])[0].argmax()
    true = y[n].argmax()
    if pred == true:
        cor += 1
        
cor/len(X)

net.save('iris_model_testp2.json')



net2 = Network()



net2.load('iris_model_86p.json')
net2.connect_data(X=X,y=y)


