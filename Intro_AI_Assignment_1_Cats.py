#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import pickle as pkl
import random
import matplotlib.pyplot as plt


# In[83]:


def sigmoid(z):

    return 1.0 / (1.0 + np.exp(-z))

def lrloss(yhat, y):

    return 0.0 if yhat == y else -1.0*(y*np.log(yhat)+(1-y)*np.log(1-yhat))

def lrpredict(self, x):

    return 1.0 if self(x)>0.5 else 0.0


# In[46]:


cat_data = None
with open('C:/Users/frank/keio2019aia/data/assignment1/cat_data.pkl' , mode='rb') as f:
    cat_data = pkl.load(f)


# In[47]:


cat_data.keys()


# In[48]:


cat_data['train'],cat_data.keys()


# In[49]:


cat_example = cat_data['train']['cat'][0]

print(cat_example.shape)


# In[51]:


plt.imshow(cat_example)


# In[52]:


vector_cat = np.reshape(cat_example, cat_example.size)

print(vector_cat.size)
print(64*64*3)
print(vector_cat)


# Model

# In[81]:


class Cat_Model:
    
    def __init__(self,dimension=12288, weights=None, bias=None, activation=(lambda x:x), predict=lrpredict):
    
        self._dim = dimension
        self.w = weights or np.random.normal(size=self._dim)
        self.n = np.array (weights)
        self.b = bias if bias is not None else np.random.normal()
        self._a = activation
        self.predict = predict.__get__(self)
    
    def __str__(self):
        
        return 'Simple cell neuron\n        \tInput dimension: %d\n        \tBias: %f\n        \tWeights: %s\n        \tActivation: %s' % (self._dim, self.b, self.w, self._a.__name__)
    
    def __call__(self,x):
    
        yhat = self._a(np.dot(self.w, np.array(x)) + self.b)
        return yhat
    
    def predict(self, x):
    
        return self(x)
    
    def load_model(self, file_path):
        
        with open(file_path, mode='rb') as f:
            mm = pkl.load(f)
            
        self._dim = mm._dim
        self.w = mm.w
        self.b = mm.b
        self._a = mm._a
    
    def save_model(self):
        
        f = open('cat_model.pkl','wb')
        pkl.dump(self, f)
        f.close


# Trainer

# In[91]:


class Cat_Trainer:
    
    def __init__(self, dataset, model):
        
        self.dataset = dataset
        self.model = model
        self.loss = lrloss
        
    def accuracy(self, data):
        
        return 100*np.mean([1 if self.model.predict(x) == y else 0 for x, y in self.dataset.samples])
    
    def train(self, lr, ne):
        
        print(lr)
        
        print('training model on data...')
        accuracy = self.accuracy(self.dataset)
        print("initial accuracy: %.3f" % (accuracy))
        
        costs = []
        accuracies = []
        
        for epoch in range(1, ne+1):
            
            self.dataset.shuffle()
            J = 0
            dw = 0
            for d in self.dataset.samples:
                xi, yi = d
                yhat = self.model(xi)
                J += self.loss(yhat, yi)
                dz = yhat - yi
                dw += xi*dz
            J /= len(self.dataset.samples)
            dw /= len(self.dataset.samples)
            self.model.w = self.model.w - lr*dw
            
            accuracy = self.accuracy(self.dataset)
            
            if epoch%10 == 0:
                print('--> epoch=%d, accurcacy=%.3f' % (epoch, accuracy))
            costs.append(J)
            accuracies.append(accuracy)
            
        print('training complete')
        print('final accuracy: %.3f' % (self.accuracy(self.dataset)))
        costs = list(map(lambda t: np.mean(t), [np.array(costs)[i-10:i+11] for i in range(1, len(costs)-10)]))
        accuracies = list(map(lambda t: np.mean(t), [np.array(accuracies)[i-10:i+11] for i in range(1, len(accuracies)-10)]))

        return (costs,accuracies)


# Data Loader

# In[93]:


class Cat_Data():
    
    def __init__(self, data_file_path='C:/Users/frank/keio2019aia/data/assignment1', data_file_name='cat_data.pkl'):
        
        self.index = -1
        with open('C:/Users/frank/keio2019aia/data/assignment1/cat_data.pkl', mode='rb') as f:
            cat_data = pkl.load(f)
        self.samples = [(np.reshape(vector, vector.size), 1) for vector in cat_data['train']['cat']] + [(np.reshape(vector, vector.size), 0) for vector in cat_data['train']['no_cat']]
        random.shuffle(self.samples)
    
    def __iter__(self):
        
        return self
    
    def __next__(self):
        
        self.index += 1
        if self.index == len(self.samples):
            raise StopIteration
        return self.samples[self.index][0], self.samples[self.index][1]
    
    def shuffle(self):
        
        random.shuffle(self.samples)


# Run Script

# In[94]:


data = Cat_Data()
model = Cat_Model(activation=sigmoid)
trainer = Cat_Trainer(data, model)
costs, accuracies = trainer.train(0.000001, 500)
model.save_model()


# In[ ]:




