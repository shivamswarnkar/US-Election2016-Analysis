
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Activation
import keras.backend as K
from keras import optimizers


# # Load & Scale Data

# In[2]:


df_train = pd.read_csv("votes-train.csv")
df_test = pd.read_csv("votes-test.csv")
df_train.head(20)


# In[23]:


#Preparing training and testing inputs and target matrices
#using all features
xtr = np.array(df_train[df_train.columns[1:]])
ytr = np.array(df_train['Democrat'])
xts = np.array(df_test[df_test.columns[1:]])
yts = np.array(df_test['Democrat'])


#feature scaling of input matrices xtr, xts
xtr = (xtr - np.mean(xtr, axis=0))/(np.std(xtr, axis=0))
xts = (xts - np.mean(xts, axis=0))/(np.std(xts, axis=0))


# # Creating Neural Network

# In[28]:


K.clear_session()
nh=1000             #number of hidden units
N = xtr.shape[1]  #number of features, aka number of input neurons
M = 2             #number of output neurons. 
model = Sequential()
model.add(Dense(nh, input_shape=(N,), activation='sigmoid', name='hidden'))
model.add(Dense(M, activation="softmax", name="output"))

opt = optimizers.Adam(lr=0.001)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()


# # Training the Neural Network

# In[30]:


model.fit(xtr, ytr, epochs=291, batch_size=100, validation_data=(xts, yts))


# In[33]:


score, acc1 = model.evaluate(xts, yts, verbose=0)
print("Accuracy on testing samples=", acc1)


# # Training with the best feature combination from subset regression

# In[36]:


#returns np array made of columns named i feature list
def colum(df, feature):
    return np.column_stack([ np.array(df[f]) for f in feature])


# In[39]:


col = ('population', 'population_change', 'Black', 'Hispanic', 'Edu_bachelors', 'income', 'Poverty', 'Density')
xtr_best = colum(df_train, col)
xts_best = colum(df_test, col)

xtr_best = (xtr_best - np.mean(xtr_best, axis=0))/np.std(xtr_best, axis=0)
xts_best = (xts_best - np.mean(xts_best, axis=0))/np.std(xts_best, axis=0)


# In[40]:


K.clear_session()
nh=1000             #number of hidden units
N = xtr_best.shape[1]  #number of features, aka number of input neurons
M = 2             #number of output neurons. 
model = Sequential()
model.add(Dense(nh, input_shape=(N,), activation='sigmoid', name='hidden'))
model.add(Dense(M, activation="softmax", name="output"))

opt = optimizers.Adam(lr=0.001)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()


# In[46]:


model.fit(xtr_best, ytr, epochs=200, batch_size=100, validation_data=(xts_best, yts))


# In[62]:


score, acc2 = model.evaluate(xts_best, yts, verbose=0)
print("Accuracy on testing samples for best Feature Combo=", acc2)


# # Training with Worst Feature Combo from subset regression

# In[57]:


xtr_worst = np.array(df_train["age65plus"])
xts_worst = np.array(df_test["age65plus"])

xtr_worst = (xtr_worst - np.mean(xtr_worst, axis=0))/np.std(xtr_worst, axis=0)
xts_worst = (xts_worst - np.mean(xts_worst, axis=0))/np.std(xts_worst, axis=0)


# In[59]:


K.clear_session()
nh=1000             #number of hidden units
N = 1 #number of features, aka number of input neurons
M = 2             #number of output neurons. 
model = Sequential()
model.add(Dense(nh, input_shape=(N,), activation='sigmoid', name='hidden'))
model.add(Dense(M, activation="softmax", name="output"))

opt = optimizers.Adam(lr=0.001)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()


# In[61]:


model.fit(xtr_worst, ytr, epochs=200, batch_size=100, validation_data=(xts_worst, yts))


# In[63]:


score, acc3 = model.evaluate(xts_worst, yts, verbose=0)
print("Accuracy on testing samples for best Feature Combo=", acc3)


# # Plotting All Features, Best featuers and Worst Features training accuracy

# In[96]:


plt.barh([1,2,3],[acc1,acc2,acc3],color=['red', 'blue', 'green'])
plt.yticks([1,2,3], ["All Features", "Top Feature Comb", "Worst Feature Comb"])
plt.xlabel("Accuracy")

f = plt.figure()
ax = f.add_subplot(111)
ax.plot(["All Features","Top Features","Worst Feature"],[acc1,acc2,acc3], color='orange')
ax.scatter(["All Features","Top Features","Worst Feature"],[acc1,acc2,acc3], color=['red', 'blue', 'green'])
plt.ylabel("Accuracy")

