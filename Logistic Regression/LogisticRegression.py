
# coding: utf-8

# In[122]:


import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import itertools
import operator
get_ipython().run_line_magic('matplotlib', 'inline')


# # Reading Data

# In[33]:


df_train = pd.read_csv("votes-train.csv")
df_test = pd.read_csv("votes-test.csv")
df_train.head(20)


# # Logistic Regression Function
# Trains the model using xtr (input) and ytr (target) and returns the accuracy test for both training (xtr, ytr)and testing samples (xts, yts)

# In[70]:



def logistic_calc(xtr, ytr, xts, yts):
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(xtr,ytr)
    yhat_tr = logreg.predict(xtr)
    yhat_ts = logreg.predict(xts)
    return accuracy_score(yhat_tr, ytr), accuracy_score(yhat_ts, yts)    


# # Logistic Regression Using all Features

# In[139]:


#Preparing training and testing inputs and target matrices
#using all features
xtr = np.array(df_train[df_train.columns[1:]])
ytr = np.array(df_train['Democrat'])
xts = np.array(df_test[df_test.columns[1:]])
yts = np.array(df_test['Democrat'])


#feature scaling of input matrices xtr, xts
xtr = (xtr - np.mean(xtr, axis=0))/(np.std(xtr, axis=0))
xts = (xts - np.mean(xts, axis=0))/(np.std(xts, axis=0))


accuracy_tr, accuracy_ts = logistic_calc(xtr, ytr, xts, yts)
print("Accuracy with Training samples=", accuracy_tr)
print("Accuracy with Test samples", accuracy_ts)


# # Subset Regression
# In this section, I trained and predicted Logistic Regression using all possible combinations of features which was factorial(9)=362880. Then I selected top 5 best models to analyze. 

# In[110]:


#returns np array made of columns named i feature list
def colum(df, feature):
    return np.column_stack([ np.array(df[f]) for f in feature])


# In[120]:


features = df.columns[1:]
all_accuracy = {}
for r in range(1, len(features)+1):
    all_comb = itertools.combinations(features, r)
    print(all_comb)
    for comb in all_comb:
        
        xtr = colum(df_train, comb)
        ytr = np.array(df_train['Democrat'])
        xts = colum(df_test, comb)
        yts = np.array(df_test['Democrat'])


        #feature scaling of input matrices xtr, xts
        xtr = (xtr - np.mean(xtr, axis=0))/(np.std(xtr, axis=0))
        xts = (xts - np.mean(xts, axis=0))/(np.std(xts, axis=0))


        accuracy_tr, accuracy_ts = logistic_calc(xtr, ytr, xts, yts)
        all_accuracy[comb] = (accuracy_tr + accuracy_ts)/2 


# In[150]:


#sorting all results with accuracy
sorted_x = sorted(all_accuracy.items(), key=operator.itemgetter(1))
sorted_x.reverse()
for i in range(0, len(sorted_x)):
    print("Features\n", [key for key in pair[0]], sep='')
    print("Avg. Accuracy=", pair[1], '\n')


# # Ploting Best 500 Accuracy

# In[203]:


best_acc =[]
best_feature = []

for pair in sorted_x[:500]:
    best_acc.append(pair[1])
    best_feature.append(pair[0])

plt.plot([i for i in range(1,501)], best_acc)
plt.ylabel('Accuracy')
plt.xlabel('Rank')


# In[209]:


plt.plot([i for i in range(1,501)], [len(i) for i in best_feature])
plt.ylabel('Best Performance : Number of Features')
plt.xlabel('Rank')


# In[204]:


print("Followings are top 5 features combination\n")
for i in range(5):
    print(best_feature[i],end='\n\n')


# # Ploting 20 Worst Accuracy 

# In[216]:


worst_acc =[]
worst_feature = []

for pair in sorted_x[len(sorted_x)-1:len(sorted_x)-21:-1]:
    worst_acc.append(pair[1])
    worst_feature.append(pair[0])

plt.plot([i for i in range(20,0,-1)], worst_acc)
plt.ylabel('Accuracy')
plt.xlabel('Rank')



# In[217]:


plt.plot([i for i in range(20, 0, -1)], [len(i) for i in worst_feature])
plt.ylabel('Worst Performance : Number of Features')
plt.xlabel('Rank')


# In[218]:


print("Followings are 20 worst feature combinations")
for f in worst_feature:
    print(f)


# # Ploting Best Pair Feature Accuracy

# In[252]:


#Finding top accuracy with two features
best_pair = []
i = 0
while(len(best_pair) < 10 and i <len(sorted_x)):
    if(len(sorted_x[i][0]) ==2):
        best_pair.append(sorted_x[i])
    i+=1
    
print("Best pair features=\n")
for pair in best_pair:
    print(pair[0][0], pair[0][1], sep=" & ")
    print("Accuracy=",pair[1], end="\n\n")


# In[288]:


#Plots two features with class
#Modified version from Intro to ML (see refrences in report) example codes
def plot_cnt(X,y):
    
    # Compute the bin edges for the 2d histogram
    x0val = np.array(list(set(X[:,0]))).astype(float)
    x1val = np.array(list(set(X[:,1]))).astype(float)
    x0, x1 = np.meshgrid(x0val,x1val)
    x0e= np.hstack((x0val,np.max(x0val)+1))
    x1e= np.hstack((x1val,np.max(x1val)+1))

    # Make a plot for each class
    yval = list(set(y))
    color = ['b','r']
    for i in range(len(yval)):
        I = np.where(y==yval[i])[0]
        cnt, x0e, x1e = np.histogram2d(X[I,0],X[I,1],[x0e,x1e])
        x0, x1 = np.meshgrid(x0val,x1val)
        plt.scatter(x0.ravel(), x1.ravel(), s=2*cnt.ravel(),alpha=0.5,
                    c=color[i],edgecolors='none')
    plt.legend(['Democrat','Republican'], loc='upper right')
    #plt.xlabel(xnames[0], fontsize=16)
    #plt.ylabel(xnames[1], fontsize=16)
    return plt


# In[297]:


def plot_norm(x,y,ax):
    Iben = (y==0)
    Imal = (y==1)
    ax.plot(x[Imal,0],x[Imal,1],'b.')
    ax.plot(x[Iben,0] ,x[Iben,1],'r.')
    ax.legend(['Democrat', 'Republican'])


# In[298]:


for i in range(10):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x= np.column_stack((np.array(df_train[best_pair[i][0][0]]),np.array(df_train[best_pair[i][0][1]])))    
    y=np.array(df_train['Democrat'])
    #ax.scatter()
    plot_norm(x,y, ax)
    plt.xlabel(best_pair[i][0][0])
    plt.ylabel(best_pair[i][0][1])
    #break

