# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:29:09 2019

@author: emmet
"""
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import KFold


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
path = "diamonds.csv"


#read from data using pandas
data = pd.read_csv(path)  
data = data.drop(data.columns[0], axis=1)

import pandas as pd


data = pd.read_csv('diamonds.csv')
data = data.drop(data.columns[0], axis=1)

#Task 1 
data.cut = data.cut.map({"Premium":0,"Ideal":1,"Very Good":2,"Good":3,"Fair":4})
data.color = data.color.map({"D":0,"E":1,"F":2,"G":3,"H":4,"I":5,"J":6})
data.clarity = data.clarity.map({"IF":0,"VVS2":1,"VVS1":2,"VS2":3,"VS1":4,"SI2":5,"SI1":6,"I1":7})
#print(data.head)
combinations = {(1,2,3)}
count=[]
for item in data.iterrows():
    cut = item[1][1]
    color = item[1][2]
    clarity = item[1][3]
    combinations.add((cut,color,clarity))
print("1")
#print (combinations)
count_comb=[]
features_train=[]
features_test=[]
target_train=[]
target_test=[]

for combination in combinations:
    count=0
    for item in data.iterrows():
        cut = item[1][1]
        color = item[1][2]
        clarity = item[1][3]
        if set(combination) & set((cut,color,clarity)):
            count+=1
    count_comb.append(count)
  
for i,combination in enumerate(combinations):
    features=[]
    target=[]
    count=count_comb[i]
    idx = int(0.80*count)
    for item in data.iterrows():
        carat=item[1][0]
        depth=item[1][4]
        table=item[1][5]
        sp=item[1][6]
        features.append([carat,depth,table])
        target.append(sp)
    if count>800:
        features_train.extend(features[:idx])
        features_test.extend(features[idx:])
        target_train.extend(target[:idx])
        target_test.extend(target[idx:])
print(len(features_train), " ", len(target_test))

def num_coefficients_2(d):
    t = 0
    for n in range(d+1):
        for i in range(n+1):
            for j in range(n+1):
                for k in range(n+1):
                    if i+j+k==n:
                        t = t+1
    return t
#Task 2
def calculate_model_function(deg, data, p):
    print(data.shape, "MODEL")
    result = np.zeros(data.shape[0])
    print(data.shape)
    t = 0
    for n in range(deg+1):
        for i in range(n+1):
            for j in range(n+1):
                for k in range(n+1):
                    if i + j + k == n:
                        result += p[t]*(data[:, 0]**i)*(data[:, 1]**j)*(data[:, 2]**k)
                        t += 1
                        print (result)
    return result

#Task3
def linearize(deg, data, p0):
    f0 = calculate_model_function(deg, data, p0)
    j = np.zeros((len(f0), len(p0)))
    epsilon = 1e-6
    for i in range(len(p0)):
        p0[i] += epsilon
        fi = calculate_model_function(deg, data, p0)
        p0[i] -= epsilon
        di = (fi - f0)/epsilon
        j[:, i] = di
        print (f0)
        print (j)
    return f0, j
#Task 4
def calculate_update(y,f0,J):
    l=1e-2
    N = np.matmul(J.T,J) + l*np.eye(J.shape[1])
    r = y-f0
    n = np.matmul(J.T,r)    
    dp = np.linalg.solve(N,n)       
    return dp


#Task 6
def kfold(data,target):
    print("Main 1")
    print(data.shape)
    kf = KFold(n_splits=2)
    max_iter = 5
    for train_index, test_index in kf.split(data):
         
        for deg in range(5):
            p0 = np.zeros(num_coefficients_2(deg))
            for i in range(max_iter):
                f0,J = linearize(deg,data[train_index], p0)
                dp = calculate_update(target[train_index],f0,J)
                p0 += dp
    
            x, y,z = np.meshgrid(np.arange(np.min(data[test_index,0]), np.max(data[test_index,0]), 0.1), 
                           np.arange(np.min(data[test_index,1]), np.max(data[test_index,1]), 0.1),
                           np.arange(np.min(data[test_index,2]), np.max(data[test_index,2]), 0.1),indexing='ij' )   
            test_data = np.array([x.flatten(),y.flatten(),z.flatten()]).transpose()                
            test_target = calculate_model_function(deg,test_data, p0)
            print(x.shape)
            print(y.shape)
            print(test_target.shape)
            x=x.reshape(x.shape[1],x.shape[0]*x.shape[2])
            y=y.reshape(y.shape[1],y.shape[0]*y.shape[2])
            z=z.reshape(z.shape[1],z.shape[0]*z.shape[2])
            print(x.shape)
            print(y.shape)
            print(z.shape)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data[test_index,0],data[test_index,1],data[test_index,2],target[test_index],c='r')
            ax.plot_surface(x,y,test_target.reshape(x.shape))
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data[test_index,0],data[test_index,1],data[test_index,2],target[test_index],c='r')
            ax.plot_surface(y,z,test_target.reshape(y.shape))
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data[test_index,0],data[test_index,1],data[test_index,2],target[test_index],c='r')
            ax.plot_surface(x,z,test_target.reshape(z.shape))
            print("Main 1")
            print(data.shape)

kfold(np.concatenate([features_train,features_test]),np.concatenate([target_train,target_test]))
#Task  5

def regression(data,target,data_test,target_test):
    print("HERE",data.shape)

    max_iter = 5
    for deg in [0,1,2,3,4]:
        p0 = np.zeros(num_coefficients_2(deg))
        for i in range(max_iter):
            f0,J = linearize(deg,data, p0)
            dp = calculate_update(target,f0,J)
            p0 += dp            

        x, y,z = np.meshgrid(np.arange(np.min(data_test[:,0]), np.max(data_test[:,0]), 0.1), 
                           np.arange(np.min(data_test[:,1]), np.max(data_test[:,1]), 0.1),
                           np.arange(np.min(data_test[:,2]), np.max(data_test[:,2]), 0.1),indexing='ij' )   
        test_data = np.array([x.flatten(),y.flatten(),z.flatten()]).transpose()                
        test_target = calculate_model_function(deg,test_data, p0)
        x=x.reshape(x.shape[1],x.shape[0]*x.shape[2])
        y=y.reshape(y.shape[1],y.shape[0]*y.shape[2])
        z=z.reshape(z.shape[1],z.shape[0]*z.shape[2])
        print(x.shape)
        print(y.shape)
        print(test_target.shape)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data_test[:,0],data_test[:,1],data_test[:,2],target_test,c='r')
        ax.plot_surface(x,y,test_target.reshape(x.shape))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data_test[:,0],data_test[:,1],data_test[:,2],target_test,c='r')
        ax.plot_surface(y,z,test_target.reshape(y.shape))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data_test[:,0],data_test[:,1],data_test[:,2],target_test,c='r')
        ax.plot_surface(x,z,test_target.reshape(x.shape))    
        
    
regression(np.array(features_train),np.array(target_train),np.array(features_test) ,np.array(features_train))   