# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:11:49 2019

@author: emmet
"""

import pandas as pd
import time
from sklearn import metrics
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import KFold
from PIL import Image
import numpy as np


path = "product_images.csv"

#read from data using pandas
data = pd.read_csv(path)   
#here we split the data and get the total sample size
labels = data.iloc[:,0]
print (labels.iloc[0])
features = data.iloc[:,1:]
print (len(features))
#getting the number of values that match data needed for a ankle boot
print ("Sum of data for ankle boot: ",sum(x.count("1") for x in features))
#getting the number of values that match data needed for a sneaker
print ("Sum of data for sneaker: ",sum(x.count("0") for x in features),"\n")
# Image of Sneaker - data[3] since 4th(3rd by index) row has label 0
srow =data.iloc[3]
sneaker_matrix =[srow[start:start+28]for start in range(1,785,28)]
sneakerarray = np.array(sneaker_matrix, dtype=np.uint8)
# Use PIL to create an image from the new array of pixels
sneaker_image = Image.fromarray(sneakerarray)
sneaker_image.save('sneaker.png')
print("The image of a sneaker has been saved to your computer \n")
# Image of Ankle boot - data[0] since first row has label 1
brow = data.iloc[0]
boot_matrix = [brow[start:start+28] for start in range(1,785,28)] 
bootarray = np.array(boot_matrix, dtype=np.uint8)
# Use PIL to create an image from the new array of pixels
boot_image = Image.fromarray(bootarray)
boot_image.save('boot.png')
print("The image of a boot has been saved to your computer \n")

def getFeatures(noOfSamples,features,labels):
     #arrays to store time of each classifier
     mlptimes = []
     svmtimes = []
     featureToUse = features.iloc[:noOfSamples,:]
     labelsToUse = labels.iloc[:noOfSamples]
    # plt.imshow(features.iloc[0,:].reshape(8,8))
     print ("Sample size used is :", noOfSamples)
     # Setting up our SVM with gamm values, linear wont be affected by a gamma 
     clf1 = linear_model.Perceptron()       
     clf2 = svm.SVC(kernel="rbf", gamma= 0.01)      
     clf3 = svm.SVC(kernel="linear")   
     clf4 = svm.SVC(kernel="rbf", gamma= 0.001)      
     clf5 = svm.SVC(kernel="linear") 
     clf6 = svm.SVC(kernel="rbf", gamma= 0.000001)      
     clf7 = svm.SVC(kernel="linear") 
     clf8 = svm.SVC(kernel="rbf", gamma= 0.0000000001)      
     clf9 = svm.SVC(kernel="linear") 
     # Create a Kfold for 9 splits
     cv = KFold(n_splits =9)
     counter = 1
     #for loop that gets the training data needed from the data
     for train_index, test_index in cv.split(featureToUse):
         trainX = featureToUse.iloc[train_index]
         trainY = labelsToUse.iloc[train_index]
         testX = featureToUse.iloc[test_index]
         testY = labelsToUse.iloc[test_index]
         #for each of these, we time how long it take using time import
         start1 = time.clock()
         clf1.fit(trainX,trainY) 
         end1 = time.clock()
         
         start2 = time.clock()
         clf2.fit(trainX,trainY) 
         end2 = time.clock()
         
         start3 = time.clock()
         clf3.fit(trainX,trainY) 
         end3 = time.clock()
         
         start4 = time.clock()
         clf4.fit(trainX,trainY)
         end4 = time.clock()
         
         start5 = time.clock()
         clf5.fit(trainX,trainY)
         end5 = time.clock()
         
         start6 = time.clock()
         clf6.fit(trainX,trainY) 
         end6 = time.clock()
         
         start7 = time.clock()
         clf7.fit(trainX,trainY)  
         end7 = time.clock()
         
         start8 = time.clock()
         clf8.fit(trainX,trainY)  
         end8 = time.clock()
         
         start9 = time.clock()
         clf9.fit(trainX,trainY)  
         end9 = time.clock()
         
         pstart1 = time.clock()
         pred1 =clf1.predict(testX)
         pend1 = time.clock()
         pstart2 =time.clock()
         pred2 =clf2.predict(testX)
         pend2 = time.clock()
         pstart3 = time.clock()
         pred3 =clf3.predict(testX)
         pend3 = time.clock()
         pstart4 = time.clock()
         pred4 =clf4.predict(testX)
         pend4 = time.clock()
         pstart5 = time.clock()
         pred5 =clf5.predict(testX)
         pend5 = time.clock()
         pstart6 = time.clock()
         pred6 =clf6.predict(testX)
         pend6 = time.clock()
         pstart7 =time.clock()
         pred7 =clf7.predict(testX)
         pend7 =time.clock()
         pstart8 =time.clock()
         pred8 =clf8.predict(testX)
         pend8 =time.clock()
         pstart9 =time.clock()
         pred9 =clf9.predict(testX)
         pend9 =time.clock()
         # appending each time to our MLP time list
         mlp1 =(end1-start1)*1000
         mlptimes.append(mlp1)
         mlp2 =(end2-start2)*1000
         mlptimes.append(mlp2)
         mlp3 =(end3-start3)*1000
         mlptimes.append(mlp3)
         mlp4 =(end4-start4)*1000
         mlptimes.append(mlp4)
         mlp5 =(end5-start5)*1000
         mlptimes.append(mlp5)
         mlp6 =(end6-start6)*1000
         mlptimes.append(mlp6)
         mlp7 =(end7-start7)*1000
         mlptimes.append(mlp7)
         mlp8 =(end8-start8)*1000
         mlptimes.append(mlp8)
         mlp9 =(end9-start9)*1000
         mlptimes.append(mlp9)
         #All time for the MLP with a mx, min and avrg
         mlpMax = max(mlptimes)
         mlpMin = min(mlptimes)
         mlpAverage = sum(mlptimes) / len(mlptimes)
         print("\n For split ",counter,"\n")  
         print("MLP maximum time : ", mlpMax, "millseconds")
         print("MLP minimum time : ", mlpMin, "millseconds")
         print("MLP average time : ", mlpAverage, "millseconds \n")
       #  print("MLP training time 2: ", (end2-start2)*1000, "millseconds")
        # print("MLP training time 3: ", (end3-start3)*1000, "millseconds")
         #print("MLP training time 4: ", (end4-start4)*1000, "millseconds")
         #print("MLP training time 5: ", (end5-start5)*1000, "millseconds")
         #print("MLP training time 6: ", (end6-start6)*1000, "millseconds")
         #print("MLP training time 7: ", (end7-start7)*1000, "millseconds")
         #Obtaining our accuracy scores
         score1 = metrics.accuracy_score(pred1, testY)      
         score2 = metrics.accuracy_score(pred2 , testY)  
         score3 = metrics.accuracy_score(pred3 , testY)    
         score4 = metrics.accuracy_score(pred4 , testY)   
         score5 = metrics.accuracy_score(pred5 , testY)
         score6 = metrics.accuracy_score(pred6 , testY)   
         score7 = metrics.accuracy_score(pred7 , testY)  
         score8 = metrics.accuracy_score(pred8 , testY)   
         score9 = metrics.accuracy_score(pred9 , testY)   
         # appending each time to our SVM time list
         svm1 =(pend1-pstart1)*1000
         svmtimes.append(svm1)
         svm2 =(pend2-pstart2)*1000
         svmtimes.append(svm2)
         svm3 =(pend3-pstart3)*1000
         svmtimes.append(svm3)
         svm4 =(pend4-pstart4)*1000
         svmtimes.append(svm4)
         svm5 =(pend5-pstart5)*1000
         svmtimes.append(svm5)
         svm6 =(pend6-pstart6)*1000
         svmtimes.append(svm6)         
         svm7 =(pend7-pstart7)*1000
         svmtimes.append(svm7)
         svm8 =(pend8-pstart8)*1000
         svmtimes.append(svm8)         
         svm9 =(pend9-pstart9)*1000
         svmtimes.append(svm9)
         
         svmMax = max(svmtimes)
         svmMin = min(svmtimes)
         svmAverage = sum(svmtimes) / len(svmtimes)
         print("For split ",counter,"\n")  
         print("SVM maximum time : ", svmMax, "millseconds")
         print("SVM minimum time : ", svmMin, "millseconds")
         print("SVM average time : ", svmAverage, "millseconds \n")
         #printing out scores and confusion matrix across our split
         print("Perceptron accuracy score : ", score1)      
         print("Perceptron confusion matrix : ", metrics.confusion_matrix(testY,pred1),"\n")
         print("SVM with RBF kernel accuracy score with gamma value 0.01 : ", score2)   
         print("SVM confusion matrix : ", metrics.confusion_matrix(testY,pred2),"\n")
         print("SVM with Linear kernel accuracy score with gamma value 0.01 : ", score3) 
         print("SVM confusion matrix : ", metrics.confusion_matrix(testY,pred3),"\n")
         print("SVM with RBF kernel accuracy score with gamma value 0.001 : ", score4) 
         print("SVM confusion matrix : ", metrics.confusion_matrix(testY,pred4),"\n")
         
         print("SVM with Linear kernel accuracy score with gamma value 0.001: ", score5)
         print("SVM confusion matrix : ", metrics.confusion_matrix(testY,pred5),"\n")
         print("SVM with RBF kernel accuracy score with gamma value 0.0001 : ", score6) 
         print("SVM confusion matrix : ", metrics.confusion_matrix(testY,pred6),"\n")
         print("SVM with Linear kernel accuracy score with gamma value 0.0001: ", score7)
         print("SVM confusion matrix : ", metrics.confusion_matrix(testY,pred7),"\n")
         print("SVM with RBF kernel accuracy score with gamma value  0.000001 : ", score8)
         print("SVM confusion matrix : ", metrics.confusion_matrix(testY,pred8),"\n")
         print("SVM with Linear kernel accuracy score with gamma value  0.000001: ", score9)  
         print("SVM confusion matrix : ", metrics.confusion_matrix(testY,pred9),"\n")
         print()
        # train_Time = train_index.pop('exec_time')
        # print(train_Time)
        #finding Max and min value in a training data during a given split
         print ("Maximum value in the training set: ", max(train_index))
         print ("Minimum value in the training set: ", min(train_index),"\n")
         counter = counter+1
    
    
    
    
#plt.imshow(data[0,:].reshape(8,8))   
#kf = model_selection.KFold(n_splits=2, shuffle=True)   
''' 
for image in kf.split(data):         
     clf1 = linear_model.Perceptron()       
     clf2 = svm.SVC(kernel="rbf", gamma= 0.001)      
     clf3 = svm.SVC(kernel="linear", gamma= 0.001)      
     data = pd.read_csv("product_images.csv")    
     shoe = data[data["pixel"]==0]
     boot = data[data["pixel"]==1]
     '''
     #A for loop we use to  loop around a list of diffrenet samples to use in our predictions
sampleSize = [500,1000,2000,5000,10000]
for i in range(len(sampleSize)):
 getFeatures(sampleSize[i],features,labels)
 i= i+1
 
