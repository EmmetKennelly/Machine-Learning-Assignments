# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 21:07:03 2019

@author: emmet
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 21:07:03 2019

@author: emmet
"""
import pandas as pd
import xlrd
from pandas import ExcelWriter
from pandas import ExcelFile
from collections import Counter
import numpy as np
import parser as p
import os
import re
import nltk
import collections
import math
from sklearn import metrics
from sklearn import neighbors
from sklearn import model_selection


#Global variales
path = "movie_reviews.xlsx"
number = 0
posFreq = []
negFreq = []
newReview = ["This is a new Review, used for predicition"]
correctn = 0
minWordLength = 5
minWordOccurence = 20000



data = pd.read_excel(path)    
totalindex = data['Split']
reviews = data["Review"]
filterindextrain = ((data['Split'] == 'train') )
filterindextest = ((data['Split'] == 'test') )
positive = ((data['Sentiment'] == 'positive') )
negative = ((data['Sentiment'] == 'negative') )
positivereviews = (data[positive & reviews])
negativereviews = (data[negative & reviews])
noOfpositivereviews =len (data[positive & reviews])
noOfnegativereviews =len (data[negative & reviews])
train_index = np.arange(int(len(reviews)))    
test_index = np.arange(int(len(reviews)), len(reviews))

train = (data[filterindextrain & reviews])
test = (data[filterindextest & reviews])
train_index = np.arange(int(len(train)))    
test_index = np.arange(int(len(test)))

numberpostivetestreviews = len(data[positive & filterindextest])
numberpositivetrainreviews = len(data[positive & filterindextrain])
numbernegativetestreviews = len(data[negative & filterindextest])
numbernegativetrainreviews = len(data[negative & filterindextrain])


#Method used for Task 1 
def count():
 
 print("Column headings: " )
 
 print ("Total Reviews : ", len (totalindex))
 train 
 print("Train Reviews:", len (data[filterindextrain]))
 print("Test Reviews:",len (data[filterindextest]))
 print ("Positive Sentiments" ,noOfpositivereviews)
 print ("Negative Sentiments" ,noOfnegativereviews)
 
 print("Positive Reviews if test:", numberpostivetestreviews )
         
 print("Positive Reviews if train:",numberpositivetrainreviews)
         
 print("Negative Reviews if test:", numbernegativetestreviews )
       
 print("Negative Reviews if train:", numbernegativetrainreviews)
 

   
 #Method used for Task 2
def cleanWords(data):

     cleanedData = (data["Review"]).str.replace('[^a-zA-Z0-9]', ' ').str.lower()
     cleanedData.str.split()
     data['Review']  = cleanedData
     return data
 
def countWords(data):
   
    count = (data['Review']).str.len()
   
    return count

 
 
#Method used for Task 2
def countOccurences(data,minWordLength, minWordOccurence):
 reviewWords = data['Review']
 dataindex = (data['Split'] == 'train')
 data = data[dataindex == True]
 wordOccurences = {}
 wordList = []
 
 allWords = (reviewWords).str.split()
 
 
 for review in allWords:

   for word in review:

    if (len(word)>=minWordLength):
       
         if (word in wordOccurences):
               
                wordOccurences[word] = wordOccurences[word] + 1
               
         else:
                wordOccurences[word]=1    
 
 for word in wordOccurences:
 
      if wordOccurences[word]>=minWordOccurence:
           
        
      
      
      #
      # print( "The word : "+ word + "  appeared : " + str(wordOccurences[word])+" times" )
       wordList.append(word)
         
         
   
 return wordList


#Method for Task 3
def posCount(data,wordOccur):
  
   wordFind= []
   #[None] * 1000
   #for i in range(1000):
  #  wordFind[i] = [None] * 1000
    
   reviews = data['Review']
   dataindex = (data['Sentiment'] == 'positive')
   posData = reviews[dataindex]
   
   for review in posData:  
      words = review.split()
      for word in wordOccur:
         
          if word in words:
               wordFind.append(word)
         
   wordCount = Counter(wordFind)     
   for word in wordCount:
       print ("The number of positive reviews that had  : "+ word  +" is "+ str(wordCount[word]))
   
   print ("\n")
   return wordCount
 

 #Method for Task 3
def negaCount(data,wordOccur):
  
   wordFind= []
   #[None] * 1000
   #for i in range(1000):
  #  wordFind[i] = [None] * 1000
    
   reviews = data['Review']
   dataindex = (data['Sentiment'] == 'negative')
   negData = reviews[dataindex]
   
   for review in negData:  
      words = review.split()
      for word in wordOccur:
         
          if word in words:
               wordFind.append(word)
         
   wordCount = Counter(wordFind)     
   for word in wordCount:
       print ("The number of negative reviews that had  : "+ word  +" is "+ str(wordCount[word]))
   
   
   return wordCount  
           #  print( "The word : "+ word + "  appeared : " + pos +" positive reviews" )
 
 
   

       
   
   
   
   
   
   #Method for Task 4
def calcFreq(posCount, negaCount):
   alpha = 1
   
   positivePrior = str((numberpositivetrainreviews + alpha ) / (noOfpositivereviews +  2*alpha))
   negativePrior = str((numbernegativetrainreviews + alpha) / ( noOfnegativereviews + 2*alpha))
   print ("The Positive Prior :" + positivePrior)
   print ("The Negative Prior :" + negativePrior)
    
    
   for word in posCount:
       alpha = 1
       likelihood_positive = (posCount[word] + alpha) / (numberpositivetrainreviews + 2*alpha)
       likelihood_negative = (negaCount[word] + alpha) / (numbernegativetrainreviews + 2*alpha)
       
       print("Likelihood positive if countains: "+ word + " ", likelihood_positive)
       
       
       posFreq.append(likelihood_positive) 
    
       print("Likelihood negative if contains: "+ word + " ", likelihood_negative)
       
       negFreq.append(likelihood_negative) 
       
       
       print("\n")
  
   return posFreq ,negFreq, positivePrior, negativePrior
        

#Method for Task 5
def maxLikelihood(posFreq, negFreq,positivePrior , negativePrior, updatedReviews):
   i=0
   j=0
   
   prediction = []
   
   for review in train_index:
      while i < len (posFreq) and j < len(negFreq): 
       posNums = posFreq[i]
       
       negNums = negFreq[j]
      
       
       logLikelihood_positiveword = 0
       logLikelihood_negativeword = 0
           
           
       if data.iloc[review]["Sentiment"]=="positive":
               
                    logLikelihood_positiveword = logLikelihood_positiveword + math.log(posNums + 1) 
            
       elif data.iloc[review]["Sentiment"]=="negative":
            
                   logLikelihood_negativeword = logLikelihood_negativeword + math.log(negNums + 1)
           
       i += 1
       j += 1  

      if (logLikelihood_positiveword - logLikelihood_negativeword) >( math.log(float(negativePrior)) - math.log(float(positivePrior))):
            prediction.append("positive")
      else:
            prediction.append("negative")
 
   updatedTest = (updatedReviews[(data['Split'] == 'test') & reviews])
   print(updatedTest)
   accuracy = metrics.accuracy_score(updatedTest , prediction)
   print(accuracy)   
   return accuracy, prediction


 #Method for Task 6  
def crossValid(minWordLength, minWordOccurence,accuracy,prediction):
    data[["Review","Split"]]
    data["Review"] = data["Review"].map({"positive","negative"})    

    target = data["Review"]
   
    positive = len(data[target=="positive"])
    negative = len(data[target=="negative"])
               
    kf = model_selection.StratifiedKFold(n_splits=min(positive,negative), shuffle=True)
    cvX = []
    cvY = []
    for k in range(1,10):
        true_positive = []
        true_negative= []
        false_positive = []
        false_negative= []
    
    for train_index, test_index in kf.split(data, reviews):
            clf = neighbors.KNeighborsClassifier(n_neighbors=k)
            clf.fit(data.iloc[train_index], data[train_index])
            predicted_labels = clf.predict(data.iloc[test_index])
            
            C = metrics.confusion_matrix(reviews[test_index], predicted_labels)
            
            true_positive.append(C[1,1])
            true_negative.append(C[0,0])            
            false_positive.append(C[0,1])
            false_negative.append(C[1,0])
        
            print("K fold validation =",k)
            print("True positive:", np.sum(true_positive))
            print("True negative:", np.sum(true_negative))
            print("False positive:", np.sum(false_positive))
            print("False negative:", np.sum(false_negative))
            print("Accuracy of Model with Cross Validation is:",accuracy.mean() * 100)
                    
            cvX.append(np.sum(false_positive))
            cvY.append(np.sum(true_positive))
            
        
    print(cvX)
    print(cvY)
    
    confusion = metrics.confusion_matrix(data[test_index], prediction)
    print(confusion)
    
       
def main():
  
    
    count()
    cleanWords(data)
    countWords(cleanWords(data))
   
    posFreq , negFreq, positivePrior, negativePrior =  calcFreq(posCount(data,countOccurences(data,minWordLength, minWordOccurence)) , negaCount(data,countOccurences(data,minWordLength, minWordOccurence)) )
    
    newTestReview = pd.DataFrame({'Review':newReview,'Sentiment':[""], 'Split':["test"]})   
    updatedReviews =data.append(newTestReview)
  
    crossValid(countOccurences(data,minWordLength, minWordOccurence),maxLikelihood(posFreq, negFreq, positivePrior , negativePrior,updatedReviews ))

main()
