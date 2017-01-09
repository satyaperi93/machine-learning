# -*- coding: utf-8 -*-
import pandas as pd 
import math
import random
def datapreprocessing():
    iris = pd.read_csv('irisdata.csv', header = None, names = ['sepallen','sepalwid','petallen','petalwid','class'] )

    randomdata=iris.sample(frac=1).reset_index(drop=1) # df mthd
    train = randomdata[0:100]
    test=randomdata[100:]
    return train, test
def dist(training,testing):
    distance = math.sqrt((testing[0]-training[0])**2+(testing[1]-training[1])**2+(testing[2]-training[2])**2+(testing[3]-training[3])**2)
    return distance
def knn(kvalue):
    for row in range(0,(len(train)-1)):
        currentrow = train.ix[row,0:5]
        distance = dist(currentrow, random_test_value)
        distancemeasured.append(distance)
        df_dist = pd.DataFrame(distancemeasured, columns=['distancemeasured'])
        df_dist['class'] = train['class']
        sorted_dist = df_dist.sort(columns=['distancemeasured'])
        unkownclass = sorted_dist['class'].head(kvalue)
        random_test_value['class_classified']=max(unkownclass.value_counts().index.tolist())
    return random_test_value['class_classified'], random_test_value['class']	      
# func calling
kvalue = input('enter the value')
train,test =  datapreprocessing()
correct = 0
for index in range(100,149):
    random_test_value = test.ix[index, 0:5]  
    distancemeasured = []
    prediction, testset = knn(kvalue)
    #print random_test_value['class_classified'], random_test_value['class']
    if prediction == testset:
        correct += 1
accuracy = (correct/float(len(test))) * 100.0
    