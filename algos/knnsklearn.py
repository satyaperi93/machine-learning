import pandas as pd 
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
# import the data set 
# clean data check for null values 
# remove unwanted columns
#split the data for training and testng using crossvalidation
# call knn with a k value
#fit the model
#predict and score the model
breast_cancer = pd.read_csv('breast-cancer-wisconsindata.csv', names=['sample number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class'])
breast_cancer.replace('?',-99999, inplace=True)
breast_cancer.drop(['sample number'], 1, inplace=True)
# store feature matrix in "X"
X = np.array(breast_cancer.drop(['Class'], 1))
y = np.array(breast_cancer['Class'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

krange = list(range(1,30))
score = []
for k in krange:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    score.append(accuracy)
plt.plot(krange, score)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')

#example_measures = np.array([4,2,1,1,1,2,3,2,1])
#example_measures = example_measures.reshape(1, -1)
#prediction = knn.predict(example_measures)
#print(prediction)