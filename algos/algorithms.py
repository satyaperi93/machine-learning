# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
iris = load_iris()
print (iris.data)
print(iris.target)
print(iris.feature_names)
# store feature matrix in "X"
X = iris.data

# store response vector in "y"
y = iris.target