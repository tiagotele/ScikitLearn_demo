# http://scikit-learn.org/stable/datasets/index.html#datasets
from sklearn import datasets

print("--Iris--")
iris = datasets.load_iris()

print(iris.feature_names)
print(iris.data)
print(iris.target_names)
print(iris.target)


# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/data/boston_house_prices.csv
print("--Boston--")
boston = datasets.load_boston()
print(boston.feature_names)
print(boston.data)

print("--Digit--")
digits = datasets.load_digits()

# print(digits.feature_names)
print(digits.data)
print(digits.target)


import matplotlib.pyplot as plt 
plt.gray() 
plt.matshow(digits.images[5])
plt.show()