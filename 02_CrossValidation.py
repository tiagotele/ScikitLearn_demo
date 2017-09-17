from sklearn import datasets
from sklearn.cross_validation import train_test_split

iris = datasets.load_iris()
X = iris.data
y = iris.target

print(X.size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=4)

print(X_train.size)
print(X_test.size)


