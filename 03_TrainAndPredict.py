from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=4)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predicted = knn.predict(X_train)

print(y_train)
print(predicted)

