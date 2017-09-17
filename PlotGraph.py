from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=4)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)

print (y_train.shape)
print (X_train[:,0].shape)
plt.scatter(X_train[:,0], y_train, color = 'red')
plt.plot(X_train, knn.predict(X_train), color = 'blue')
plt.title('Iris predict knn')
plt.xlabel('x')
plt.ylabel('y')
# plt.show()