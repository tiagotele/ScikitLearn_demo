from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np

dataset=pd.read_csv('small_iris_missing_data.csv')

print(dataset)

X=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 4].values

np.set_printoptions(threshold=np.nan)

imputer=Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer=imputer.fit(X [:, 0:3] )
X[:, 0:3]=imputer.transform(X[:, 0:3])

print(X)


