import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

data = pd.read_csv("train.csv")
y = data["y"].to_numpy()
data = data.drop(columns="y")
# print a few data samples
#print(data.head())

X = data.to_numpy()
# The function calculating the average RMSE
lambdas = [0.1, 1, 10, 100, 200]
n_folds = 10


x = X.T


I = np.eye(X.shape[1])

w = np.linalg.inv(X.T.dot(X) + 0.1 * I).dot(X.T.dot(y))
assert w.shape == (13,)
print(w)
