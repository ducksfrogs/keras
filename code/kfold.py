import numpy as np
from sklearn.model_selection import KFold


x = np.array([[1,2],[3,4],[1,2],[3,4],[3,4],[3,4],[3,5],[3,6],[3,8]])
y = np.array([0,0,1,1,1,1,1,1,1,1])

kf = KFold(n_splits=5)
for train_index, test_index in kf.split(x, y):
    print("train_index:", train_index, "test_index", test_index)
