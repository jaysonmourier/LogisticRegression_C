import numpy as np
from sklearn.datasets import make_classification

# generate artificial data for classfication
X, y = make_classification(n_samples = 1000, n_features = 10, n_classes=2)

# Export to csv
data = np.concatenate((X, np.reshape(y, (y.shape[0], 1))), axis=1)
np.savetxt("data.txt", data, delimiter=',')