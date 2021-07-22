import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import metrics

import data
import svm
import util

# Collect Data
##################
daily = data.get_transit_daily()
daily = data.data_transforms(daily)
daily = daily[data.USED_COLS]
daily = pd.get_dummies(daily)  # turns all categoricals into one hot encoded columns!
label = daily.pop("label")
train_data, test_data, train_labels, test_labels = train_test_split(daily.values, label.values)

# Create PCA
##################
pca = PCA(4)
proj = pca.fit_transform(train_data)
test_proj = pca.transform(test_data)
print("sum of variance_ratio: ", np.sum(pca.explained_variance_ratio_))
print("shape of new data: ", proj.shape)

# Train SVM
###################
model = svm.train_svm(proj, train_labels)
results = model.predict(test_proj)

# Results
###################
util.plot(69,420,results,test_labels)
print("MAE: ", metrics.mean_absolute_error(test_labels,results))
print("MSE: ", metrics.mean_squared_error(test_labels,results))
print("R2: ", metrics.r2_score(test_labels,results))