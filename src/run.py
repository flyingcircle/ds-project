import pandas as pd
import numpy as np
import csv
import time

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import metrics

import data
import svm
import sgd
import voting_regressor
import util

start_time = time.time()

# Collect Data
##################
print("Collecting Data: %s\n" % (time.time() - start_time))
daily = data.get_transit_daily()
daily = data.data_transforms(daily)
daily = daily[data.USED_COLS]
daily = pd.get_dummies(daily)  # turns all categoricals into one hot encoded columns!
label = daily.pop("label")
train_data, test_data, train_labels, test_labels = train_test_split(daily.values, label.values)

# Create PCA
##################
print("Fitting PCA: %s\n" % (time.time() - start_time))
pca = PCA(4)
proj = pca.fit_transform(train_data)
test_proj = pca.transform(test_data)
print("sum of variance_ratio: ", np.sum(pca.explained_variance_ratio_))
print("shape of new data: ", proj.shape)

def run_model(model_name, train_func, num):
  # Train Model
  ###################
  model = train_func(proj, train_labels)
  results = model.predict(test_proj)

  # Model Results
  ###################
  print("Results: %s\n" % (time.time() - start_time))
  util.plot(69,420,results,test_labels, filename=model_name + "_results_" + num + ".png")
  print("MAE: ", metrics.mean_absolute_error(test_labels,results))
  print("MSE: ", metrics.mean_squared_error(test_labels,results))
  print("R2: ", metrics.r2_score(test_labels,results))

  # Model Write Results
  ###################
  file = open("../results/" + model_name + "_results_" + num + ".csv", "w")
  writer = csv.writer(file)
  for w in range(results.shape[0]):
    writer.writerow([results[w], test_labels[w]])

  file.close()
  print("Done: %s\n" % (time.time() - start_time))


print("Running SVM: %s\n" % (time.time() - start_time))
run_model("svm", svm.train_svm, "1")

print("Running SGD: %s\n" % (time.time() - start_time))
run_model("sgd", sgd.train_sgd, "1")

print("Running Voting Ensemble: %s\n" % (time.time() - start_time))
run_model("ens", voting_regressor.train_ensemble_voter, "1")
