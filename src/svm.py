from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def train_svm(data, targets):
  regr = make_pipeline(StandardScaler(), LinearSVR())
  return regr.fit(data, targets)

