from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def train_sgd(data, targets):
  regr = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
  return regr.fit(data, targets)

