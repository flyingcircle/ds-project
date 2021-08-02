from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor


def train_ensemble_voter(data, targets):
  reg1 = GradientBoostingRegressor(random_state=1)
  reg2 = RandomForestRegressor(random_state=1)
  reg3 = LinearRegression()
  ereg = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])
  return ereg.fit(data, targets)