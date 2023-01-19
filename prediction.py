from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

dataset = pd.read_csv("..\dataset\\accidents_date_voie_flux_cat.csv", sep=",")

print(dataset)


X_train, X_test, y_train, y_test = train_test_split(dataset)


param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(
    forest_reg, param_grid, cv=5, scoring="mean_squared_error", return_train_score=True
)
grid_search.fit(X_train, y_train)

final_model = grid_search.best_estimator_

accident_predictions = forest_reg.predict(X_test)
forest_mse = mean_squared_error(X_test, accident_predictions)
forest_rmse = np.sqrt(forest_mse)
