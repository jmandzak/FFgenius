# some of the grid search taken from 
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

from numpy.core.fromnumeric import mean
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from math import sqrt

QBs = pd.read_csv("stats/filteredQBs.csv")
RBs = pd.read_csv("stats/filteredRBs.csv")
WRs = pd.read_csv("stats/filteredWRs.csv")
TEs = pd.read_csv("stats/filteredTEs.csv")
DEFs = pd.read_csv("stats/filteredDEFs.csv")
Ks = pd.read_csv("stats/filteredKs.csv")

all_dfs = [QBs, RBs, WRs, TEs, Ks]
titles = ["QBs", "RBs", "WRs", "TEs", "Ks"]
# col_labels = ['R2 Score', 'MAE', 'MSE', 'RMSE']
# row_labels = ['MLR', 'Pipe MLR', 'RFR', 'Pipe RFR']

# building the hyperparameter grid

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'n_jobs': [-1]}

print('\n************************************************************\n')

for df, title in zip(all_dfs, titles):
    df.drop('name', inplace=True, axis=1)
    df = pd.get_dummies(df)
    print(title)

    #X = df.drop('Points', inplace=False, axis=1)
    #y = df['Points']
    #X = df.drop('Rank', inplace=False, axis=1)
    #y = df['Rank']
    X = df.drop('Avg', inplace=False, axis=1)
    y = df['Avg']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    grid_random_forest = RandomForestRegressor()
    rf_random = GridSearchCV(estimator= grid_random_forest, 
                                   param_grid= random_grid,
                                   cv= 5,
                                   verbose=0,
                                   n_jobs= -1)

    rf_random.fit(X_train, y_train)
    print(rf_random.best_params_)
    
    # evaluate best grid model
    best_random = rf_random.best_estimator_
    print('Grid Random Forest r2 Score:               ', round(best_random.score(X_test, y_test), 2))
    predicted_vals = best_random.predict(X_test)
    errors = abs(predicted_vals - y_test)
    print('Grid Random Forest Mean Absolute Error:    ', round(np.mean(errors), 2), 'points.')
    print('Grid Random Forest Mean Square Error:      ', round(mean_squared_error(y_test, predicted_vals), 2), 'points.')
    print('Grid Random Forest Root Mean Square Error: ', round(sqrt(mean_squared_error(y_test, predicted_vals)), 2), 'points.\n')


    # evaluate base model
    rfreg = RandomForestRegressor()
    rfreg.fit(X_train, y_train)
    print('Random Forest r2 Score:                    ', round(rfreg.score(X_test, y_test), 2))
    predicted_vals = rfreg.predict(X_test)
    errors = abs(predicted_vals - y_test)
    print('Random Forest Mean Absolute Error:         ', round(np.mean(errors), 2), 'points.')
    print('Random Forest Mean Square Error:           ', round(mean_squared_error(y_test, predicted_vals), 2), 'points.')
    print('Random Forest Root Mean Square Error:      ', round(sqrt(mean_squared_error(y_test, predicted_vals)), 2), 'points.\n')
    
    # rfr.append(round(r2_score(y_test, predicted_vals), 2))
    # rfr.append(round(np.mean(errors), 2))
    # rfr.append(round(mean_squared_error(y_test, predicted_vals), 2))
    # rfr.append(round(sqrt(mean_squared_error(y_test, predicted_vals)), 2))

    # pipe = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=10000, n_jobs=-1))
    # pipe.fit(X_train, y_train)
    # print('RFR Pipe R2 score:                    ', round(pipe.score(X_test, y_test), 2))
    # predicted_vals = pipe.predict(X_test)
    # errors = abs(predicted_vals - y_test)
    # print('RFR Pipe Mean Absolute Error:         ', round(np.mean(errors), 2), 'points.')
    # print('RFR Pipe Mean Square Error:           ', round(mean_squared_error(y_test, predicted_vals), 2), 'points.')
    # print('RFR Pipe RootMean Square Error:       ', round(sqrt(mean_squared_error(y_test, predicted_vals)), 2), 'points.\n')
    
    # pipe_rfr.append(round(r2_score(y_test, predicted_vals), 2))
    # pipe_rfr.append(round(np.mean(errors), 2))
    # pipe_rfr.append(round(mean_squared_error(y_test, predicted_vals), 2))
    # pipe_rfr.append(round(sqrt(mean_squared_error(y_test, predicted_vals)), 2))

    # fig, ax = plt.subplots()
    # ax.set_axis_off()
    # table = ax.table(cell_text, 
    #                  colLabels=col_labels,
    #                  rowLabels=row_labels,
    #                  loc='upper left',
    #                  rowColours = ["palegreen"] * 4,
    #                  colColours =["palegreen"] * 4)
    # ax.set_title(title)
    # plt.show()
    # plt.close()

    print('\n************************************************************\n')