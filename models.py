import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from numpy.core.fromnumeric import mean
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from math import sqrt
from keras import models, layers
from keras.wrappers.scikit_learn import KerasRegressor

def create_mlp(num_layers):
    model = models.Sequential()

    # input layer
    model.add(layers.Dense(16, activation='relu'))

    # hidden layers
    for _ in range(num_layers):
        model.add(layers.Dense(64, activation='relu'))

    model.add(layers.Dense(1, activation='linear'))
    model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_absolute_error"])
    return model

def main():
    QBs = pd.read_csv("stats/filteredQBs.csv")
    RBs = pd.read_csv("stats/filteredRBs.csv")
    WRs = pd.read_csv("stats/filteredWRs.csv")
    TEs = pd.read_csv("stats/filteredTEs.csv")
    DEFs = pd.read_csv("stats/filteredDEFs.csv")
    Ks = pd.read_csv("stats/filteredKs.csv")

    all_dfs = [QBs, RBs, WRs, TEs, Ks]
    titles = ["QBs", "RBs", "WRs", "TEs", "Ks"]
    col_labels = ['R2 Score', 'MAE', 'MSE', 'RMSE']
    row_labels = ['MLR', 'Pipe MLR', 'RFR', 'Pipe RFR', 'Neural Net']

    rf_regressors = {
        'QBs': RandomForestRegressor(bootstrap=True, max_depth=30, max_features='auto', min_samples_leaf=2, min_samples_split=2, n_estimators=400, n_jobs=-1),
        'RBs': RandomForestRegressor(bootstrap=True, max_depth=70, max_features='sqrt', min_samples_leaf=2, min_samples_split=2, n_estimators=400, n_jobs=-1),
        'WRs': RandomForestRegressor(bootstrap=True, max_depth=110, max_features='sqrt', min_samples_leaf=1, min_samples_split=10, n_estimators=200, n_jobs=-1),
        'TEs': RandomForestRegressor(bootstrap=True, max_depth=10, max_features='auto', min_samples_leaf=1, min_samples_split=5, n_estimators=400, n_jobs=-1),
        'Ks':  RandomForestRegressor(bootstrap=False, max_depth=110, max_features='sqrt', min_samples_leaf=1, min_samples_split=2, n_estimators=200, n_jobs=-1),
    }

    print('\n************************************************************\n')

    for df, title in zip(all_dfs, titles):
        df.drop('name', inplace=True, axis=1)
        df = pd.get_dummies(df)

        mlr = []
        pipe_mlr = []
        rfr = []
        pipe_rfr = []
        neural_net = []
        cell_text = [mlr, pipe_mlr, rfr, pipe_rfr, neural_net]

        #X = df.drop('Points', inplace=False, axis=1)
        #y = df['Points']
        #X = df.drop('Rank', inplace=False, axis=1)
        #y = df['Rank']
        X = df.drop('Avg', inplace=False, axis=1)
        y = df['Avg']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        regr = LinearRegression()
        regr.fit(X_train, y_train)

        predicted_vals = regr.predict(X_test)
        errors = abs(predicted_vals - y_test)
        print(title)
        print('MLR R2 score:                         ', round(r2_score(y_test, predicted_vals), 2))
        print('MLR Mean Absolute Error:              ', round(np.mean(errors), 2), 'points.')
        print('MLR Mean Square Error:                ', round(mean_squared_error(y_test, predicted_vals), 2), 'points.')
        print('MLR Root Mean Square Error:           ', round(sqrt(mean_squared_error(y_test, predicted_vals)), 2), 'points.\n')

        mlr.append(round(r2_score(y_test, predicted_vals), 2))
        mlr.append(round(np.mean(errors), 2))
        mlr.append(round(mean_squared_error(y_test, predicted_vals), 2))
        mlr.append(round(sqrt(mean_squared_error(y_test, predicted_vals)), 2))

        pipe = make_pipeline(StandardScaler(), PCA(), LinearRegression())
        pipe.fit(X_train, y_train)
        print('MLR Pipe R2 score:                    ', round(pipe.score(X_test, y_test), 2))
        predicted_vals = pipe.predict(X_test)
        errors = abs(predicted_vals - y_test)
        print('MLR Pipe Mean Absolute Error:         ', round(np.mean(errors), 2), 'points.')
        print('MLR Pipe Mean Square Error:           ', round(mean_squared_error(y_test, predicted_vals), 2), 'points.')
        print('MLR Pipe RootMean Square Error:       ', round(sqrt(mean_squared_error(y_test, predicted_vals)), 2), 'points.\n')

        pipe_mlr.append(round(r2_score(y_test, predicted_vals), 2))
        pipe_mlr.append(round(np.mean(errors), 2))
        pipe_mlr.append(round(mean_squared_error(y_test, predicted_vals), 2))
        pipe_mlr.append(round(sqrt(mean_squared_error(y_test, predicted_vals)), 2))

        rfreg = rf_regressors[title]
        rfreg.fit(X_train, y_train)
        print('Random Forest r2 Score:               ', round(rfreg.score(X_test, y_test), 2))
        predicted_vals = rfreg.predict(X_test)
        errors = abs(predicted_vals - y_test)
        print('Random Forest Mean Absolute Error:    ', round(np.mean(errors), 2), 'points.')
        print('Random Forest Mean Square Error:      ', round(mean_squared_error(y_test, predicted_vals), 2), 'points.')
        print('Random Forest Root Mean Square Error: ', round(sqrt(mean_squared_error(y_test, predicted_vals)), 2), 'points.\n')
        
        rfr.append(round(r2_score(y_test, predicted_vals), 2))
        rfr.append(round(np.mean(errors), 2))
        rfr.append(round(mean_squared_error(y_test, predicted_vals), 2))
        rfr.append(round(sqrt(mean_squared_error(y_test, predicted_vals)), 2))

        pipe = make_pipeline(StandardScaler(), rf_regressors[title])
        pipe.fit(X_train, y_train)
        print('RFR Pipe R2 score:                    ', round(pipe.score(X_test, y_test), 2))
        predicted_vals = pipe.predict(X_test)
        errors = abs(predicted_vals - y_test)
        print('RFR Pipe Mean Absolute Error:         ', round(np.mean(errors), 2), 'points.')
        print('RFR Pipe Mean Square Error:           ', round(mean_squared_error(y_test, predicted_vals), 2), 'points.')
        print('RFR Pipe RootMean Square Error:       ', round(sqrt(mean_squared_error(y_test, predicted_vals)), 2), 'points.\n')
        
        pipe_rfr.append(round(r2_score(y_test, predicted_vals), 2))
        pipe_rfr.append(round(np.mean(errors), 2))
        pipe_rfr.append(round(mean_squared_error(y_test, predicted_vals), 2))
        pipe_rfr.append(round(sqrt(mean_squared_error(y_test, predicted_vals)), 2))

        # Neural Network
        model = KerasRegressor(build_fn=create_mlp, num_layers=3, batch_size=5, epochs=200, verbose=0)
        model.fit(X_train, y_train)
        predicted_vals = model.predict(X_test)
        errors = abs(predicted_vals - y_test)
        print('Neural Net R2 score:                    ', round(r2_score(y_test, predicted_vals), 2))
        print('Neural Net Mean Absolute Error:         ', round(np.mean(errors), 2), 'points.')
        print('Neural Net Mean Square Error:           ', round(mean_squared_error(y_test, predicted_vals), 2), 'points.')
        print('Neural Net RootMean Square Error:       ', round(sqrt(mean_squared_error(y_test, predicted_vals)), 2), 'points.\n')

        neural_net.append(round(r2_score(y_test, predicted_vals), 2))
        neural_net.append(round(np.mean(errors), 2))
        neural_net.append(round(mean_squared_error(y_test, predicted_vals), 2))
        neural_net.append(round(sqrt(mean_squared_error(y_test, predicted_vals)), 2))

        fig, ax = plt.subplots()
        ax.set_axis_off()
        table = ax.table(cell_text, 
                        colLabels=col_labels,
                        rowLabels=row_labels,
                        loc='upper left',
                        rowColours = ["palegreen"] * 5,
                        colColours =["palegreen"] * 4)
        ax.set_title(title)
        plt.show()
        plt.close()

        print('\n************************************************************\n')

if __name__ == "__main__":
    main()