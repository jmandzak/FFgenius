import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from math import sqrt
from keras import models, layers
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
import matplotlib.pyplot as plt

def create_mlp(hidden_layers=3, activation='relu', num_neurons=16, optimizer='Adam', learning_rate=1e-2):
    model = models.Sequential()

    # input layer
    model.add(layers.Dense(16, activation=activation))

    # hidden layers
    for _ in range(hidden_layers):
        model.add(layers.Dense(num_neurons, activation=activation))
        model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1, activation='linear'))
    model.compile(loss= "mean_squared_error" , optimizer=optimizer, metrics=["mean_absolute_error"])
    #model.compile(loss= "mean_squared_error" , optimizer=SGD(learning_rate=learning_rate), metrics=["mean_absolute_error"])
    return model

def main():
    #QBs = pd.read_csv("stats/filteredQBs.csv")
    RBs = pd.read_csv("stats/filteredRBs.csv")
    WRs = pd.read_csv("stats/filteredWRs.csv")
    TEs = pd.read_csv("stats/filteredTEs.csv")
    #DEFs = pd.read_csv("stats/filteredDEFs.csv")
    Ks = pd.read_csv("stats/filteredKs.csv")

    all_dfs = [RBs, WRs, TEs, Ks]
    titles = ["RBs", "WRs", "TEs", "Ks"]
    param_grid = {
        'batch_size': [5, 10, 20],
        'epochs': [100, 200, 300],
        #'learning_rate': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        'optimizer': ['SGD', 'RMSprop', 'Adam', 'Adamax', 'Adagrad', 'Adadelta', 'Nadam', 'Frtl'],
        'hidden_layers': [1, 2, 3, 4, 5],
        'activation': ['relu', 'sigmoid', 'softmax', 'softsign', 'softplus', 'tanh', 'selu', 'elu', 'exponential'],
        'num_neurons': [8, 16, 32, 64, 128, 256]
    }
    # param_grid = {
    #     'batch_size': [10],
    #     'epochs': [100, 300],
    #     'optimizer': ['SGD'],
    #     'hidden_layers': [4, 5],
    #     'activation': ['softsign'],
    #     'num_neurons': [128]
    # }

    f = open('nn_tuning_results.txt', 'a')

    for df, title in zip(all_dfs, titles):
        df.drop('name', inplace=True, axis=1)
        df = pd.get_dummies(df)

        X = df.drop('Avg', inplace=False, axis=1)
        X = X.values.astype(np.float64)
        y = df['Avg']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = KerasRegressor(build_fn=create_mlp, verbose=0)
        #grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=-1)
        grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=500, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)
        grid_result = grid.fit(X_train_scaled, y_train)

        # summarize results
        f.write(f'Position: {title}\n')
        f.write(f'Best: {grid_result.best_score_} using {grid_result.best_params_}\n')

        best_model = grid_result.best_estimator_
        predicted_vals = best_model.predict(X_test_scaled)
        errors = abs(predicted_vals - y_test)
        f.write(f'Neural Net R2 score:                   {round(r2_score(y_test, predicted_vals), 2)}\n')
        f.write(f'Neural Net Mean Absolute Error:        {round(np.mean(errors), 2)} points.\n')
        f.write(f'Neural Net Mean Square Error:          {round(mean_squared_error(y_test, predicted_vals), 2)} points.\n')
        f.write(f'Neural Net RootMean Square Error:      {round(sqrt(mean_squared_error(y_test, predicted_vals)), 2)} points.\n\n')

        ax = plt.gca()
        ax.scatter((list(range(len(predicted_vals)))), list(predicted_vals), color="b")
        ax.scatter((list(range(len(predicted_vals)))), list(y_test), color="r")
        plt.yticks(np.arange(min((list(predicted_vals) + list(y_test))), max((list(predicted_vals) + list(y_test)))+1, 1.0))
        plt.xticks(np.arange(min(list(range(len(predicted_vals)))), max(list(range(len(predicted_vals))))+1, 1.0))
        plt.grid()
        plt.show()
        plt.close()

        return


if __name__ == '__main__':
    main()