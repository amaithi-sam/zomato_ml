import numpy as np


def hyper_parameter_tuning(X_train, y_train, random_seed):
    # define random parameters grid
    n_estimators = [5,12,21,51,101] # number of trees in the random forest
    max_features = ['sqrt'] # number of features in consideration at every split
    max_depth = [int(x) for x in np.linspace(10, 120, num = 12)] # maximum number of levels allowed in each decision tree
    min_samples_split = [2, 6, 10] # minimum sample number to split a node
    min_samples_leaf = [1, 3, 4] # minimum sample number that can be stored in a leaf node
    bootstrap = [True, False] # method used to sample data points

    random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap
                  }
    
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestRegressor
    
    regressor = RandomForestRegressor()
    model_tuning = RandomizedSearchCV(estimator = regressor, param_distributions = random_grid,
                   n_iter = 100, cv = 5, verbose=2, random_state=random_seed, n_jobs = -1)
    model_tuning.fit(X_train, y_train)

    print ('\nRandom grid: ', random_grid, '\n')
    # print the best parameters
    print ('Best Parameters: ', model_tuning.best_params_, ' \n')

    best_params = model_tuning.best_params_
    print("\nFinished HYPERPARAMETER TUNING\n")
    
    
    return best_params