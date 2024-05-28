import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, KFold
from scipy.stats import randint, uniform
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from skopt import BayesSearchCV



# Load data
def load_data(file_path, exclude_vars, y_var):
    data = pd.read_csv(file_path)
    X = data.drop(columns=exclude_vars)
    y = data[y_var].dropna()
    X = X.loc[y.index]
    X = pd.get_dummies(X)  # Convert categorical columns to dummy variables
    return X, y

def remove_outliers(X, y):
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    condition = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
    return X[condition], y[condition]

def random_hyperparam_opt_XGB(X_train, y_train):
    # Define the parameter distribution
    param_dist = {
        'n_estimators': randint(100, 400),  # Adjusted range for more exploration
        'learning_rate': uniform(0.01, 0.2),  # Adjusted range to focus on more likely beneficial values
        'subsample': uniform(0.5, 0.5),  # Corrected to ensure valid values
        'max_depth': randint(3, 10),  # Expanded range for depth
        'colsample_bytree': uniform(0.3, 0.7),  # Adjusted for better exploration of feature sampling
        'min_child_weight': randint(1, 10),
        'gamma': uniform(0, 0.5),  # Added gamma for better control of tree growth
        'reg_lambda': uniform(0.5, 1.5)  # Added L2 regularization
    }
    
    # Initialize the XGBRegressor
    xgb = XGBRegressor(random_state=42)

    # Initialize the RandomizedSearchCV object with increased CV folds and iterations
    random_search = RandomizedSearchCV(
        xgb, 
        param_distributions=param_dist, 
        n_iter=100,  # Increased iterations
        scoring='neg_mean_squared_error', 
        n_jobs=-1,  # Use all available cores
        cv=5,  # Increased number of folds
        random_state=42
    )

    # Fit the model with early stopping
    random_search.fit(X_train, y_train, verbose=False)
    
    # Print best parameters and best score
    print("Best parameters:", random_search.best_params_)
    print("Best train score:", -random_search.best_score_)


    return random_search.best_estimator_

def random_hyperparam_opt_RF(X_train, y_train):
    # Define the parameter distribution
    param_dist = {
        'n_estimators': randint(100, 1000),  # Number of trees in the forest
        'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider at every split
        'max_depth': randint(3, 20),  # Maximum number of levels in tree
        'min_samples_split': randint(2, 20),  # Minimum number of samples required to split a node
        'min_samples_leaf': randint(1, 10)  # Minimum number of samples required at each leaf node
    }
    
    # Initialize the RandomForestRegressor
    rf = RandomForestRegressor(random_state=42)

    # Initialize the RandomizedSearchCV object with increased cross-validation and iterations
    random_search = RandomizedSearchCV(
        rf, 
        param_distributions=param_dist, 
        n_iter=200,  # Increased number of iterations
        scoring='neg_mean_squared_error', 
        n_jobs=-1,  # Use all available cores
        cv=5,  # Increased number of folds for better stability
        random_state=42
    )

    # Fit the model
    random_search.fit(X_train, y_train)
    
    # Print best parameters and best score
    print("Best parameters:", random_search.best_params_)
    print("Best train score:", -random_search.best_score_)

    return random_search.best_estimator_

def bayesian_hyperparam_opt_RF(X_train, y_train):
    param_dist = {
        'n_estimators': (100, 1000),  # Continuous range for Bayesian optimization
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': (3, 20),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 10)
    }
    
    rf = RandomForestRegressor(random_state=42)
    bayes_search = BayesSearchCV(rf, param_dist, n_iter=100, scoring='neg_mean_squared_error', n_jobs=-1, cv=5, random_state=42)
    
    bayes_search.fit(X_train, y_train)
    
    print("Best parameters:", bayes_search.best_params_)
    print("Best train score:", -bayes_search.best_score_)
    

    return bayes_search.best_estimator_

def grid_hyperparam_opt_RF(X_train, y_train):
    """
    Load data, tune hyperparameters using Grid Search with K-Fold cross-validation, 
    train the model using the best parameters, and evaluate it on the test data.

    Args:
    file_path (str): Path to the CSV file.
    exclude_vars (list): List of columns to exclude from the feature set.
    y_var (str): Name of the target variable.
    watershed_value (str, optional): Value in the 'Watershed' column to filter by. If None, load all data.

    Returns:
    float: Mean squared error of the model on the test data.
    """
    print("Optimizing hyperparameters using Grid Search with K-Fold cross-validation...")
    # Define hyperparameters grid
    param_grid = {
        'n_estimators': [250, 300, 350, 400, 450],
        'max_features': [1.0, 'sqrt'],
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
    }

    # Set up Grid Search with K-fold cross-validation
    cv = KFold(n_splits=10, random_state=42, shuffle=True)
    grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

    # Fit the grid search model
    grid_search.fit(X_train, y_train)

    # Get the best estimator and print best parameters and score
    best_model = grid_search.best_estimator_
    print("Best parameters:", grid_search.best_params_)
    print("Best score (MSE):", -grid_search.best_score_)

    # Evaluate the model on the test set using the best estimator
    return best_model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    score = model.score(X_test, y_test)
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {mse**0.5}')
    print(f'R^2 Score: {score}')
    return score, mse, predictions

def plot_predictions(y_test, predictions, mean_score, field_name):
    df = pd.DataFrame(y_test)
    #name the column y_field
    df['y_test'] = y_test
    df['predictions'] = predictions
    #plot the predictions vs y_test
    plt.scatter(df['y_test'], df['predictions'])
    plt.xlabel(f'Actual {field_name} (m)')
    plt.ylabel(f'Predicted {field_name} (m)')
    #set axis to be the same bounds
    plt.xlim(min(df['y_test']), max(df['y_test']))
    plt.ylim(min(df['y_test']), max(df['y_test']))
    #plot a perfect correlation line
    plt.plot([min(df['y_test']), max(df['y_test'])], [min(df['y_test']), max(df['y_test'])], color='red', linestyle='--')
    plt.title('Predicted vs Actual Erosion for each Catchment')

    # Ensure you have defined the appropriate y_field in your DataFrame
    y_min = min(df['y_test'])
    y_max = max(df['y_test'])
    x_min = min(df['predictions'])
    x_max = max(df['predictions'])

    # Adjust the position values to top left
    x_position = x_min + 0.05 * (x_max - x_min)  # More towards the left
    y_position = y_max - 0.05 * (y_max - y_min)  # Less from the top to ensure it is inside the plot

    plt.text(x_position, y_position, r"$r^2={:.3f}$".format(mean_score), fontsize=12, fontweight='bold',
            verticalalignment='top', horizontalalignment='left', backgroundcolor='white', color='black')

    plt.legend(['Catchment', 'Perfect Correlation'], loc='lower right')

    plt.show()

def plot_y_ytest_histogram(y, y_test):
    #make a data frame from y_test and predictions
    df = pd.DataFrame(y_test)
    df_y = pd.DataFrame(y)
    #name the column y_field
    df['y_test'] = y_test
    df_y['y'] = y = y.to_numpy()
    #plot histogram of y_test and y
    plt.hist(df_y['y'], bins=100, alpha=0.5, label='y')
    plt.hist(df['y_test'], bins=100, alpha=0.5, label='y_test', color='red')
    plt.title(f'Distribution of all target data and testing dataset')
    plt.xlabel(f'Normalized Erosion per Catchment $m^3/m^2$')
    plt.ylabel('Frequency')
    #add legend
    plt.legend(loc='upper left')
    plt.show()

def main():
    # Configuration
    csv_fn = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Hillslopes\Hillslope_Stats_Combined.csv"
    variables_to_exclude = ['ID', 'Erosion mean', "Category",	"OBJECTID",	"Watershed", "Deposition mean",	"Deposition stdev",	"Erosion stdev",
                        "Deposition mean Masked",	"Deposition stdev Masked",	"Erosion mean Masked",	"Erosion stdev Masked"]


if __name__ == '__main__':
    main()

