import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import joblib

# Load data
def load_data(file_path, exclude_vars, y_var):
    data = pd.read_csv(file_path)
    X = data.drop(columns=exclude_vars)
    y = data[y_var].dropna()
    X = X.loc[y.index]
    X = pd.get_dummies(X)  # Convert categorical columns to dummy variables
    return X, y


# Train the model
def train_model(X_train, y_train):
    #Best parameters: {'max_depth': 18, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 350}
    model = RandomForestRegressor(n_estimators=350, max_features='sqrt', max_depth=18, min_samples_split=2, random_state=42)
    model.fit(X_train, y_train)
    return model

def tune_model(X_train, y_train):
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

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')
    return mse, predictions

# Display feature importances
def plot_feature_importances(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

# Plot Partial Dependence
def partial_dependence_plots(model, X, features):
    # Create the partial dependence plot display
    disp = PartialDependenceDisplay.from_estimator(model, X, features)
    # Show the plot
    disp.figure_.suptitle("Partial Dependence Plots")
    # make the figure large
    disp.figure_.set_size_inches(20, 16)
    plt.show()

def main():
    # Configuration
    csv_fn = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Hillslopes\Hillslope_Stats_Combined.csv"
    variables_to_exclude = ['ID', 'Erosion mean', "Category",	"OBJECTID",	"Watershed", "Deposition mean",	"Deposition stdev",	"Erosion stdev",
                        "Deposition mean Masked",	"Deposition stdev Masked",	"Erosion mean Masked",	"Erosion stdev Masked"]

    y_field = 'Erosion mean Masked'
    watershed_value = 'MM'
    # Load and prepare data
    X, y = load_data(csv_fn, variables_to_exclude, y_field)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    # Train the Random Forest model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Feature importances
    plot_feature_importances(model, X.columns)

    # Partial Dependence Plots (select features you are interested in)
    partial_dependence_plots(model, X_train, [5,6])  # Adjust index to select features

    # Save the model
    joblib.dump(model, 'random_forest_model.pkl')

if __name__ == '__main__':
    main()

