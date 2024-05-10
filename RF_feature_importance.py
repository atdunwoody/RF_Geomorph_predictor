import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
from RF_utils import load_data, train_model, evaluate_model

def evaluate_model_with_shap(model, X_train, X_test, plot_title="", plot = True):
    """
    Evaluate the model using SHAP values to understand feature importance.

    Args:
    model (model object): Trained machine learning model.
    X_train (DataFrame): Training dataset used to fit the model.
    X_test (DataFrame): Test dataset used to explain the model predictions.
    plot_title (str): Title for the SHAP summary plot.

    Returns:
    None: This function prints the SHAP summary plot with a custom title.
    """
    # Create the SHAP Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Calculate the mean absolute SHAP values for each feature
    shap_summaries = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame(list(zip(X_train.columns, shap_summaries)),
                                      columns=['Feature', 'SHAP Importance'])
    feature_importance = feature_importance.sort_values('SHAP Importance', ascending=False).reset_index(drop=True)
    
    # Print the SHAP values summary plot with a custom title
    if plot:
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)  # Disable automatic showing
        plt.title(plot_title)  # Set the custom title
        #set x axis label to none
        plt.xlabel('SHAP Feature Importance Ranking')

        plt.show()  # Show the plot manually to apply the title

        # Visualize the first prediction's explanation
        shap.initjs()  # Initialize JavaScript visualization in Jupyter Notebook/Colab if applicable
    return feature_importance, shap.force_plot(explainer.expected_value, shap_values[0, :], X_test.iloc[0, :])


def evaluate_feature_with_shap(model, X_train, X_test, feature_to_plot, plot_title="", plot=True):
    """
    Evaluate the model using SHAP values to understand feature importance and plot the dependence
    plot for a specified feature. Additionally, return a DataFrame with features ranked by their
    average absolute SHAP values.

    Args:
    model (model object): Trained machine learning model.
    X_train (DataFrame): Training dataset used to fit the model.
    X_test (DataFrame): Test dataset used to explain the model predictions.
    feature_to_plot (str): Feature for which to create a dependence plot.
    plot_title (str): Title for the SHAP summary plot.

    Returns:
    DataFrame: A DataFrame with features ranked by their average absolute SHAP values.
    """
    # Create the SHAP Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Calculate the mean absolute SHAP values for each feature
    shap_summaries = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame(list(zip(X_train.columns, shap_summaries)),
                                      columns=['Feature', 'SHAP Importance'])
    feature_importance = feature_importance.sort_values('SHAP Importance', ascending=False).reset_index(drop=True)

    # Plot the dependence plot for the specified feature
    if plot:
        shap.dependence_plot(feature_to_plot, shap_values, X_train, show=False)
        plt.title(f"Dependence Plot for {feature_to_plot}")
        plt.show()

        # Optional: Visualize the first prediction's explanation with a force plot
        shap.initjs()


    return feature_importance, shap.force_plot(explainer.expected_value, shap_values[0, :], X_test.iloc[0, :])


def tune_and_evaluate_model_with_shap(file_path, exclude_vars, y_var, watershed_value=None):
    # Assume data loading and model training functions are defined elsewhere in the script
    X, y = load_data(file_path, exclude_vars, y_var) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)  # Assume train_model function exists and trains the model

    # Evaluate the model with SHAP
    evaluate_model_with_shap(model, X_train, X_test)

    # Continue with any other evaluation metrics or functions
    mse = evaluate_model(model, X_test, y_test)
    print(f'Mean Squared Error: {mse}')


def main():
    pass
if __name__ == "__main__":
    main()
