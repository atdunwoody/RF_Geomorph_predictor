import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
import shap

import RF_lib.RF_utils as rfut


def quick_shap_eval(model, X_train, X_test, plot_title="", plot = True):
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

def calculate_shap_importance(model, X_train, X_test):
    RANKED_shap, _ = quick_shap_eval(model, X_train, X_test, plot=False)  # Assuming evaluate_model_with_shap is defined
    return RANKED_shap


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

def quick_RFE(model, X, y, n_features_to_select=10):
    """
    Perform Recursive Feature Elimination (RFE) using a Random Forest classifier.
    
    Args:
    X (DataFrame): The feature set.
    y (Series): The target variable.
    n_features_to_select (int): The number of features to select.
    
    Returns:
    DataFrame: A DataFrame with feature rankings (1 is most important).
    """

    # Create the RFE model and select attributes
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
    rfe = rfe.fit(X, y)
    
    # Summarize the selection of the attributes
    feature_ranking = pd.DataFrame({'Feature': X.columns, 
                                    'Ranking': rfe.RFE_})
    
    return feature_ranking.sort_values('Ranking')

def correlation_matrix_table(features_df):
    """
    Calculate and display a correlation matrix with Pearson's R^2 for all features.

    Parameters:
    - features_df (pd.DataFrame): DataFrame containing all the features.

    Returns:
    - Displays a heatmap of the correlation matrix.
    """
    # Calculate the Pearson correlation matrix
    corr_matrix = features_df.corr()
    
    # Create a heatmap to visualize the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'Correlation coefficient'})
    plt.title("Pearson's R² Correlation Matrix")
    plt.show()

def plot_permutation_importance(model, X_test, y_test, n_repeats=30):
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats = n_repeats, random_state=42)

    # Prepare the data for plotting
    importances = perm_importance.importances_mean[perm_importance.importances_mean.argsort()]
    std = perm_importance.importances_std[perm_importance.importances_mean.argsort()]
    features = X_test.columns[perm_importance.importances_mean.argsort()]

    # Reverse the order so the most important feature is at the top
    importance_data = pd.DataFrame({
        'Features': features[::-1],
        'Importances': importances[::-1],
        'Std': std[::-1]
    })

    # Plot using Seaborn
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importances', y='Features', data=importance_data, xerr=importance_data['Std'], color = 'skyblue', capsize=3)

    plt.xlabel('Permutation Importance')
    plt.title('Permutation Feature Importance')
    plt.show()

def plot_Gini_feature_importances(model, feature_names):
    #Gini Impurity Feature Importance Ranking
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.title('Gini Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

def partial_dependence_plots(model, X, features):
    # Create the partial dependence plot display
    disp = PartialDependenceDisplay.from_estimator(model, X, features)
    # Show the plot
    disp.figure_.suptitle("Partial Dependence Plots")
    # make the figure large
    disp.figure_.set_size_inches(20, 16)
    plt.show()

def split_data(X, y, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return kf.split(X)

def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    r2, mse, predictions = rfut.evaluate_model(model, X_test, y_test)  # Assuming rfut is defined somewhere
    return {'MSE': mse, 'score': model.score(X_test, y_test)}


def calculate_rfe_importance(model, X_train, y_train, n_folds = 3):
    rfe = RFECV(estimator=model, step=1, cv=KFold(n_folds), scoring='neg_mean_squared_error')
    rfe.fit(X_train, y_train)
    return rfe.ranking_

def print_results(results):
    mean_MSE = np.mean([r['MSE'] for r in results])
    mean_score = np.mean([r['score'] for r in results])
    print("\n Model Evaluation Results:")
    print(f"Average Score: {mean_score}")
    print(f"Average MSE: {mean_MSE}")

def prepare_shap_data(shap_dict):
    shap_mean_std = {key: {'mean': np.mean(vals), 'std': np.std(vals)} for key, vals in shap_dict.items()}
    shap_df = pd.DataFrame(list(shap_dict.items()), columns=['Feature', 'SHAP Importances'])
    shap_df['Mean SHAP'] = shap_df['SHAP Importances'].apply(np.mean)
    shap_df['STD Errors'] = shap_df['SHAP Importances'].apply(np.std)
    shap_df = shap_df.sort_values(by='Mean SHAP', ascending=True).reset_index(drop=True)
    return shap_df

def assign_colors(df, color_mapping=None, default_color='skyblue'):
    """
    Assigns colors to features based on a provided mapping or a default color.

    Args:
    df (DataFrame): DataFrame containing the 'Feature' column.
    color_mapping (dict, optional): Dictionary mapping features to colors.
                                   If a feature is not in the dictionary, the default color is used.
    default_color (str): Default color to use for features not in the color_mapping dictionary.

    Returns:
    DataFrame: Updated DataFrame with a 'Color' column.
    """
    if color_mapping is None:
        color_mapping = {}
    df['Color'] = df['Feature'].apply(lambda feature: color_mapping.get(feature, default_color))
    return df

def plot_shap_features(df, top_n=10):
    top_features = df.head(top_n).sort_values(by='Mean SHAP', ascending=False).reset_index(drop=True)
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x='Mean SHAP', y='Feature', data=top_features, palette=top_features['Color'])
    for index, row in top_features.iterrows():
        bar_plot.errorbar(row['Mean SHAP'], index, xerr=row['STD Errors'], fmt='o', color='black', capsize=3)
    plt.xlabel('Mean SHAP Value')
    plt.title('SHAP Feature Rankings across k-folds:')
    plt.show()

def plot_rfe_features(rfe_results, X, color_mapping=None, default_color='skyblue', top_n=10):
    """
    Plots the Recursive Feature Elimination (RFE) results with colors specified for certain features.

    Args:
    rfe_results (array): Array of RFE rankings across different folds.
    X (DataFrame): DataFrame with feature columns used in RFE.
    color_mapping (dict, optional): Dictionary mapping specific features to colors.
    default_color (str): Color used for features not specified in color_mapping.
    top_n (int): Number of top features to display in the plot.
    """
    mean_rankings = np.mean(rfe_results, axis=0)
    std_rankings = np.std(rfe_results, axis=0)
    rfe_df = pd.DataFrame({'Feature': X.columns, 'Mean RFE': mean_rankings, 'STD RFE': std_rankings})
    rfe_df = assign_colors(rfe_df, color_mapping, default_color)
    rfe_df = rfe_df.sort_values(by='Mean RFE', ascending=True).reset_index(drop=True)

    top_features = rfe_df.head(top_n).sort_values(by='Mean RFE', ascending=False).reset_index(drop=True)
    
    plt.figure(figsize=(12, 8))
    rfe_bar_plot = sns.barplot(x='Mean RFE', y='Feature', data=top_features, palette=top_features['Color'])
    for index, row in top_features.iterrows():
        rfe_bar_plot.errorbar(row['Mean RFE'], index, xerr=row['STD RFE'], fmt='o', color='black', capsize=3)
    plt.xlabel('Average Ranking Across Folds')
    plt.ylabel('Features')
    plt.title('Feature Rankings by Recursive Feature Elimination with Error Bars')
    plt.show()

def robust_feature_ranking(model, X, y, perform_rfe = True, plot = True, color_mapping = None):
    performance_results = []
    shap_dict = {}
    rfe_dict = []
    for train_index, test_index in split_data(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        result = train_and_evaluate(model, X_train, X_test, y_train, y_test)
        performance_results.append(result)

        shap_importance = calculate_shap_importance(model, X_train, X_test)
        for key in shap_importance['Feature']:
            shap_value = shap_importance['SHAP Importance'][shap_importance['Feature'] == key].values[0]
            if key in shap_dict:
                shap_dict[key].append(shap_value)
            else:
                shap_dict[key] = [shap_value]
        
        if perform_rfe:
            rfe_ranking = calculate_rfe_importance(model, X_train, y_train)
            rfe_dict.append(rfe_ranking)

    print_results(performance_results)
    if plot:
        shap_df = prepare_shap_data(shap_dict)
        # Uncomment to assign custom colors to bar chart
        # custom_colors = {
        #     'feature1': 'green',
        #     'dummy': 'red'
        # }
        # shap_df = assign_colors(shap_df, color_mapping=custom_colors, default_color='skyblue')
        
        shap_df = assign_colors(shap_df, default_color='skyblue')

        plot_shap_features(shap_df, 10)  # Plot top 10 features

        if perform_rfe:
            #color_mapping = {'dummy': 'red'}
            color_mapping = None
            plot_rfe_features(rfe_dict, X, color_mapping, 'skyblue')

    return performance_results, shap_dict, rfe_dict




def main():
    pass
if __name__ == "__main__":
    main()
