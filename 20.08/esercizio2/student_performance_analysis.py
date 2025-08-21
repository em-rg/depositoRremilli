#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Student Performance Analysis

This script performs preprocessing and exploratory data analysis on the student-por.csv
dataset, focusing on data cleaning, normalization of numerical variables, 
encoding of categorical variables, and exploring distributions and correlations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_data(file_path):
    """
    Load the dataset from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.
    """
    return pd.read_csv(file_path)


def explore_data(df):
    """
    Perform exploratory data analysis on the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to explore.

    Returns
    -------
    None
        Displays information about the dataset.
    """
    print("Dataset Shape:", df.shape)
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    # Check data types and convert if necessary
    print("\nData Types:")
    print(df.dtypes)
    
    # Display unique values for categorical variables
    print("\nUnique Values for Categorical Variables:")
    for col in df.select_dtypes(include=['object']).columns:
        print(f"{col}: {df[col].unique()}")


def visualize_data(df):
    """
    Create visualizations to better understand the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to visualize.

    Returns
    -------
    None
        Displays visualizations.
    """
    # Set the style
    sns.set(style="whitegrid")
    plt.figure(figsize=(20, 15))
    
    # Distribution of final grade
    plt.subplot(3, 3, 1)
    sns.histplot(df['G3'], kde=True, color='blue')
    plt.title('Distribution of Final Grade (G3)')
    plt.xlabel('Final Grade')
    plt.ylabel('Count')
    
    # Study time vs final grade
    plt.subplot(3, 3, 2)
    sns.boxplot(x='studytime', y='G3', data=df, palette='viridis')
    plt.title('Study Time vs Final Grade')
    plt.xlabel('Study Time (1-4)')
    plt.ylabel('Final Grade')
    
    # Absences vs final grade
    plt.subplot(3, 3, 3)
    sns.scatterplot(x='absences', y='G3', data=df, hue='sex', palette='Set1')
    plt.title('Absences vs Final Grade')
    plt.xlabel('Number of Absences')
    plt.ylabel('Final Grade')
    
    # Grade comparison by gender
    plt.subplot(3, 3, 4)
    sns.boxplot(x='sex', y='G3', data=df, palette='Set2')
    plt.title('Final Grade by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Final Grade')
    
    # Grade comparison by school
    plt.subplot(3, 3, 5)
    sns.boxplot(x='school', y='G3', data=df, palette='Set2')
    plt.title('Final Grade by School')
    plt.xlabel('School')
    plt.ylabel('Final Grade')
    
    # Grade comparison by internet access
    plt.subplot(3, 3, 6)
    sns.boxplot(x='internet', y='G3', data=df, palette='Set2')
    plt.title('Final Grade by Internet Access')
    plt.xlabel('Internet Access')
    plt.ylabel('Final Grade')
    
    # Grade progression (G1 -> G2 -> G3)
    plt.subplot(3, 3, 7)
    g1 = df['G1'].astype(float).mean()
    g2 = df['G2'].astype(float).mean()
    g3 = df['G3'].mean()
    sns.barplot(x=['G1', 'G2', 'G3'], y=[g1, g2, g3], palette='Blues_d')
    plt.title('Grade Progression (Mean Values)')
    plt.xlabel('Period')
    plt.ylabel('Mean Grade')
    
    # Weekend alcohol consumption vs final grade
    plt.subplot(3, 3, 8)
    sns.boxplot(x='Walc', y='G3', data=df, palette='YlOrRd')
    plt.title('Weekend Alcohol Consumption vs Final Grade')
    plt.xlabel('Weekend Alcohol Consumption (1-5)')
    plt.ylabel('Final Grade')
    
    # Relationship between parents' education and final grade
    plt.subplot(3, 3, 9)
    df['ParentEduc'] = (df['Medu'] + df['Fedu'])/2  # Average of mother's and father's education
    sns.scatterplot(x='ParentEduc', y='G3', data=df, hue='sex', palette='Set1')
    plt.title("Parents' Education vs Final Grade")
    plt.xlabel("Parents' Education Level (avg)")
    plt.ylabel('Final Grade')
    
    plt.tight_layout()
    plt.show()
    
    # Create correlation heatmap
    plt.figure(figsize=(14, 12))
    # Convert relevant columns to numeric to include in correlation
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap
    sns.heatmap(correlation, mask=mask, cmap=cmap, annot=True, 
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Correlation Heatmap of Numerical Variables')
    plt.show()
    
    # Feature importance based on correlation with G3
    plt.figure(figsize=(12, 8))
    corr_with_g3 = correlation['G3'].sort_values(ascending=False)
    sns.barplot(x=corr_with_g3.values, y=corr_with_g3.index, palette='viridis')
    plt.title('Features Correlation with Final Grade (G3)')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Features')
    plt.show()


def preprocess_data(df):
    """
    Preprocess the dataset by handling missing values, encoding categorical variables,
    and scaling numerical variables.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to preprocess.

    Returns
    -------
    pd.DataFrame
        The preprocessed dataset.
    dict
        Dictionary with encoded categorical columns and scaler for numerical columns.
    """
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Convert numeric columns stored as strings to float
    for col in ['G1', 'G2']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    
    # Identify categorical and numerical columns
    categorical_cols = df_copy.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\nCategorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    # Handle missing values
    print("\nChecking for missing values...")
    missing_values = df_copy.isnull().sum()
    if missing_values.sum() > 0:
        print("Handling missing values:")
        print(missing_values[missing_values > 0])
        
        # Impute missing values
        for col in numerical_cols:
            if df_copy[col].isnull().sum() > 0:
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
        
        for col in categorical_cols:
            if df_copy[col].isnull().sum() > 0:
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
    else:
        print("No missing values found.")
    
    # Encode categorical variables
    print("\nEncoding categorical variables...")
    encoders = {}
    
    for col in categorical_cols:
        # Create a label encoder for the column
        le = LabelEncoder()
        df_copy[f"{col}_encoded"] = le.fit_transform(df_copy[col])
        encoders[col] = le
        
        # Create dummy variables for categorical columns
        dummies = pd.get_dummies(df_copy[col], prefix=col, drop_first=True)
        df_copy = pd.concat([df_copy, dummies], axis=1)
    
    # Scale numerical variables
    print("\nScaling numerical variables...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_copy[numerical_cols])
    
    # Create a new DataFrame with scaled numerical data
    scaled_df = pd.DataFrame(scaled_data, columns=[f"{col}_scaled" for col in numerical_cols])
    df_copy = pd.concat([df_copy, scaled_df], axis=1)
    
    preprocessing_info = {
        'encoders': encoders,
        'scaler': scaler,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols
    }
    
    return df_copy, preprocessing_info


def identify_key_factors(df):
    """
    Identify key factors that influence student performance.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyze.

    Returns
    -------
    None
        Displays analysis of key factors.
    """
    # Convert G1 and G2 to numeric if they're not already
    df['G1'] = pd.to_numeric(df['G1'], errors='coerce')
    df['G2'] = pd.to_numeric(df['G2'], errors='coerce')
    
    # Correlation with final grade
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()['G3'].sort_values(ascending=False)
    
    print("\nFeatures most correlated with final grade (G3):")
    print(correlations)
    
    # Analyze categorical variables' impact on grades
    categorical_vars = ['school', 'sex', 'address', 'famsize', 'Pstatus', 
                        'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 
                        'famsup', 'paid', 'activities', 'nursery', 'higher', 
                        'internet', 'romantic']
    
    print("\nMean G3 grade by categorical variables:")
    for var in categorical_vars:
        print(f"\n{var}:")
        mean_grades = df.groupby(var)['G3'].mean().sort_values(ascending=False)
        print(mean_grades)
        
        # Create a visualization for each categorical variable
        plt.figure(figsize=(10, 6))
        sns.barplot(x=mean_grades.index, y=mean_grades.values, palette='viridis')
        plt.title(f'Mean Final Grade (G3) by {var}')
        plt.xlabel(var)
        plt.ylabel('Mean Final Grade')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    # Feature importance based on information gain
    print("\nFeature importance for final grade prediction:")
    print("(Based on correlation with G3)")
    
    # Bivariate analysis - selected pairs of variables
    important_pairs = [
        ('studytime', 'G3'),
        ('absences', 'G3'),
        ('failures', 'G3'),
        ('Medu', 'G3'),
        ('Fedu', 'G3'),
        ('age', 'G3'),
        ('Walc', 'G3'),
        ('Dalc', 'G3'),
        ('G1', 'G3'),
        ('G2', 'G3')
    ]
    
    plt.figure(figsize=(15, 10))
    for i, (x, y) in enumerate(important_pairs):
        plt.subplot(2, 5, i+1)
        sns.regplot(x=x, y=y, data=df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        plt.title(f'{x} vs {y}')
    
    plt.tight_layout()
    plt.show()
    
    # Distribution of important numerical variables
    numerical_vars = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 
                     'failures', 'famrel', 'freetime', 'goout', 'Dalc', 
                     'Walc', 'health', 'absences', 'G1', 'G2', 'G3']
    
    plt.figure(figsize=(15, 10))
    for i, var in enumerate(numerical_vars[:9]):  # Show first 9 variables
        plt.subplot(3, 3, i+1)
        sns.histplot(df[var], kde=True, color='blue')
        plt.title(f'Distribution of {var}')
    
    plt.tight_layout()
    plt.show()
    
    # Cross-tabulation of selected categorical variables
    print("\nCross-tabulation of selected categorical variables:")
    print("\nSex vs School:")
    print(pd.crosstab(df['sex'], df['school'], normalize='index'))
    
    print("\nSex vs Internet:")
    print(pd.crosstab(df['sex'], df['internet'], normalize='index'))
    
    print("\nSchool vs Higher Education Aspiration:")
    print(pd.crosstab(df['school'], df['higher'], normalize='index'))


def perform_clustering(X, n_clusters=4):
    """
    Perform K-means clustering to identify groups of students with similar characteristics.

    Parameters
    ----------
    X : array-like
        The preprocessed feature matrix.
    n_clusters : int, default=4
        The number of clusters to form.

    Returns
    -------
    np.ndarray
        The cluster labels for each sample.
    """
    # Find optimal number of clusters using silhouette score
    silhouette_scores = []
    range_n_clusters = range(2, 11)
    
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"For n_clusters = {n_clusters}, the silhouette score is {silhouette_avg:.4f}")
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(range_n_clusters, silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Method For Optimal k')
    plt.show()
    
    # Use the optimal number of clusters
    optimal_clusters = range_n_clusters[silhouette_scores.index(max(silhouette_scores))]
    print(f"Optimal number of clusters: {optimal_clusters}")
    
    # Fit KMeans with optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(X)


def main():
    """
    Main function to execute the analysis.
    """
    # Load data
    file_path = "student-por.csv"
    data = load_data(file_path)
    
    # Explore data
    explore_data(data)
    
    # Identify key factors influencing performance
    identify_key_factors(data)
    
    # Visualize data
    visualize_data(data)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, feature_names, preprocessor = preprocess_data(data)
    
    print(f"\nPreprocessing complete. Training set shape: {X_train.shape}")
    
    # Perform clustering analysis
    cluster_labels = perform_clustering(X_train)
    
    # Now the data is ready for modeling (clustering, classification, etc.)
    # Additional code for modeling can be added here


if __name__ == "__main__":
    main()
