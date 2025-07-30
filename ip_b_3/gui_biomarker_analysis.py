import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import sys
import subprocess
import normalization_functions
def normalize_data(sample_norm_method, data_trans_method, data_scaling_method, file_path):
    try:
        sample_specific_factors = None  # Define or fetch these values if required
        reference_samples_pqn = None    # Define or fetch these values if required
        reference_features = None       # Define or fetch these values if required

        normalization_functions.apply_normalization(
            file_path,
            sample_norm_method,
            data_trans_method,
            data_scaling_method,
            sample_specific_factors,
            reference_samples_pqn,
            reference_features
        )
        messagebox.showinfo("Normalization", "Normalization completed successfully.")
    except Exception as e:
        messagebox.showerror("Error", str(e))
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()  # Drop rows with NaN values
    return df


def perform_pca(df):
    df_scaled = StandardScaler().fit_transform(df)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_scaled)
    return principal_components, pca


def plot_pca(df, principal_components, output_path):
    plt.figure(figsize=(10, 7))
    colors = df.index.to_series().astype(str).str.contains("cachexic").map({True: 'r', False: 'g'})
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c=colors)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA - Normalized Data')
    plt.savefig(output_path)
    plt.close()


def perform_roc_analysis(df, selected_feature):
    y_true = df['Muscle loss'].str.contains("cachexic").astype(int)
    y_scores = df[selected_feature]

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        raise ValueError("No positive samples in y_true or no negative samples in y_true.")

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return fpr, tpr, roc_auc, optimal_threshold, fpr[optimal_idx], tpr[optimal_idx]


def plot_roc_and_box(df, selected_feature, output_path):
    plt.figure(figsize=(14, 7))

    ax1 = plt.subplot(121)
    fpr, tpr, roc_auc, optimal_threshold, optimal_fpr, optimal_tpr = perform_roc_analysis(df, selected_feature)
    ax1.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax1.scatter(optimal_fpr, optimal_tpr, color='red', s=100, label=f'Optimal Threshold ({optimal_threshold:.2f})')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'Receiver Operating Characteristic (ROC) - {selected_feature}')
    ax1.legend(loc='lower right')

    ax2 = plt.subplot(122)
    df['Group'] = np.where(df['Muscle loss'].str.contains("cachexic"), 'cachexic', 'control')
    sns.boxplot(x='Group', y=selected_feature, data=df, ax=ax2)
    sns.stripplot(x='Group', y=selected_feature, data=df, ax=ax2, color='black', jitter=0.2, size=3)
    ax2.set_title(f'Boxplot - {selected_feature}')

    plt.savefig(output_path)
    plt.close()


def random_forest_classification(df, selected_feature):
    X = df.drop(columns=['Muscle loss'])
    y = df['Muscle loss'].str.contains("cachexic").astype(int)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the model
    rf = RandomForestClassifier(random_state=42)

    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Perform grid search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_rf = grid_search.best_estimator_

    # Make predictions
    y_pred = best_rf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy, best_rf


def analyze_and_plot_pca():
    try:
        df = load_data("C:\\Users\\Kartik\\PycharmProjects\\ip_b_3\\normalisation result\\normalized_data.csv")
        df_numeric = df.select_dtypes(include=[np.number])  # Select only numeric columns
        principal_components, pca = perform_pca(df_numeric)
        output_path = "C:\\Users\\Kartik\\PycharmProjects\\ip_b_3\\biomarker_results\\pca_plot.png"
        plot_pca(df, principal_components, output_path)
        messagebox.showinfo("PCA Analysis", f"PCA plot saved to {output_path}")
    except Exception as e:
        messagebox.showerror("Error", str(e))


def analyze_and_plot_roc():
    try:
        selected_feature = feature_combobox.get()
        df = load_data("C:\\Users\\Kartik\\PycharmProjects\\ip_b_3\\normalisation result\\normalized_data.csv")
        output_path = f"C:\\Users\\Kartik\\PycharmProjects\\ip_b_3\\biomarker_results\\roc_{selected_feature}.png"
        plot_roc_and_box(df, selected_feature, output_path)
        messagebox.showinfo("ROC Analysis", f"ROC and box plot saved to {output_path}")
    except Exception as e:
        messagebox.showerror("Error", str(e))


def proceed_to_biomarker_2():
    try:
        subprocess.run(["python", "biomarker_2.py"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to launch biomarker_2.py: {str(e)}")


def analyze_random_forest():
    try:
        selected_feature = feature_combobox.get()
        df = load_data("C:\\Users\\Kartik\\PycharmProjects\\ip_b_3\\normalisation result\\normalized_data.csv")
        accuracy, model = random_forest_classification(df, selected_feature)
        messagebox.showinfo("Random Forest Analysis", f"Random Forest model accuracy: {accuracy:.2f}")
    except Exception as e:
        messagebox.showerror("Error", str(e))


def create_gui():
    root = tk.Tk()
    root.title("Biomarker Analysis")
    root.geometry("600x400")

    global feature_combobox

    ttk.Label(root, text="Select Feature:").grid(row=0, column=0, padx=10, pady=10, sticky='w')
    feature_combobox = ttk.Combobox(root)
    feature_combobox.grid(row=0, column=1, padx=10, pady=10, sticky='w')

    ttk.Button(root, text="Analyze and Plot PCA", command=analyze_and_plot_pca).grid(row=1, column=0, columnspan=3,
                                                                                     pady=20)
    ttk.Button(root, text="Analyze and Plot ROC", command=analyze_and_plot_roc).grid(row=2, column=0, columnspan=3,
                                                                                     pady=20)
    ttk.Button(root, text="Analyze Random Forest", command=analyze_random_forest).grid(row=3, column=0, columnspan=3,
                                                                                       pady=20)
    ttk.Button(root, text="Proceed to Biomarker 2", command=proceed_to_biomarker_2).grid(row=4, column=0, columnspan=3,
                                                                                         pady=20)

    load_features()

    root.mainloop()


def load_features():
    try:
        df = pd.read_csv("C:\\Users\\Kartik\\PycharmProjects\\ip_b_3\\normalisation result\\normalized_data.csv")
        features = [feature.strip("'") for feature in
                    df.columns[1:]]  # Exclude the first column (assumed to be the index or ID column)
        feature_combobox['values'] = features
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load features: {str(e)}")


if __name__ == "__main__":
    create_gui()
