import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import subprocess

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
    from sklearn.metrics import roc_curve, auc

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

def create_gui():
    root = tk.Tk()
    root.title("Biomarker Analysis")
    root.geometry("600x400")

    global feature_combobox

    ttk.Label(root, text="Select Feature:").grid(row=0, column=0, padx=10, pady=10, sticky='w')
    feature_combobox = ttk.Combobox(root)
    feature_combobox.grid(row=0, column=1, padx=10, pady=10, sticky='w')

    ttk.Button(root, text="Analyze and Plot PCA", command=analyze_and_plot_pca).grid(row=1, column=0, columnspan=3, pady=20)
    ttk.Button(root, text="Analyze and Plot ROC", command=analyze_and_plot_roc).grid(row=2, column=0, columnspan=3, pady=20)
    ttk.Button(root, text="Proceed to Biomarker 2", command=proceed_to_biomarker_2).grid(row=3, column=0, columnspan=3, pady=20)

    load_features()

    root.mainloop()

def load_features():
    try:
        df = pd.read_csv("C:\\Users\\Kartik\\PycharmProjects\\ip_b_3\\normalisation result\\normalized_data.csv")
        features = [feature.strip("'") for feature in df.columns[1:]]  # Exclude the first column (assumed to be the index or ID column)
        feature_combobox['values'] = features
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load features: {str(e)}")

if __name__ == "__main__":
    create_gui()
