import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
import os
import matplotlib.colors as mcolors


def perform_multivariate_roc_analysis(file_path, classification_method, feature_ranking_method, latent_variables):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['Muscle loss'])
    y = df['Muscle loss'].apply(lambda x: 1 if 'cachexic' in x.lower() else 0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=10)

    if classification_method == "Linear SVM":
        classifier = SVC(kernel="linear", probability=True)
        y_proba = cross_val_predict(classifier, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
    elif classification_method == "PLS-DA":
        classifier = PLSRegression(n_components=latent_variables)
        y_pred = cross_val_predict(classifier, X_scaled, y, cv=cv)
        y_proba = y_pred[:, 0] if y_pred.ndim > 1 else y_pred  # PLSRegression doesn't have predict_proba, using predictions directly

    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multivariate ROC Analysis')
    plt.legend(loc='lower right')

    output_path = os.path.join("C:\\Users\\Kartik\\PycharmProjects\\ip_b_3\\biomarker_results",
                               "multivariate_roc_analysis.png")
    plt.savefig(output_path)
    plt.close()

    return f"Multivariate ROC analysis performed with {classification_method}, {feature_ranking_method}, {latent_variables} latent variables. Plot saved to {output_path}"


def plot_class_probabilities(file_path, classification_method, latent_variables):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['Muscle loss'])
    y = df['Muscle loss'].apply(lambda x: 1 if 'cachexic' in x.lower() else 0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=10)

    if classification_method == "Linear SVM":
        classifier = SVC(kernel="linear", probability=True)
        y_proba = cross_val_predict(classifier, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
    elif classification_method == "PLS-DA":
        classifier = PLSRegression(n_components=latent_variables)
        y_pred = cross_val_predict(classifier, X_scaled, y, cv=cv)
        y_proba = y_pred[:, 0] if y_pred.ndim > 1 else y_pred

    # Creating the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y_proba)), y_proba, c=y, cmap='coolwarm', alpha=0.7, edgecolor='k')
    plt.axvline(len(y_proba) / 2, color='gray', linestyle='--')
    plt.axhline(0.5, color='gray', linestyle='--')
    plt.xlabel('Samples')
    plt.ylabel('Predicted Class Probabilities')
    plt.title('Predicted Class Probabilities')
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(['Control', 'Cachexic'])

    output_path = os.path.join("C:\\Users\\Kartik\\PycharmProjects\\ip_b_3\\biomarker_results", "class_probabilities.png")
    plt.savefig(output_path)
    plt.close()

    # Confusion matrix
    y_pred_class = (y_proba >= 0.5).astype(int)
    conf_matrix = confusion_matrix(y, y_pred_class)

    fig, ax = plt.subplots()
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.6)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center')

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')

    confusion_matrix_path = os.path.join("C:\\Users\\Kartik\\PycharmProjects\\ip_b_3\\biomarker_results", "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    plt.close()

    return f"Class probabilities plot saved to {output_path}. Confusion matrix saved to {confusion_matrix_path}."


def plot_accuracies_with_features(file_path, classification_method, feature_ranking_method, latent_variables):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['Muscle loss'])
    y = df['Muscle loss'].apply(lambda x: 1 if 'cachexic' in x.lower() else 0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    feature_counts = [3, 5, 10, 20, 32, 63]
    accuracies = []

    cv = StratifiedKFold(n_splits=10)

    for k in feature_counts:
        if feature_ranking_method == "Univariate":
            selector = SelectKBest(f_classif, k=k)
            X_new = selector.fit_transform(X_scaled, y)
        elif feature_ranking_method == "T-statistic":
            selector = SelectKBest(score_func=f_classif, k=k)
            X_new = selector.fit_transform(X_scaled, y)
        elif feature_ranking_method == "SVM built-in":
            classifier = SVC(kernel="linear", probability=True)
            classifier.fit(X_scaled, y)
            feature_importance = np.abs(classifier.coef_[0])
            top_k_idx = np.argsort(feature_importance)[-k:]
            X_new = X_scaled[:, top_k_idx]
        elif feature_ranking_method == "PLS-DA built-in":
            classifier = PLSRegression(n_components=latent_variables)
            classifier.fit(X_scaled, y)
            feature_importance = np.abs(classifier.coef_[:, 0] if classifier.coef_.ndim > 1 else classifier.coef_)
            top_k_idx = np.argsort(feature_importance)[-k:]
            X_new = X_scaled[:, top_k_idx]
        elif feature_ranking_method == "RandomForest built-in":
            classifier = RandomForestClassifier()
            classifier.fit(X_scaled, y)
            feature_importance = classifier.feature_importances_
            top_k_idx = np.argsort(feature_importance)[-k:]
            X_new = X_scaled[:, top_k_idx]

        if classification_method == "Linear SVM":
            classifier = SVC(kernel="linear", probability=True)
        elif classification_method == "PLS-DA":
            classifier = PLSRegression(n_components=latent_variables)

        fold_accuracies = []

        for train, test in cv.split(X_new, y):
            if classification_method == "Linear SVM":
                model = classifier.fit(X_new[train], y[train])
                y_pred = model.predict(X_new[test])
            elif classification_method == "PLS-DA":
                model = classifier.fit(X_new[train], y[train])
                y_pred = model.predict(X_new[test])[:, 0] > 0.5

            fold_accuracies.append(accuracy_score(y[test], y_pred))

        accuracies.append(np.mean(fold_accuracies))

    plt.figure(figsize=(8, 6))
    plt.plot(feature_counts, accuracies, marker='o', color='b')
    plt.title('Predictive Accuracies with Different Features')
    plt.xlabel('Number of features')
    plt.ylabel('Predictive Accuracy')
    plt.ylim(0, 1)

    for i, txt in enumerate(accuracies):
        plt.annotate(f'{txt:.1%}', (feature_counts[i], accuracies[i]))

    output_path = os.path.join("C:\\Users\\Kartik\\PycharmProjects\\ip_b_3\\biomarker_results",
                               "predictive_accuracies.png")
    plt.savefig(output_path)
    plt.close()

    return f"Predictive accuracies plot saved to {output_path}"


def plot_important_biomarkers(file_path, classification_method, latent_variables):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['Muscle loss'])
    y = df['Muscle loss'].apply(lambda x: 1 if 'cachexic' in x.lower() else 0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if classification_method == "Linear SVM":
        classifier = SVC(kernel="linear", probability=True)
        classifier.fit(X_scaled, y)
        feature_importance = np.abs(classifier.coef_[0])
    elif classification_method == "PLS-DA":
        classifier = PLSRegression(n_components=latent_variables)
        classifier.fit(X_scaled, y)
        feature_importance = np.abs(classifier.coef_[:, 0] if classifier.coef_.ndim > 1 else classifier.coef_)

    feature_names = df.drop(columns=['Muscle loss']).columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Adding cachexic and control levels
    control_mean = df[df['Muscle loss'].apply(lambda x: 'control' in x.lower())].mean()
    cachexic_mean = df[df['Muscle loss'].apply(lambda x: 'cachexic' in x.lower())].mean()

    # Adjusting the plot to look like the first provided image
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=importance_df['Importance'].min(), vmax=importance_df['Importance'].max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    bars = ax.barh(importance_df['Feature'][:10][::-1], importance_df['Importance'][:10][::-1], color=cmap(norm(importance_df['Importance'][:10][::-1])))
    ax.set_xlabel('Average Importance')
    ax.set_ylabel('Feature')
    ax.set_title('Top 10 Important Features')

    for index, value in enumerate(importance_df['Importance'][:10][::-1]):
        ax.text(value, index, f'{value:.2f}')

    # Adding the paired color boxes for control and cachexic
    feature_names_top = importance_df['Feature'][:10].values
    for i, feature in enumerate(feature_names_top[::-1]):
        control_color = cmap(norm(control_mean[feature]))
        cachexic_color = cmap(norm(cachexic_mean[feature]))
        ax.add_patch(plt.Rectangle((0.03, i - 0.4), 0.02, 0.8, color=control_color, transform=ax.transAxes, clip_on=False))
        ax.add_patch(plt.Rectangle((0.06, i - 0.4), 0.02, 0.8, color=cachexic_color, transform=ax.transAxes, clip_on=False))

    # Adding labels for the color boxes
    plt.text(1.02, -0.1, 'Control', ha='left', va='center', transform=ax.transAxes, rotation=45)
    plt.text(1.08, -0.1, 'Cachexic', ha='left', va='center', transform=ax.transAxes, rotation=45)
    cbar = plt.colorbar(sm, ticks=[0, 1], aspect=50)
    cbar.set_label('Importance')

    plt.tight_layout()

    output_path = os.path.join("C:\\Users\\Kartik\\PycharmProjects\\ip_b_3\\biomarker_results",
                               "important_biomarkers.png")
    plt.savefig(output_path)
    plt.close()

    return f"Important biomarkers plot saved to {output_path}"


def create_gui():
    def on_analyze():
        try:
            file_path = "C:\\Users\\Kartik\\PycharmProjects\\ip_b_3\\normalisation result\\normalized_data.csv"
            classification_method = classification_method_var.get()
            feature_ranking_method = feature_ranking_method_var.get()
            latent_variables = int(latent_variables_var.get())

            if not os.path.isfile(file_path):
                messagebox.showerror("Error", "Invalid file path.")
                return

            results1 = perform_multivariate_roc_analysis(file_path, classification_method, feature_ranking_method, latent_variables)
            results2 = plot_class_probabilities(file_path, classification_method, latent_variables)
            results3 = plot_accuracies_with_features(file_path, classification_method, feature_ranking_method, latent_variables)
            results4 = plot_important_biomarkers(file_path, classification_method, latent_variables)

            messagebox.showinfo("Results", f"{results1}\n\n{results2}\n\n{results3}\n\n{results4}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    root = tk.Tk()
    root.title("Metabolomics Data Analysis")

    mainframe = ttk.Frame(root, padding="10 10 20 20")
    mainframe.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    classification_method_var = tk.StringVar(value="Linear SVM")
    feature_ranking_method_var = tk.StringVar(value="Univariate")
    latent_variables_var = tk.StringVar(value="2")

    ttk.Label(mainframe, text="Classification Method:").grid(row=0, column=0, sticky=tk.W)
    ttk.Combobox(mainframe, textvariable=classification_method_var, values=["Linear SVM", "PLS-DA"]).grid(row=0, column=1, sticky=(tk.W, tk.E))

    ttk.Label(mainframe, text="Feature Ranking Method:").grid(row=1, column=0, sticky=tk.W)
    ttk.Combobox(mainframe, textvariable=feature_ranking_method_var, values=["Univariate", "T-statistic", "SVM built-in", "PLS-DA built-in", "RandomForest built-in"]).grid(row=1, column=1, sticky=(tk.W, tk.E))

    ttk.Label(mainframe, text="Latent Variables (for PLS-DA):").grid(row=2, column=0, sticky=tk.W)
    ttk.Entry(mainframe, width=5, textvariable=latent_variables_var).grid(row=2, column=1, sticky=(tk.W, tk.E))

    ttk.Button(mainframe, text="Analyze", command=on_analyze).grid(row=3, column=1, sticky=tk.E)

    root.mainloop()

create_gui()
