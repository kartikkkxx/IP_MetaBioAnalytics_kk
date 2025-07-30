import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import normalization_functions
import os
import subprocess

sample_specific_factors = {}
reference_samples_pqn = []
reference_features = []

def create_gui(file_path):
    root = tk.Tk()
    root.title("Normalization - Concentration Table")
    root.geometry("800x800")

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        messagebox.showerror("File Not Found", f"The file at {file_path} was not found.")
        return

    features = list(df.columns[1:])  # Assuming first column is the sample ID
    samples = list(df.iloc[:, 0])  # Assuming first column contains sample IDs

    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TButton', font=('Calibri', 12), padding=10)
    style.configure('TLabel', font=('Calibri', 12), padding=10)

    # Create a canvas and scrollbar for scrolling
    container = ttk.Frame(root)
    canvas = tk.Canvas(container)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    container.pack(fill="both", expand=True)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    ttk.Label(scrollable_frame, text="Normalization Overview:", font=('Calibri', 14, 'bold')).grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky='w')
    ttk.Label(scrollable_frame, text="Sample normalization is for general-purpose adjustment for systematic differences among samples.\n"
                                      "Data transformation applies a mathematical transformation on individual values.\n"
                                      "Data scaling adjusts each variable by a scaling factor based on the dispersion of the variable.",
              font=('Calibri', 12)).grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky='w')

    ttk.Label(scrollable_frame, text="Sample Normalization:", font=('Calibri', 12, 'bold')).grid(row=2, column=0, padx=10, pady=5, sticky='w')
    sample_norm_var = tk.StringVar(value="Normalization by sum")
    sample_norm_options = [
        "None", "Normalization by sum", "Normalization by median",
        "Normalization by a reference sample (PQN)",
        "Normalization by a pooled sample from group (group PQN)", "Normalization by reference feature",
        "Quantile normalization", "Sample-specific normalization (i.e., weight, volume)"
    ]
    for i, option in enumerate(sample_norm_options):
        ttk.Radiobutton(scrollable_frame, text=option, variable=sample_norm_var, value=option).grid(row=3 + i, column=0, padx=10, sticky='w')

    ttk.Label(scrollable_frame, text="Data Transformation:", font=('Calibri', 12, 'bold')).grid(row=2, column=1, padx=10, pady=5, sticky='w')
    data_trans_var = tk.StringVar(value="Log transformation (base 10)")
    data_trans_options = [
        "None", "Log transformation (base 10)", "Square root transformation", "Cube root transformation"
    ]
    for i, option in enumerate(data_trans_options):
        ttk.Radiobutton(scrollable_frame, text=option, variable=data_trans_var, value=option).grid(row=3 + i, column=1, padx=10, sticky='w')

    ttk.Label(scrollable_frame, text="Data Scaling:", font=('Calibri', 12, 'bold')).grid(row=7, column=1, padx=10, pady=5, sticky='w')
    data_scaling_var = tk.StringVar(value="Auto scaling")
    data_scaling_options = [
        "None", "Mean centering", "Auto scaling", "Pareto scaling", "Range scaling"
    ]
    for i, option in enumerate(data_scaling_options):
        ttk.Radiobutton(scrollable_frame, text=option, variable=data_scaling_var, value=option).grid(row=8 + i, column=1, padx=10, sticky='w')

    ttk.Button(scrollable_frame, text="Specify Sample-specific Normalization Factors",
               command=lambda: specify_sample_specific_factors(samples)).grid(row=13, column=0, columnspan=2, padx=10, pady=5)
    ttk.Button(scrollable_frame, text="Specify Reference Samples for PQN",
               command=lambda: specify_reference_samples_pqn(samples)).grid(row=14, column=0, columnspan=2, padx=10, pady=5)
    ttk.Button(scrollable_frame, text="Specify Reference Features",
               command=lambda: specify_reference_features(features)).grid(row=15, column=0, columnspan=2, padx=10, pady=5)

    ttk.Button(scrollable_frame, text="Normalize",
               command=lambda: normalize_data(sample_norm_var.get(), data_trans_var.get(), data_scaling_var.get(), file_path)).grid(row=16, column=0, columnspan=2, pady=20)
    ttk.Button(scrollable_frame, text="View Sample View Result", command=lambda: view_sample_result(file_path)).grid(row=17, column=0, columnspan=2, pady=10)
    ttk.Button(scrollable_frame, text="View Feature View Result", command=lambda: view_feature_result(file_path)).grid(row=18, column=0, columnspan=2, pady=10)
    ttk.Button(scrollable_frame, text="Proceed", command=proceed).grid(row=19, column=0, columnspan=2, pady=10)

    root.mainloop()

def create_scrollable_top_window(title, elements_func, command_save):
    top = tk.Toplevel()
    top.title(title)
    container = ttk.Frame(top)
    canvas = tk.Canvas(container)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    container.pack(fill="both", expand=True)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    elements = elements_func(scrollable_frame)
    ttk.Button(scrollable_frame, text="Save", command=lambda: command_save(top)).grid(row=len(elements) + 1, column=0, columnspan=2, padx=10, pady=10)

def specify_sample_specific_factors(samples):
    sample_entries = {}

    def save_factors(top):
        global sample_specific_factors
        sample_specific_factors = {sample: float(entry.get()) for sample, entry in sample_entries.items()}
        top.destroy()

    def create_elements(scrollable_frame):
        for i, sample in enumerate(samples):
            ttk.Label(scrollable_frame, text=sample).grid(row=i + 1, column=0, padx=10, pady=5, sticky='w')
            entry = ttk.Entry(scrollable_frame)
            entry.insert(0, '1.0')
            entry.grid(row=i + 1, column=1, padx=10, pady=5, sticky='w')
            sample_entries[sample] = entry
        return sample_entries

    create_scrollable_top_window("Sample-specific Normalization Factors", create_elements, save_factors)

def specify_reference_samples_pqn(samples):
    reference_vars = {}

    def save_samples(top):
        global reference_samples_pqn
        reference_samples_pqn = [sample for sample, var in reference_vars.items() if var.get()]
        top.destroy()

    def create_elements(scrollable_frame):
        for i, sample in enumerate(samples):
            var = tk.BooleanVar()
            ttk.Checkbutton(scrollable_frame, text=sample, variable=var).grid(row=i + 1, column=0, padx=10, pady=5, sticky='w')
            reference_vars[sample] = var
        return reference_vars

    create_scrollable_top_window("Reference Samples for PQN", create_elements, save_samples)

def specify_reference_features(features):
    feature_vars = {}

    def save_features(top):
        global reference_features
        reference_features = [feature for feature, var in feature_vars.items() if var.get()]
        top.destroy()

    def create_elements(scrollable_frame):
        for i, feature in enumerate(features):
            var = tk.BooleanVar()
            ttk.Checkbutton(scrollable_frame, text=feature, variable=var).grid(row=i + 1, column=0, padx=10, pady=5, sticky='w')
            feature_vars[feature] = var
        return feature_vars

    create_scrollable_top_window("Reference Features", create_elements, save_features)

def normalize_data(sample_norm_method, data_trans_method, data_scaling_method, file_path):
    global sample_specific_factors, reference_samples_pqn, reference_features
    normalization_functions.apply_normalization(file_path, sample_norm_method, data_trans_method, data_scaling_method,
                                                sample_specific_factors, reference_samples_pqn, reference_features)
    messagebox.showinfo("Normalization", f"Normalization completed with:\nSample Normalization: {sample_norm_method}\nData Transformation: {data_trans_method}\nData Scaling: {data_scaling_method}")

def view_sample_result(file_path):
    output_folder = "C:\\Users\\Kartik\\PycharmProjects\\ip_b_3\\normalisation result"
    sample_view_file = os.path.join(output_folder, 'sample_view.png')
    if os.path.exists(sample_view_file):
        os.system(f'start {sample_view_file}')
    else:
        messagebox.showinfo("View Result", "Sample view normalization result file not found.")

def view_feature_result(file_path):
    output_folder = "C:\\Users\\Kartik\\PycharmProjects\\ip_b_3\\normalisation result"
    feature_view_file = os.path.join(output_folder, 'feature_view.png')
    if os.path.exists(feature_view_file):
        os.system(f'start {feature_view_file}')
    else:
        messagebox.showinfo("View Result", "Feature view normalization result file not found.")

def proceed():
    normalized_file_path = "C:\\Users\\Kartik\\PycharmProjects\\ip_b_3\\normalisation result\\normalized_data.csv"
    if os.path.exists(normalized_file_path):
        subprocess.Popen(["python", "C:\\Users\\Kartik\\PycharmProjects\\ip_b_3\\gui_3.py", normalized_file_path])
        messagebox.showinfo("Proceed", "Proceeding to the next step.")
    else:
        messagebox.showerror("File Not Found", "Normalized data file not found. Please ensure normalization is completed successfully.")

if __name__ == "__main__":
    file_path = "C:\\Users\\Kartik\\Downloads\\human_cachexia_handled.csv"  # Change this to the path of your data file
    create_gui(file_path)
