import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
import determine_data_type  # Assumes functionality to load data and determine data type
import pandas as pd
import os

file_path = ""

def create_gui():
    root = tk.Tk()
    root.title("Upload Your Data")
    root.geometry("1000x700")  # Adjusted for additional content and layout

    # Styling
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TButton', font=('Calibri', 12), padding=10)
    style.configure('TLabel', font=('Calibri', 12), padding=10)
    style.configure('TRadiobutton', font=('Calibri', 12))
    style.configure('TCombobox', font=('Calibri', 12))

    # Data Type Selection
    data_type_var = tk.StringVar(value="Concentrations")
    ttk.Label(root, text="Data Type:").grid(row=0, column=0, padx=10, pady=5, sticky='w')
    ttk.Radiobutton(root, text="Concentrations", variable=data_type_var, value="Concentrations").grid(row=0, column=1, sticky='w')
    ttk.Radiobutton(root, text="Spectral bins", variable=data_type_var, value="Spectral bins").grid(row=0, column=2, sticky='w')
    ttk.Radiobutton(root, text="Peak intensities", variable=data_type_var, value="Peak intensities").grid(row=0, column=3, sticky='w')

    # File Format Selection
    data_type_label = ttk.Label(root, text="Detected Data Type: Unknown")
    data_type_label.grid(row=5, column=0, columnspan=4)
    format_var = tk.StringVar(value="Samples in rows")
    ttk.Label(root, text="Format:").grid(row=1, column=0, padx=10, pady=5, sticky='w')
    ttk.Combobox(root, textvariable=format_var, values=['Samples in rows', 'Samples in columns'], state="readonly").grid(row=1, column=1, sticky='w', columnspan=3)

    # File Selection
    ttk.Label(root, text="Data File:").grid(row=2, column=0, padx=10, pady=5, sticky='w')
    file_label = ttk.Label(root, text="No file selected", font=('Calibri', 12), relief='sunken')
    file_label.grid(row=2, column=1, columnspan=3, sticky='ew', padx=10, pady=10)
    ttk.Button(root, text="Choose", command=lambda: choose_file(file_label, report_text, data_type_label)).grid(row=2, column=4, padx=10, pady=10)

    # Detailed Text Box for Data Processing Information
    report_text = scrolledtext.ScrolledText(root, height=20, width=120)
    report_text.grid(row=3, column=0, columnspan=5, padx=10, pady=10)

    # Buttons for additional actions
    ttk.Button(root, text="Edit Groups", command=lambda: edit_groups(report_text)).grid(row=4, column=0, padx=10, pady=10)
    ttk.Button(root, text="Missing Values", command=lambda: handle_missing_values_action(report_text)).grid(row=4, column=1, padx=10, pady=10)
    ttk.Button(root, text="Proceed", command=lambda: proceed_with_data(report_text, data_type_var.get(), file_path)).grid(row=4, column=4, padx=10, pady=10)

    root.mainloop()

def edit_groups(report_text):
    report_text.insert(tk.END, "\nEdit Groups button clicked. Group editing functionality not implemented yet.\n")

def handle_missing_values_action(report_text):
    report_text.insert(tk.END, "\nHandle Missing Values button clicked. Missing values handling not implemented yet.\n")
    handle_missing_values_gui()

def handle_missing_values_gui():
    def apply_missing_values_handling():
        global file_path  # Update global file_path to the new file
        method = missing_values_var.get()
        if file_path:
            df = pd.read_csv(file_path)
            if method == "Replace with Mean":
                df.fillna(df.mean(), inplace=True)
            elif method == "Replace with Median":
                df.fillna(df.median(), inplace=True)
            elif method == "Replace with Mode":
                df.fillna(df.mode().iloc[0], inplace=True)
            elif method == "Replace with Zero":
                df.fillna(0, inplace=True)
            elif method == "Replace with Custom Value":
                df.fillna(float(custom_value.get()), inplace=True)

            # Save the updated dataframe to a new file
            updated_file_path = file_path.replace(".csv", "_handled.csv")
            df.to_csv(updated_file_path, index=False)
            messagebox.showinfo("Missing Values", f"Missing values handled. Updated file saved as {updated_file_path}")
            file_path = updated_file_path
        else:
            messagebox.showerror("Error", "No file selected.")

    missing_values_root = tk.Toplevel()
    missing_values_root.title("Handle Missing Values")
    missing_values_root.geometry("400x300")

    missing_values_var = tk.StringVar(value="None")
    ttk.Label(missing_values_root, text="Choose a method to handle missing values:").pack(pady=10)
    options = [
        "None", "Replace with Mean", "Replace with Median", "Replace with Mode", "Replace with Zero", "Replace with Custom Value"
    ]
    for option in options:
        ttk.Radiobutton(missing_values_root, text=option, variable=missing_values_var, value=option).pack(anchor='w')

    custom_value = tk.StringVar()
    ttk.Label(missing_values_root, text="Custom Value:").pack(pady=5)
    ttk.Entry(missing_values_root, textvariable=custom_value).pack()

    ttk.Button(missing_values_root, text="Apply", command=apply_missing_values_handling).pack(pady=20)

def proceed_with_data(report_text, data_type, file_path):
    report_text.insert(tk.END, "\nProceed button clicked. Further processing steps would go here.\n")
    if data_type == "Concentrations":
        import gui2_concentration_table
        gui2_concentration_table.create_gui(file_path)
    elif data_type == "Spectral bins":
        import gui2_spectral_bins
        gui2_spectral_bins.create_gui(file_path)
    elif data_type == "Peak intensities":
        import gui2_peak_intensity
        gui2_peak_intensity.create_gui(file_path)

def choose_file(file_label, report_text, data_type_label):
    global file_path
    file_path = filedialog.askopenfilename(title="Select a data file", filetypes=[("CSV Files", "*.csv"), ("All files", "*.*")])
    if file_path:
        df, data_type, data_info = determine_data_type.analyze_data(file_path)
        file_label.config(text=f"Selected file: {file_path.split('/')[-1]}")
        data_type_label.config(text=f"Detected Data Type: {data_type}")
        report_text.insert(tk.END, f"Data Integrity Check:\n{data_info}\n")

if __name__ == "__main__":
    create_gui()
