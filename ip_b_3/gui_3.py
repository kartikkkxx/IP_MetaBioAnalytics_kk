import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys

def proceed_with_analysis(selected_option, normalized_file_path):
    if selected_option == "Biomarker Analysis":
        subprocess.Popen(["python", "C:\\Users\\Kartik\\PycharmProjects\\ip_b_3\\gui_biomarker_analysis.py", normalized_file_path])
    else:
        messagebox.showerror("Error", "Selected analysis option not supported.")

def create_gui(normalized_file_path):
    root = tk.Tk()
    root.title("Select Analysis")
    root.geometry("400x200")

    ttk.Label(root, text="Select Analysis Option:").pack(pady=10)
    analysis_option = tk.StringVar()
    ttk.Radiobutton(root, text="Biomarker Analysis", variable=analysis_option, value="Biomarker Analysis").pack()
    ttk.Radiobutton(root, text="Statistical Analysis", variable=analysis_option, value="Statistical Analysis").pack()
    ttk.Radiobutton(root, text="Joint Pathway Analysis", variable=analysis_option, value="Joint Pathway Analysis").pack()

    ttk.Button(root, text="Proceed", command=lambda: proceed_with_analysis(analysis_option.get(), normalized_file_path)).pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python gui_3.py <normalized_file_path>")
    else:
        create_gui(sys.argv[1])
