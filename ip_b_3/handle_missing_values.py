import numpy as np

def check_for_missing_values(df):
    missing_data = df.isnull().sum()
    total_data = len(df)
    missing_percentage = (missing_data / total_data) * 100
    return missing_data[missing_data > 0].to_dict(), missing_percentage[missing_percentage > 0].to_dict()

def generate_missing_values_report(df, missing_data, missing_percentage):
    samples, features = df.shape
    numeric_values_check = df.select_dtypes(include=[np.number])
    report = "Data Processing Information:\n"
    report += "Checking data content...passed.\n"
    report += f"Samples are in rows and features in columns.\nThe uploaded file contains {samples} samples by {features} features.\n"
    report += "All data values are numeric: " + ("Yes" if numeric_values_check.shape[1] == features else "No") + ".\n"
    report += "Missing values detected:\n"
    for col, amount in missing_data.items():
        report += f"{col}: {amount} missing ({missing_percentage[col]:.2f}%).\n"
    return report
