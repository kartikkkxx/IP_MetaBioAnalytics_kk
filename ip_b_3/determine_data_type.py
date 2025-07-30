import pandas as pd

def analyze_data(filepath):
    df = pd.read_csv(filepath)
    data_type = determine_data_type(df)
    data_info = check_data_integrity(df)
    return df, data_type, data_info

def determine_data_type(df):
    columns = [col.lower() for col in df.columns]
    if any("concentration" in col for col in columns):
        return "Concentrations"
    elif any("bin" in col for col in columns):
        return "Spectral bins"
    elif any("intensity" in col for col in columns):
        return "Peak intensities"
    return "Unknown Data Type"

def check_data_integrity(df):
    info = []

    # Check for non-numeric values
    non_numeric = df.select_dtypes(exclude=['number']).columns
    if len(non_numeric) > 0:
        info.append("Non-numeric values found in columns: " + ", ".join(non_numeric))
    else:
        info.append("All data values are numeric.")

    # Check for missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        info.append(f"A total of {missing_values} missing values were detected.")
    else:
        info.append("A total of 0 (0%) missing values were detected.")

    # Other checks can be added here as needed

    # Example checks (Modify as per actual requirements)
    info.append("Samples are in rows and features in columns.")
    info.append(f"The uploaded data file contains {df.shape[0]} (samples) by {df.shape[1]} (compounds) data matrix.")
    # Add other checks here

    return "\n".join(info)