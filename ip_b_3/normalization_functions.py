import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os



def plot_box_plots(original_df, normalized_df, output_folder):
    fig, axes = plt.subplots(1, 2, figsize=(24, 18))

    numeric_original_df = original_df.select_dtypes(include=[np.number])
    numeric_normalized_df = normalized_df.select_dtypes(include=[np.number])

    if not numeric_original_df.empty:
        numeric_original_df.T.boxplot(ax=axes[0], vert=False)
        axes[0].set_title('Box Plot - Before Normalization')
        axes[0].set_xlabel('Concentration')
    else:
        axes[0].text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
        axes[0].set_title('No Data - Before Normalization')

    if not numeric_normalized_df.empty:
        numeric_normalized_df.T.boxplot(ax=axes[1], vert=False)
        axes[1].set_title('Box Plot - After Normalization')
        axes[1].set_xlabel('Normalized Concentration')
    else:
        axes[1].text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
        axes[1].set_title('No Data - After Normalization')

    plt.tight_layout(pad=6.0)
    plot_file = os.path.join(output_folder, 'box_plots.png')
    plt.savefig(plot_file)
    plt.close()

    print(f"Box plots saved as {plot_file}")

def plot_sample_view(original_df, normalized_df, output_folder):
    fig, axes = plt.subplots(2, 2, figsize=(30, 24))

    numeric_original_df = original_df.select_dtypes(include=[np.number])
    numeric_normalized_df = normalized_df.select_dtypes(include=[np.number])

    original_density = numeric_original_df.T.melt()['value']
    normalized_density = numeric_normalized_df.T.melt()['value']

    if not original_density.empty:
        original_density.plot(kind='density', ax=axes[0, 0], color='blue', label='Original')
        axes[0, 0].set_title('Density Plot - Before Normalization')
        axes[0, 0].set_xlabel('Concentration')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
    else:
        axes[0, 0].text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
        axes[0, 0].set_title('No Data - Before Normalization')

    if not normalized_density.empty:
        normalized_density.plot(kind='density', ax=axes[0, 1], color='red', label='Normalized')
        axes[0, 1].set_title('Density Plot - After Normalization')
        axes[0, 1].set_xlabel('Normalized Concentration')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
        axes[0, 1].set_title('No Data - After Normalization')

    if not numeric_original_df.empty:
        numeric_original_df.T.boxplot(ax=axes[1, 0], vert=True)
        axes[1, 0].set_title('Box Plot - Before Normalization')
        axes[1, 0].set_xlabel('Concentration')
    else:
        axes[1, 0].text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
        axes[1, 0].set_title('No Data - Before Normalization')

    if not numeric_normalized_df.empty:
        numeric_normalized_df.T.boxplot(ax=axes[1, 1], vert=True)
        axes[1, 1].set_title('Box Plot - After Normalization')
        axes[1, 1].set_xlabel('Normalized Concentration')
    else:
        axes[1, 1].text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
        axes[1, 1].set_title('No Data - After Normalization')

    for ax in axes[1, :]:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
        plt.setp(ax.get_xticklabels(), fontsize=10)

    plt.tight_layout(pad=8.0)
    sample_view_file = os.path.join(output_folder, 'sample_view.png')
    plt.savefig(sample_view_file)
    plt.close()

    print(f"Sample view plots saved as {sample_view_file}")

def plot_feature_view(original_df, normalized_df, output_folder):
    fig, axes = plt.subplots(2, 2, figsize=(30, 24))

    numeric_original_df = original_df.select_dtypes(include=[np.number])
    numeric_normalized_df = normalized_df.select_dtypes(include=[np.number])

    original_density = numeric_original_df.melt()['value']
    normalized_density = numeric_normalized_df.melt()['value']

    if not original_density.empty:
        original_density.plot(kind='density', ax=axes[0, 0], color='blue', label='Original')
        axes[0, 0].set_title('Density Plot - Before Normalization')
        axes[0, 0].set_xlabel('Concentration')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
    else:
        axes[0, 0].text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
        axes[0, 0].set_title('No Data - Before Normalization')

    if not normalized_density.empty:
        normalized_density.plot(kind='density', ax=axes[0, 1], color='red', label='Normalized')
        axes[0, 1].set_title('Density Plot - After Normalization')
        axes[0, 1].set_xlabel('Normalized Concentration')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
        axes[0, 1].set_title('No Data - After Normalization')

    if not numeric_original_df.empty:
        numeric_original_df.boxplot(ax=axes[1, 0], vert=True)
        axes[1, 0].set_title('Box Plot - Before Normalization')
        axes[1, 0].set_xlabel('Concentration')
    else:
        axes[1, 0].text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
        axes[1, 0].set_title('No Data - Before Normalization')

    if not numeric_normalized_df.empty:
        numeric_normalized_df.boxplot(ax=axes[1, 1], vert=True)
        axes[1, 1].set_title('Box Plot - After Normalization')
        axes[1, 1].set_xlabel('Normalized Concentration')
    else:
        axes[1, 1].text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
        axes[1, 1].set_title('No Data - After Normalization')

    for ax in axes[1, :]:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
        plt.setp(ax.get_xticklabels(), fontsize=10)

    plt.tight_layout(pad=8.0)
    feature_view_file = os.path.join(output_folder, 'feature_view.png')
    plt.savefig(feature_view_file)
    plt.close()

    print(f"Feature view plots saved as {feature_view_file}")

# Normalization functions



def normalization_by_reference_sample(df, reference_samples):
    reference_df = df.loc[reference_samples]
    median_reference = reference_df.median(axis=0)
    return df.div(median_reference, axis=1)

def normalization_by_reference_feature(df, reference_features):
    numeric_df = df.select_dtypes(include=[np.number])
    reference_values = numeric_df[reference_features].mean(axis=1)
    df[numeric_df.columns] = numeric_df.div(reference_values, axis=0)
    return df

def quantile_normalization(df):
    ranks = df.stack().groupby(df.rank(method='first').stack().astype(int)).mean()
    return df.rank(method='min').stack().astype(int).map(ranks).unstack()

def sample_specific_normalization(df, sample_factors):
    numeric_df = df.select_dtypes(include=[np.number])
    for sample, factor in sample_factors.items():
        numeric_df.loc[sample] = numeric_df.loc[sample] / factor
    df[numeric_df.columns] = numeric_df
    return df

# Transformation functions


def square_root_transformation(df):
    numeric_df = df.select_dtypes(include=[np.number])
    df[numeric_df.columns] = numeric_df.apply(np.sqrt)
    return df

def cube_root_transformation(df):
    numeric_df = df.select_dtypes(include=[np.number])
    df[numeric_df.columns] = numeric_df.apply(np.cbrt)
    return df

# Scaling functions
def mean_centering(df):
    numeric_df = df.select_dtypes(include=[np.number])
    df[numeric_df.columns] = numeric_df - numeric_df.mean()
    return df

def normalization_by_sum(df):
    numeric_df = df.select_dtypes(include=[np.number])
    df[numeric_df.columns] = numeric_df.div(numeric_df.sum(axis=1), axis=0)
    return df

def normalization_by_median(df):
    numeric_df = df.select_dtypes(include=[np.number])
    df[numeric_df.columns] = numeric_df.div(numeric_df.median(axis=1), axis=0)
    return df

def log_transformation(df):
    numeric_df = df.select_dtypes(include=[np.number])
    df[numeric_df.columns] = numeric_df.apply(lambda col: col.map(lambda x: np.log10(x) if x > 0 else 0))
    return df

def auto_scaling(df):
    numeric_df = df.select_dtypes(include=[np.number])
    df[numeric_df.columns] = (numeric_df - numeric_df.mean()) / numeric_df.std()
    return df

def pareto_scaling(df):
    numeric_df = df.select_dtypes(include=[np.number])
    df[numeric_df.columns] = (numeric_df - numeric_df.mean()) / np.sqrt(numeric_df.std())
    return df

def range_scaling(df):
    numeric_df = df.select_dtypes(include=[np.number])
    df[numeric_df.columns] = (numeric_df - numeric_df.min()) / (numeric_df.max() - numeric_df.min())
    return df

# Apply normalization
def apply_normalization(file_path, sample_norm_method, data_trans_method, data_scaling_method, sample_specific_factors=None, reference_samples_pqn=None, reference_features=None):
    df = pd.read_csv(file_path, index_col=0)

    def sanity_check_data(df):
        return df

    df = sanity_check_data(df)

    original_df = df.copy()

    def replace_min(df):
        numeric_df = df.select_dtypes(include=[np.number])
        min_val = numeric_df[numeric_df > 0].min().min() / 10
        numeric_df = numeric_df.replace(0, min_val)
        df[numeric_df.columns] = numeric_df
        return df

    df = replace_min(df)

    def prepare_prenorm_data(df):
        return df

    df = prepare_prenorm_data(df)

    # Apply sample normalization
    if sample_norm_method == "Normalization by sum":
        df = normalization_by_sum(df)
    elif sample_norm_method == "Normalization by median":
        df = normalization_by_median(df)
    elif sample_norm_method == "Normalization by a reference sample (PQN)" and reference_samples_pqn:
        df = normalization_by_reference_sample(df, reference_samples_pqn)
    elif sample_norm_method == "Normalization by a pooled sample from group (group PQN)" and reference_samples_pqn:
        df = normalization_by_reference_sample(df, reference_samples_pqn)
    elif sample_norm_method == "Normalization by reference feature" and reference_features:
        df = normalization_by_reference_feature(df, reference_features)
    elif sample_norm_method == "Quantile normalization":
        df = quantile_normalization(df)
    elif sample_norm_method == "Sample-specific normalization (i.e., weight, volume)" and sample_specific_factors:
        df = sample_specific_normalization(df, sample_specific_factors)

    # Apply data transformation
    if data_trans_method == "Log transformation (base 10)":
        df = log_transformation(df)
    elif data_trans_method == "Square root transformation":
        df = square_root_transformation(df)
    elif data_trans_method == "Cube root transformation":
        df = cube_root_transformation(df)

    # Apply data scaling
    if data_scaling_method == "Mean centering":
        df = mean_centering(df)
    elif data_scaling_method == "Auto scaling":
        df = auto_scaling(df)
    elif data_scaling_method == "Pareto scaling":
        df = pareto_scaling(df)
    elif data_scaling_method == "Range scaling":
        df = range_scaling(df)

    output_folder = "C:\\Users\\Kartik\\PycharmProjects\\ip_b_3\\normalisation result"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file = os.path.join(output_folder, "normalized_data.csv")
    df.to_csv(output_file, index=False)

    plot_box_plots(original_df, df, output_folder)

    print(f"Normalization completed and saved as {output_file}")

if __name__ == "__main__":
    file_path = "C:\\Users\\Kartik\\PycharmProjects\\ip_b_3\\normalisation result\\normalized_data.csv"
    sample_norm_method = "Normalization by sum"
    data_trans_method = "Log transformation (base 10)"
    data_scaling_method = "Auto scaling"
    apply_normalization(file_path, sample_norm_method, data_trans_method, data_scaling_method)


