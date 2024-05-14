import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    """
    Preprocess the input DataFrame by removing outliers and normalizing the data.

    Parameters:
    df (pd.DataFrame): Input DataFrame to be preprocessed.

    Returns:
    pd.DataFrame: Preprocessed DataFrame.
    """
    # Remove outliers from specific columns
    df = remove_outliers(df, ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"])

    # Normalize the data
    df = normalize_data(df)

    return df

def remove_outliers(df, columns, threshold=0.9):
    """
    Remove outliers from specified columns in the DataFrame based on a threshold.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to remove outliers from.
    threshold (float): Threshold value (quantile) for outlier removal (default: 0.9).

    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    for column in columns:
        original_shape = df.shape[0]
        df = df[df[column] < df[column].quantile(threshold)]
        new_shape = df.shape[0]
        print(f"Removed outliers from '{column}'. Original size: {original_shape}, New size: {new_shape}")

    return df

def normalize_data(df):
    """
    Normalize numerical columns in the DataFrame using Min-Max scaling.

    Parameters:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: DataFrame with normalized numerical columns.
    """
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()

    scaler = MinMaxScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    print("Data Normalized Successfully.")

    return df

if __name__ == "__main__":
    # Example usage (for testing purposes):
    raw_data = pd.read_csv("../data/preprocessed_data.csv")

    # Apply preprocessing steps
    preprocessed_data = preprocess_data(raw_data)

    # Save preprocessed data to a new CSV file
    preprocessed_data.to_csv("../data/Feature_engineering_data.csv", index=False)
    print("Preprocessed Data Saved Successfully.")
