import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from prepare_data import load_raw_data

def preprocess_data(df):
    # Convert binary categorical features into numerical (0s and 1s)
    binary_mapping = {
        'Male': 1, 'Female': 0,
        'Yes': 1, 'No': 0,
        'Graduate': 1, 'Not Graduate': 0,
        'Y': 1, 'N': 0
    }

    binary_categorical_features = ['Gender', 'Married', 'Education', 'Self_Employed']
    for feature in binary_categorical_features:
        df[feature] = df[feature].map(binary_mapping)
        df[feature].fillna(df[feature].mode()[0], inplace=True)

    # Perform one-hot encoding for non-binary categorical features
    categorical_features = df.select_dtypes(include=['object'])
    non_binary_categorical_features = categorical_features.columns[categorical_features.nunique() > 2].tolist()

    for feature in non_binary_categorical_features:
        if feature != 'Loan_ID':  # Skip Loan_ID if it's present
            one_hot_encoded = pd.get_dummies(df[feature], prefix=feature, dtype=int)
            df = pd.concat([df, one_hot_encoded], axis=1)
    
    # Drop the original categorical columns
    df.drop(categorical_features.columns, axis=1, inplace=True)

    # Handle missing values by filling with the mean
    features_with_missing = df.columns[df.isnull().any()]
    for feature in features_with_missing:
        mean_value = df[feature].mean()
        df[feature].fillna(mean_value, inplace=True)

    return df

if __name__ == "__main__":
    # Example usage: Load your DataFrame and apply preprocessing
    df = load_raw_data  # Adjust path as needed
    preprocessed_df = preprocess_data(df)

    # Optionally, perform additional processing or save preprocessed data
    preprocessed_df.to_csv('../data/preprocessed_data.csv', index=False)  # Save preprocessed data
