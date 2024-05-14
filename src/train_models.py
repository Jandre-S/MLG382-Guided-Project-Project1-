import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_deep_learning_model(df):
    """
    Train a deep learning model for loan status prediction.

    Parameters:
    df (pd.DataFrame): DataFrame containing preprocessed data.

    Returns:
    tf.keras.models.Sequential: Trained deep learning model.
    """
    # Assuming 'Loan_Status' is the target variable
    X = df.drop(['Loan_Status'], axis=1)
    y = df['Loan_Status']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build the deep learning model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, \nTest Accuracy: {accuracy}")

    return model

if __name__ == "__main__":
    # Example usage:
    #from preprocess_data import preprocess_data

    # Load preprocessed data
    df = pd.read_csv("../data/preprocessed_data.csv")

    # Preprocess data (if needed, adjust based on your preprocessing steps)
    #df = preprocess_data(df)

    # Train the deep learning model
    trained_model = train_deep_learning_model(df)

    # Save the trained model
    trained_model.save("../artifacts/model_1.h5")
    print("Trained Model Saved Successfully.")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# Assuming df is your DataFrame containing the loan data
# Replace 'Loan_Status' with your actual target column name if different
df = pd.read_csv("../data/Feature_engineering_data.csv")
# Assuming df is your DataFrame containing the loan data
# Replace 'Loan_Status' with your actual target column name if different
# Splitting data into features (X) and target variable (y)
X = df.drop(['Loan_Status'], axis=1)
y = df['Loan_Status']

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the deep learning model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
loss_fn = BinaryCrossentropy()
model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_data=(X_test_scaled, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Save the model
model.save('../artifacts/model_2.h5')
print("Model saved successfully.")