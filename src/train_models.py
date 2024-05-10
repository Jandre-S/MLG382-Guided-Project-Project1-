
#die code  vat validation en compare , hy vat validation en stuur vir die model en dan spoeg hy uit of die ou n loan gaan kry of nie 
#hierdie code  save nie die model na H5  maar jy kan dit net by las




import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


train_data = pd.read_csv('./data/raw_data.csv')
validation_data = pd.read_csv('./data/validation.csv')


le_status = LabelEncoder()


def preprocess_data(data, is_train=False):
   
    data.ffill(inplace=True)

    
    categorical_features = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']
    for feature in categorical_features:
        le = LabelEncoder()
        data[feature] = le.fit_transform(data[feature])

   
    scaler = StandardScaler()
    continuous_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    data[continuous_features] = scaler.fit_transform(data[continuous_features])
    
    if is_train:
        
        data['Loan_Status'] = le_status.fit_transform(data['Loan_Status'])
    
    return data


train_data = preprocess_data(train_data, is_train=True)
validation_data = preprocess_data(validation_data)


X_train = train_data.drop(['Loan_ID', 'Loan_Status'], axis=1)
y_train = to_categorical(train_data['Loan_Status'])
X_val = validation_data.drop(['Loan_ID'], axis=1)


model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=1)


predictions = model.predict(X_val)


predicted_status = le_status.inverse_transform(predictions.argmax(axis=1))


results = pd.DataFrame({
    'Loan_ID': validation_data['Loan_ID'],
    'Predicted_Loan_Status': ['Y' if status == 1 else 'N' for status in predicted_status]
})

print(results)





