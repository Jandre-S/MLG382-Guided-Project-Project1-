from flask import Flask, render_template, request
import pandas as pd
import joblib

from preprocess_data import preprocess_data

app = Flask(__name__)

# Load the trained model
model = joblib.load("./artifacts/model_2.pkl")

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    if request.method == 'POST':
        input_data = {
            'Gender': request.form['gender'],
            'Married': request.form['married'],
            'Dependents': request.form['dependents'],
            'Education': request.form['education'],
            'Self_Employed': request.form['self_employed'],
            'ApplicantIncome': float(request.form['applicant_income']),
            'CoapplicantIncome': float(request.form['coapplicant_income']),
            'LoanAmount': float(request.form['loan_amount']),
            'Loan_Amount_Term': float(request.form['loan_amount_term']),
            'Credit_History': float(request.form['credit_history']),
            'Property_Area': request.form['property_area']
        }
        
        # Create a DataFrame from user input
        input_df = pd.DataFrame([input_data])

        preprocess_df = preprocess_data(input_df)
        
        # Preprocess the input (encode categorical variables)
        #input_encoded = pd.get_dummies(preprocess_df)
        # Make prediction using the model

        prediction = model.predict(preprocess_df)[0]

        # Map prediction back to loan status
        if prediction == 'Y':
            loan_status = 'Approved'
        else:
            loan_status = 'Not Approved'
        
        return render_template('result.html', Loan_status=loan_status)

if __name__ == '__main__':
    app.run(debug=True)

