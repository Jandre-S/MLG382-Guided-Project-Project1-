from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('../artifacts/model_2.h5')

# Define the home route
@app.route('/')
def home():
    return render_template('/Templates/index.html')

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
        
        # Preprocess the input (encode categorical variables)
        input_encoded = pd.get_dummies(input_df)
        
        # Make prediction using the model
        prediction = model.predict(input_encoded)[0]
        
        # Map prediction back to loan status
        if prediction == 'Y':
            loan_status = 'Approved'
        else:
            loan_status = 'Not Approved'
        
        return render_template('/Templates/result.html', loan_status=loan_status)

if __name__ == '__main__':
    app.run(debug=True)

