import pandas as pd

def load_raw_data():
    raw_data = pd.read_csv("../data/raw_data.csv")
    return raw_data

def load_validation():
    validation = pd.read_csv("../data/validation.csv")
    return validation

