
import warnings
import pandas as pd
import numpy as np
import skimpy as sm

FilePath_raw_data = './data/raw_data.csv'
FilePath_validation = './data/validation.csv'

warnings.simplefilter(action="ignore", category=FutureWarning)

raw_data_DataFrame = pd.read_csv(FilePath_raw_data)

print(raw_data_DataFrame.head())

print(raw_data_DataFrame.info())

print(f'The data has {raw_data_DataFrame.shape[0]} rows and {raw_data_DataFrame.shape[1]} columns.')

print(sm.skim(raw_data_DataFrame))

missing_values = (
    raw_data_DataFrame.isnull().sum()/len(raw_data_DataFrame)*100
).astype(int)

print(f'Column\t\t\t% missing')
print(f'{"-"}'*35)
print(missing_values)