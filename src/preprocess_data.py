
import warnings
import pandas as pd
import numpy as np
import skimpy as sm

FilePath_raw_data = './data/raw_data.csv'
FilePath_validation = './data/validation.csv'

warnings.simplefilter(action="ignore", category=FutureWarning)

raw_data_DataFrame = pd.read_csv(FilePath_raw_data)