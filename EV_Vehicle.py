import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("Electric_Vehicle_Population_By_County.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nData info:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())


Q1 = df['Percent Electric Vehicles'].quantile(0.25)
Q3 = df['Percent Electric Vehicles'].quantile(0.75)
IQR = Q3 - Q1


lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print('lower_bound:', lower_bound)
print('upper_bound:', upper_bound)


outliers = df[(df['Percent Electric Vehicles'] < lower_bound) | (df['Percent Electric Vehicles'] > upper_bound)]
print("Number of outliers in 'Percent Electric Vehicles':", outliers.shape[0])


df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

df = df[df['Date'].notnull()]

df = df[df['Electric Vehicle (EV) Total'].notnull()]

df['County'] = df['County'].fillna('Unknown')
df['State'] = df['State'].fillna('Unknown')

print("Missing after fill:")
print(df[['County', 'State']].isnull().sum())

print("\nFirst 5 rows after preprocessing:")
print(df.head())

df['Percent Electric Vehicles'] = np.where(df['Percent Electric Vehicles'] > upper_bound, upper_bound,
                                 np.where(df['Percent Electric Vehicles'] < lower_bound, lower_bound, df['Percent Electric Vehicles']))

outliers = df[(df['Percent Electric Vehicles'] < lower_bound) | (df['Percent Electric Vehicles'] > upper_bound)]
print("Number of outliers in 'Percent Electric Vehicles' after capping:", outliers.shape[0])
