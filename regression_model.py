__author__ = 'kevin palm'
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from scipy import stats

#Create list of dependent and independents
df = pd.read_csv(r'/home/kevin/Desktop/Udacity/Project_1/improved_dataset/improved_dataset/turnstile_weather_v2.csv')
dependent = df['ENTRIESn_hourly']
independent = df[['tempi', 'hour', 'weekday', 'fog', 'precipi', 'wspdi', 'pressurei']]
dummy_units = pd.get_dummies(df['UNIT'], prefix='unit')
independent = independent.join(dummy_units)

#Normalize independents if desired
#independent = (independent - independent.mean()) / independent.std()

#generate predictions using multiple regression
independent = sm.add_constant(independent)
predictor = sm.OLS(dependent, independent).fit()

df['predictions'] = pd.Series(predictor.predict(independent), name = 'predictions')

#Change negative predictions to zero
df.loc[df['predictions']<0,'predictions'] = 0
predictions = df['predictions']

#Compute R2
r_squared = 1 - np.square(dependent - predictions).sum() / np.square(dependent - np.mean(dependent)).sum()

#Show results and graph residuals
print(predictor.summary())
residuals = pd.Series((dependent - predictions), name = 'residuals')
print(r_squared)
plt.figure()
residuals.hist()
plt.show()