import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
print(df.head())
from sklearn.linear_model import LinearRegression
lm = LinearRegression() # Model
lm

X = df[['highway-mpg']]
Y = df['price']

lm.fit(X,Y) # we train the model

filename = 'trained_model3.sav' # This file can go to the Web application
f = open(filename, 'wb')
pickle.dump(lm, f)
f.close()