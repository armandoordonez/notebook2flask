import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression


filename = 'trained_model3.sav' # This file can go to the Web application
loaded_model = pickle.load(open(filename, 'rb'))

data = {'highway-mpg':['29']}
X1 = pd.DataFrame (data, columns = ['highway-mpg'])

precio=loaded_model.predict(X1)
rta = "El precio es" + str (precio)
print(rta)

