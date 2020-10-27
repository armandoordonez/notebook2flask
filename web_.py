import flask
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    filename = 'trained_model.sav' # This file can go to the Web application
    # In the web page we load the file
    loaded_model = pickle.load(open(filename, 'rb'))
    # This information comes from the Web Page
    data = {'highway-mpg':['29']}
    X1 = pd.DataFrame (data, columns = ['highway-mpg'])
    precio=loaded_model.predict(X1)
    rta = "El precio es" + str (precio)
    return rta

app.run()