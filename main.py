import logging as log

import numpy as np
from cassandra.cluster import Cluster
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from timeParser import timeParser

app = Flask(__name__)
CORS(app)
model = None


# final_table_without_na.describe()
# final_table_without_hadm = final_table_without_na.drop(columns = ['hadm_id'])
# Splitting the Train and Test Dataset


def pandas_factory(colnames, rows):
    return pd.DataFrame(rows, columns=colnames)

# get training data from Cassandra
def readTrainingData():
    cluster = Cluster(['10.176.67.91'])  # Cluster(['0.0.0.0'], port=9042) #Cluster(['10.176.67.91'])
    log.info("setting DB keyspace . . .")
    session = cluster.connect('ccproj_db', wait_for_all_pools=True)
    session.row_factory = pandas_factory
    session.execute('USE ccproj_db')
    # condition to check which workflow
    rows = session.execute('SELECT * FROM employee')
    df = rows._current_rows
    print ("columns ",df.columns)
    # data = getData(df)
    x = []
    y = []
    for row in rows:
        print(row.uu_id, row.emp_id, row.dept_type)
        x.append([row.emp_id, row.dept_type, row.race, row.day_of_week])
        y.append(row.checkin_datetime)
    # print ("printing x ", x)
    # print ("printing y", y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.50, random_state=42)
    session.shutdown()
    return x_train, y_train

def getData(x_train, onehot=True):

        if onehot:
            times_encoder = OneHotEncoder()
            transformed_time = times_encoder.transform(
                x_train[4].to_numpy().reshape(-1, 1))
            times_df = pd.DataFrame(
                transformed_time, columns=times_encoder.get_feature_names())
            dayOfWeek_encoder = OneHotEncoder()
            transformed_dayOfWeek = dayOfWeek_encoder.transform(
                x_train[3].to_numpy().reshape(-1, 1))
            dayOfWeek_df = pd.DataFrame(
                transformed_dayOfWeek, columns=dayOfWeek_encoder.get_feature_names())
            x_train = pd.concat([times_df, dayOfWeek_df, x_train], axis=1).drop(
                [3, 4], axis=1)
            dept_encoder = OneHotEncoder()

        return x_train

# Model Training
def trainModel(x_train, y_train):
    lg_clf = LogisticRegression(class_weight='balanced', solver='liblinear', C=0.1, max_iter=10000)
    model = lg_clf.fit(x_train, y_train)
    return model


@app.route("/app/getPredictionLR", methods=['POST'])
def predict():
    if request.method == 'POST':
        clientRequest = request.get_json()
        y_pred = np.array2string(model.predict(clientRequest))
        # y_pred = model.predict(test_data)
        return jsonify(y_pred)


if __name__ == '__main__':
    # get the training data from Cassandra
    x_data, y_data = readTrainingData()
    # train the model
    modelTrained = trainModel(x_data, y_data)
    # read the prediction data
    app.run(debug=True)
