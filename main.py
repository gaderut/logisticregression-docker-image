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
import category_encoders as ce

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
    print("columns ", df['checkin_datetime'])
    data = getData(df)
    x = data.drop(['duration','uu_id'], axis=1).to_numpy()
    y = data['duration'].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.50, random_state=42)
    session.shutdown()
    return x_train, y_train


def getData(df, onehot=True):
    if onehot:
        data_encoder = ce.OneHotEncoder(cols=["checkin_datetime", "day_of_week", "dept_type", "gender", "race"])
        # times_encoder = times_encoder.fit_transform()
        # if flag == True:
        #     transformed_df = data_encoder.transform(df)
        # else:
        #     transformed_df = data_encoder.fit_transform(df)

        transformed_df = data_encoder.fit_transform(df)
        # transformed_time = times_encoder.fit_transform(df['checkin_datetime'].to_numpy().reshape(-1, 1))
        df = pd.DataFrame(transformed_df, columns=data_encoder.get_feature_names())
        print ("timesss ", df)
        print ("typessss ", type(df))
    return df


# Model Training
def trainModel(x_train, y_train):
    global model
    print("model training started *************************")
    lg_clf = LogisticRegression(class_weight='balanced', solver='liblinear', C=0.1, max_iter=10000)
    model = lg_clf.fit(x_train, y_train)
    print("model training complete*********************")
    # return model


@app.route("/app/getPredictionLR", methods=['POST'])
def predict():
    if request.method == 'POST':
        clientRequest = request.get_json()
        # df = pd.DataFrame([[d['v'] for d in x['c']] for x in clientRequest['rows']],
        #                   columns=[d['label'] for d in clientRequest['cols']])
        print("request ",clientRequest)
        # df = pd.DataFrame.from_dict(clientRequest)
        # df = pd.io.json.json_normalize(clientRequest, 'emp_id', 'dept_type','gender','race','day_of_week','checkin_datetime')
        df = pd.json_normalize(clientRequest)
        print("request columns ",df.columns)
        predict_data = getData(df)
        print("start prediction*******************************************")
        y_pred = np.array2string(model.predict(predict_data))
        print("sending the response back **************************")
        return str(y_pred)


if __name__ == '__main__':
    # get the training data from Cassandra
    print("data read ***********")
    x_data, y_data = readTrainingData()
    # train the model
    print("in main model training to be started****************")
    trainModel(x_data, y_data)
    app.run(debug=True, host="0.0.0.0", port=50)
