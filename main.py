import logging as log

import numpy as np
import logging
from cassandra.cluster import Cluster
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from logging.handlers import TimedRotatingFileHandler
import time
import requests


app = Flask(__name__)
CORS(app)
workflowdata = None
model = None
client = None
workflowId = None
lgr_startTime = None
logger = logging.getLogger('logistic_regression')

# final_table_without_na.describe()
# final_table_without_hadm = final_table_without_na.drop(columns = ['hadm_id'])
# Splitting the Train and Test Dataset


# Model Training
@app.route("/lgr/train", methods=['POST'])
def trainModel():
    # first read data from manager
    global workflowdata, client, workflowId, model, lgr_startTime
    lgr_startTime = time.process_time()
    workflowdata = request.get_json()
    client = workflowdata["client_name"]
    workflowId = workflowdata["workflow_id"]

    # then read training data from database
    logger.info(workflowId,"calling function to read training data from database *************************")
    x_train, y_train = readTrainingData(client)
    logger.info("model training started *************************")
    # then train the model
    logger.info("model training started *************************")
    lg_clf = LogisticRegression(class_weight='balanced', solver='liblinear', C=0.1, max_iter=10000)
    model = lg_clf.fit(x_train, y_train)
    print("model training complete*********************")



def pandas_factory(colnames, rows):
    return pd.DataFrame(rows, columns=colnames)


# call this from training
def readTrainingData(tablename):
    cluster = Cluster(['10.176.67.91'])  # Cluster(['0.0.0.0'], port=9042) #Cluster(['10.176.67.91'])
    log.info("setting DB keyspace . . .")
    session = cluster.connect('ccproj_db', wait_for_all_pools=True)
    session.row_factory = pandas_factory
    session.execute('USE ccproj_db')

    rows = session.execute('SELECT * FROM ' + tablename)
    df = rows._current_rows

    if df['uu_id']:
        print("columns ", df['checkin_datetime'])
        data = getData(df)
        x = data.drop(['duration', 'uu_id'], axis=1).to_numpy()
        y = data['duration'].to_numpy()
    else:
        x = df.drop(['hadm_id'], axis=1).to_numpy()
        y = df['total_time_icu'].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.50, random_state=42)
    session.shutdown()
    return x_train, y_train


def getData(df, onehot=True):
    if onehot:
        timeencodeDict = {"8:00": 0, "8:30": 1, "9:00": 2,
                          "9:30": 3, "10:00": 4, "10:30": 5, "11:00": 6,
                          "11:30": 7, "12:00": 8, "12:30": 9, "13:00": 10,
                          "13:30": 11, "14:00": 12, "14:30": 13, "15:00": 14,
                          "15:30": 15, "16:00": 16, "16:30": 17,
                          "17:00": 18, "17:30": 19, "18:00": 20,
                          "18:30": 21, "19:00": 22, "19:30": 23, '20:00': 24}
        dayencodeDict = {
            'MON': 0, "TUE": 1, 'WED': 2, 'THU': 3, "FRI": 4
        }
        genderencodeDict = {
            'male': 0, "female": 1
        }
        for i, row in df.iterrows():
            df.at[i, 'checkin_datetime'] = timeencodeDict[row['checkin_datetime']]
            df.at[i, 'day_of_week'] = dayencodeDict[row['day_of_week']]
            df.at[i, 'gender'] = genderencodeDict[row['gender']]
    return df


@app.route("/lgr/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # predict time
        clientRequest = request.get_json()
        data = clientRequest['data']
        del data['time']
        # remove id for prediction
        logger.info("request ", data)
        df = pd.json_normalize(data)
        logger.info("request columns ", df.columns)
        # encoding only if employee workflow otherwise no
        predict_data = getData(df)
        logger.info("start prediction*******************************************")
        y_pred = model.predict(predict_data)
        y_pred = list(y_pred)
        timedcodeDict = {0: "8:00", 1: "8:30", 2: "9:00",
                         3: "9:30", 4: "10:00", 5: "10:30", 6: "11:00",
                         7: "11:30", 8: "12:00", 9: "12:30", 10: "13:00",
                         11: "13:30", 12: "14:00", 13: "14:30", 14: "15:00",
                         15: "15:30", 16: "16:00", 17: "16:30",
                         18: "17:00", 19: "17:30", 20: "18:00",
                         21: "18:30", 22: "19:00", 23: "19:30", 24: "20:00"}
        logger.info("sending the response back **************************")
        lgr_endTime = time.process_time() - lgr_startTime
        nextFire()
        return timedcodeDict[int(y_pred[0])]


def nextFire(lgr_details):
    wfspec = workflowdata["workflow_specification"]
    ipMap = workflowdata["ips"]
    nextComponent = wfspec[2][0]

    nextIP = ipMap[nextComponent]
    # add lgr entries in the json
    # and call that component with its corresponding ip and call name along with the json
    # append to the end
    if nextComponent == 3: #svm
        r1 = requests.post(url="http://" + nextIP + ":50/app/getPredictionLR",
                       headers={'content-type': 'application/json'}, json=content)
    else:
        r1 = requests.post(url="http://" + nextIP + ":50/app/getPredictionLR",
                           headers={'content-type': 'application/json'}, json=content)


if __name__ == '__main__':
    # get the training data from Cassandra
    #read arguments
    fh = TimedRotatingFileHandler('logistic_regression',  when='midnight')
    fh.suffix = '%Y_%m_%d.log'
    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(lineno)04d | %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.WARNING)

    logger.info("data read from Database ***********")
    x_data, y_data = readTrainingData()
    # train the model
    logger.info("start training model on container launch ****************")
    trainModel(x_data, y_data)
    logger.info("**** start listening ****")
    app.run(debug=True, host="0.0.0.0", port=50)


    #return error and message flask
