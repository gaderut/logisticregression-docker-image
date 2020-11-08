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
import sys
import os
from flask import Response

app = Flask(__name__)
CORS(app)
workflowdata = None
model = None
client = None
workflowId = None
ipaddressMap = None
logger = logging.getLogger('logistic_regression')
lgr_analytics = {}

@app.route("/lgr/readip", methods=['POST'])
def readIPs():
    # workflow spec from here
    global workflowdata, client, workflowId, ipaddressMap
    workflowdata = request.get_json()
    client = workflowdata["client_name"]
    workflowId = workflowdata["workflow_id"]
    workflowtype = workflowdata["workflow"]
    newip = workflowdata["ips"]
    ipaddressMap[workflowtype + "#" + client] = newip["analytics"]
    return 200


# Model Training at Launch
def modeltrain(x_train, y_train):
    global model
    print("model training started *************************")
    lg_clf = LogisticRegression(class_weight='balanced', solver='liblinear', C=0.1, max_iter=10000)
    model = lg_clf.fit(x_train, y_train)
    print("model training complete*********************")
    # record time
    # return model


# Model Training
@app.route("/lgr/train", methods=['POST'])
def trainModel():
    # first read data from manager
    global workflowdata, client, workflowId, model, lgr_analytics, ipaddressMap
    training_startTime = time.process_time()
    workflowdata = request.get_json()
    client = workflowdata["client_name"]
    workflowId = workflowdata["workflow_id"]
    workflowtype = workflowdata["workflow"]
    ipaddressMap = workflowdata["ips"]
    ipaddressMap[workflowtype+"#"+client] = ipaddressMap["analytics"]
    # then read training data from database
    logger.info(workflowId, "calling function to read training data from database *************************")
    print("calling function to read training data from database *************************")
    x_train, y_train = readTrainingData(client)
    logger.info("model training started *************************")
    print("model training started *************************")
    # then train the model
    logger.info("model training started *************************")
    print("model training started *************************")
    lg_clf = LogisticRegression(class_weight='balanced', solver='liblinear', C=0.1, max_iter=10000)
    model = lg_clf.fit(x_train, y_train)
    training_endTime = time.process_time()
    lgr_analytics["start_time"] = training_startTime
    lgr_analytics["end_time"] = training_endTime
    logger.info("model training complete*********************")
    print("model training complete*********************")
    #change it
    return Response(lgr_analytics, status=200, mimetype='application/json')


def pandas_factory(colnames, rows):
    return pd.DataFrame(rows, columns=colnames)


# def validation(usr_name):
#     result = 0
#     qry = "SELECT COUNT(*) FROM " + usr_name + ";"
#     try:
#         stat = self.session.prepare(qry)
#         x = self.session.execute(stat)
#         for row in x:
#              result = row.count
#     except:
#         result = -1
#         log.info("DataLoader validation: No Table found in Cassandra database.")
#         print("DataLoader validation: No Table found in Cassandra database.")
#     return result

# call this from training
def readTrainingData(tablename):
    cluster = Cluster(['10.176.67.91'])  # Cluster(['0.0.0.0'], port=9042) #Cluster(['10.176.67.91'])
    log.info("setting DB keyspace . . .")
    session = cluster.connect('ccproj_db', wait_for_all_pools=True)
    session.row_factory = pandas_factory
    session.execute('USE ccproj_db')

    rows = session.execute('SELECT * FROM ' + tablename)
    df = rows._current_rows

    if 'emp_id' in df.columns:
        print("columns ", df['checkin_datetime'])
        data = getData(df)
        x = data.drop(['uu_id', 'emp_id', 'duration'], axis=1).to_numpy()
        y = data['duration'].to_numpy()
    else:
        # encoding for hospital
        x = df.drop(['uu_id', 'hadm_id', 'total_time_icu'], axis=1).to_numpy()
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
        global lgr_analytics
        predict_startTime = time.process_time()
        clientRequest = request.get_json()
        data = clientRequest['data']
        del data['id']  # prediction data id
        del data['emp_id']
        del data['time']
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
        predict_endTime = time.process_time() - predict_startTime
        lgr_analytics["start_time"] = predict_startTime
        lgr_analytics["end_time"] = predict_endTime
        lgr_analytics["prediction_LR"] = timedcodeDict[int(y_pred[0])]
        nextFire()
        # return timedcodeDict[int(y_pred[0])]
        return Response(lgr_analytics, status=200, mimetype='application/json')


def nextFire():
    wfspec = workflowdata["workflow_specification"]
    client = workflowdata["client_name"]
    workflowtype = workflowdata["workflow"]
    # nextComponent = wfspec[2][0]
    for i, lst in enumerate(wfspec):
        for j, component in enumerate(lst):
            if component == "2":
                indexLR = i
    # indexLR = wfspec.index(2)
    if indexLR == len(wfspec):
        nextComponent = 4
    else:
        nextComponent = wfspec[indexLR + 1][0]

    workflowdata["analytics"].append(lgr_analytics)

    if nextComponent == "3":  # svm
        nextIPport = ipaddressMap[nextComponent]
        ipp = nextIPport.split(":")
        ipaddress = ipp[0]
        port = ipp[1]
        r1 = requests.post(url="http://" + ipaddress + ":" + port + "/svm/predict",
                           headers={'content-type': 'application/json'}, json=workflowdata)
    elif nextComponent == "4":
        nextIPport = ipaddressMap[workflowtype+"#"+client]
        ipp = nextIPport.split(":")
        ipaddress = ipp[0]
        port = ipp[1]
        r1 = requests.post(url="http://" + ipaddress + ":" + port + "/put_result",
                           headers={'content-type': 'application/json'}, json=workflowdata)
    else:
        return "Error"


if __name__ == '__main__':
    # get the training data from Cassandra
    # read arguments
    workflow = os.environ['workflow']
    if os.environ['client_name'] is not None:
        table = os.environ['client_name']
    else:
        logger.error("Include variable client_name in docker swarm command")
        sys.exit(1)
    fh = TimedRotatingFileHandler('logistic_regression', when='midnight')
    fh.suffix = '%Y_%m_%d.log'
    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(lineno)04d | %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.WARNING)

    logger.info("data read from Database ***********")
    x_data, y_data = readTrainingData(table)
    # train the model
    logger.info("start training model on container launch ****************")
    modeltrain(x_data, y_data)
    logger.info("**** start listening ****")
    app.run(debug=True, host="0.0.0.0", port=50)

    # return error and message flask
