import logging as log

import numpy as np
# import logging
import logging as log, traceback
from cassandra.cluster import Cluster
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
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
workflowType = None
workflowId = None
ipaddressMap = None
workflowspec = None
lgr_analytics = {}
log.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


@app.route("/lgr/ipwfspec", methods=['POST'])
def readIPs():
    global workflowdata, client, ipaddressMap, workflowspec, workflowType

    workflowdata = request.get_json()
    client = workflowdata["client_name"]
    workflowType = workflowdata["workflow"]
    workflowspec = workflowdata["workflow_specification"]
    print("*** workflow specification*** ", workflowspec)

    ipaddressMap = workflowdata["ips"]
    ipaddressMap[workflowType + "#" + client] = ipaddressMap["4"]
    id = workflowType + "#" + client
    return jsonify(lgr_analytics), 200


# Model Training at Launch
def modeltrain(x_train, y_train):
    global model, lgr_analytics
    print("model training started *************************")
    lg_clf = LogisticRegression(class_weight='balanced', solver='liblinear', C=0.1, max_iter=10000)
    model = lg_clf.fit(x_train, y_train)
    lgr_analytics["end_time"] = time.time()
    print("model training complete*********************")


# Model Training
@app.route("/lgr/train", methods=['POST'])
def trainModel():
    # first read data from manager
    global workflowdata, client, model, lgr_analytics, ipaddressMap, workflowType, workflowspec
    lgr_analytics["start_time"] = time.time()

    workflowdata = request.get_json()
    client = workflowdata["client_name"]
    workflowType = workflowdata["workflow"]
    workflowspec = workflowdata["workflow_specification"]
    newipadd = workflowdata["ips"]
    ipaddressMap[workflowType+"#"+client] = newipadd["4"]

    # then read training data from database
    log.info("calling function to read training data from database *************************")
    print("calling function to read training data from database *************************")

    x_train, y_train, resultt = readTrainingData(client,workflowType)

    log.info("model training started *************************")
    print("model training started *************************")
    # then train the model
    log.info("model training started *************************")
    print("model training started *************************")
    lg_clf = LogisticRegression(class_weight='balanced', solver='liblinear', C=0.1, max_iter=10000)
    model = lg_clf.fit(x_train, y_train)
    lgr_analytics["end_time"] = time.time()
    log.info("model training complete*********************")
    print("model training complete*********************")
    # return Response(lgr_analytics, status=200, mimetype='application/json')
    if resultt == -1:
        return jsonify(lgr_analytics), 400
    else:
        return jsonify(lgr_analytics), 200


def pandas_factory(colnames, rows):
    return pd.DataFrame(rows, columns=colnames)


def readTrainingData(tablename, workflow):
    global client, workflowType, lgr_analytics
    lgr_analytics["start_time"] = time.time()
    client = tablename
    workflowType = workflow
    clientTable = workflowType+"_"+client
    cluster = Cluster(['10.176.67.91'])  # Cluster(['0.0.0.0'], port=9042) #Cluster(['10.176.67.91'])
    log.info("setting DB keyspace . . .")
    session = cluster.connect('ccproj_db', wait_for_all_pools=True)
    session.row_factory = pandas_factory
    session.execute('USE ccproj_db')
    result = 0
    qry = "SELECT COUNT(*) FROM " + clientTable + ";"
    # qry = "SELECT COUNT(*) FROM " + tablename + ";"
    try:
        stat = session.prepare(qry)
        x = session.execute(stat)
        for row in x:
            result = row.count
    except:
        result = -1
        log.info("Table does not exists in Cassandra, shutting down Logistic regression component")

    rows = session.execute('SELECT * FROM ' + clientTable)
    # rows = session.execute('SELECT * FROM ' + tablename)
    df = rows._current_rows

    if 'emp_id' in df.columns:
        print(df)
        print(df.dtypes)
        print("columns ", df['checkin_datetime'])
        data = encodeEmployee(df)
        x = data.drop(['uu_id', 'emp_id', 'duration'], axis=1).to_numpy()
        y = data['duration'].to_numpy()
    else:
        print("hospital data")
        data = encodeHospital(df)
        print(df)
        print(df.dtypes)
        print(df.iloc[0].head(1))
        print("checkin_datetime column hospital ", df['checkin_datetime'])
        print("num_in_icu column hospital ", df['num_in_icu'])
        x = data.drop(['uu_id', 'hadm_id', 'num_in_icu'], axis=1).to_numpy()
        y = data['num_in_icu'].to_numpy()
        y = y.astype('int')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.50, random_state=42)
    session.shutdown()
    return x_train, y_train, result


def encodeEmployee(df, onehot=True):
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


def encodeHospital(df, onehot=True):
    if onehot:
        # timeencodeDict = {"8:00": 0, "8:30": 1, "9:00": 2,
        #                   "9:30": 3, "10:00": 4, "10:30": 5, "11:00": 6,
        #                   "11:30": 7, "12:00": 8, "12:30": 9, "13:00": 10,
        #                   "13:30": 11, "14:00": 12, "14:30": 13, "15:00": 14,
        #                   "15:30": 15, "16:00": 16, "16:30": 17,
        #                   "17:00": 18, "17:30": 19, "18:00": 20,
        #                   "18:30": 21, "19:00": 22, "19:30": 23, '20:00': 24}
        timeencodeDict = {"00:00" : 0, "00:30" : 1, "01:00" : 2, "01:30" : 3,
                         "02:00" : 4, "02:30" : 5, "03:00" : 6, "03:30" : 7,
                         "04:00" : 8, "04:30"  : 9, "05:00" : 10, "05:30" : 11,
                         "06:00" : 12, "06:30" : 13, "07:00" : 13, "07:30" : 14,
                         "08:00" : 15, "08:30" : 16, "09:00" : 17, "09:30" : 18,
                         "10:00": 19, "10:30" : 20, "11:00": 21,
                         "11:30" : 21, "12:00": 22, "12:30" : 23}
        dayencodeDict = {
            'MON': 0, "TUE": 1, 'WED': 2, 'THU': 3, "FRI": 4, "SAT": 5, "SUN": 6
        }
        for i, row in df.iterrows():
            df.at[i, 'checkin_datetime'] = timeencodeDict[row['checkin_datetime']]
            df.at[i, 'day_of_week'] = dayencodeDict[row['day_of_week']]
    return df


@app.route("/lgr/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        log.info("***********prediction started*************")
        global lgr_analytics, workflowdata
        lgr_analytics["start_time"] = time.time()

        workflowdata = request.get_json()
        data = workflowdata['data']

        if "emp_id" in data:
            featureListEmployee = ['dept_type', 'gender', 'race', 'day_of_week', 'checkin_datetime']
            newdata = {}
            for i in featureListEmployee:
                newdata[i] = data[i]
            df = pd.json_normalize(newdata)
            predict_data = encodeEmployee(df)
        else:
            featureListHospital = ['hospital_expire_flag', 'insurance', 'duration', 'amount', 'rate', 'total_items',
                                   'value', 'dilution_value', 'abnormal_count', 'item_distinct_abnormal', 'checkin_datetime', 'day_of_week']
            newdata = {}
            for i in featureListHospital:
                newdata[i] = data[i]
            df = pd.json_normalize(newdata)
            predict_data = encodeHospital(df)

        log.info("start prediction*******************************************")
        y_pred = model.predict(predict_data)
        y_pred = list(y_pred)

        timedcodeDict = {0: "8:00", 1: "8:30", 2: "9:00",
                         3: "9:30", 18: "10:00", 19: "10:30", 20: "11:00",
                         7: "11:30", 8: "12:00", 9: "12:30", 10: "13:00",
                         11: "13:30", 12: "14:00", 13: "14:30", 14: "15:00",
                         15: "15:30", 16: "16:00", 17: "16:30",
                         4: "17:00", 5: "17:30", 6: "18:00",
                         21: "18:30", 22: "19:00", 23: "19:30", 24: "20:00"}

        lgr_analytics["end_time"] = time.time()
        if "emp_id" in data:
            lgr_analytics["prediction_LR"] = timedcodeDict[int(y_pred[0])]
        else:
            log.info("The hospital prediction is")
            log.info(int(y_pred[0]))
            lgr_analytics["prediction_LR"] = int(y_pred[0])
        log.info("********** calling nextFire() in predict **********")
        nextFire()
        # return Response(lgr_analytics, status=200, mimetype='application/json')
        return jsonify(lgr_analytics), 200


def nextFire():
    global client, workflowType
    print("*******in nextFire ***********")
    # workflowspec = workflowdata["workflow_specification"]
    # log.info("*** workflow specification*** "+ workflowspec)
    client = workflowdata["client_name"]
    workflowType = workflowdata["workflow"]

    # nextComponent = wfspec[2][0]
    for i, lst in enumerate(workflowspec):
        for j, component in enumerate(lst):
            if component == "2":
                indexLR = i
    # indexLR = wfspec.index(2)
    if indexLR+1 < len(workflowspec):
        nextComponent = workflowspec[indexLR + 1][0]
    else:
        nextComponent = "4"
    workflowdata["analytics"].append(lgr_analytics)

    log.info(workflowdata)

    if nextComponent == "3":  # svm
        nextIPport = ipaddressMap[nextComponent]
        ipp = nextIPport.split(":")
        ipaddress = ipp[0]
        port = ipp[1]
        log.info("making request to SVM Component")
        log.info(ipp)
        r1 = requests.post(url="http://" + ipaddress + ":" + port + "/svm/predict",
                           headers={'content-type': 'application/json'}, json=workflowdata, timeout = 60)
    elif nextComponent == "4":
        log.info(nextComponent)
        nextIPport = ipaddressMap[workflowType+"#"+client]
        ipp = nextIPport.split(":")
        ipaddress = ipp[0]
        port = ipp[1]
        log.info("making request to Analytics")
        log.info(ipp)
        r1 = requests.post(url="http://" + ipaddress + ":" + port + "/put_result",
                           headers={'content-type': 'application/json'}, json=workflowdata, timeout = 60)
    else:
        return jsonify(success=False)


if __name__ == '__main__':
    workflowtype = os.environ['workflow']
    table = ""
    if os.environ['client_name'] is not None:
        table = os.environ['client_name']
    else:
        log.error("Include variable client_name in docker swarm command")
        sys.exit(1)

    log.info("data read from Database ***********")
    rs = 0
    x_data, y_data, rs = readTrainingData(table,workflowtype)
    if rs == -1:
        sys.exit(1)
    log.info("start training model on container launch ****************")
    modeltrain(x_data, y_data)
    log.info("**** start listening ****")
    app.run(debug=True, host="0.0.0.0", port=50)

