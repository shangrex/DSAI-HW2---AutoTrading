import pandas as pd
import numpy as np
import matplotlib as plt
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow import keras
import math
from keras.callbacks import EarlyStopping
seed_value = 1234567899
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.random.set_seed(seed_value)
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

action = 0
stock = 0  # 持有
tmp1 = -1 # 買進價
tmp2 = -1 # 賣空價
trend = 0 # 股票走勢，大於0代表連漲 n 天，小於 0 代表連跌

def buildManyToOneModel(shape):
    model = Sequential()
    model.add(LSTM(50, input_length=shape[1], input_dim=shape[2], return_sequences=False))
    model.add(Dense(units = 100))
    model.add(Dropout(0.2))
    model.add(Dense(units = 10))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    return model

def create_data(data, past = 7, future = 7):
    x_train = []
    y_train = []
    for i in range(0, len(data)-past-future+1):
        t = data.iloc[i:i+past][['open', 'high', 'low', 'close']]
        y = data['open'].iloc[i+past:i+past+future]

        x_train.append(t)
        y_train.append(y)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train, y_train

def create_last_data(data, past = 7, future = 7):
    x_train = []
    t = data.iloc[-past:][['open', 'high', 'low', 'close']]

    x_train.append(t)

    x_train = np.array(x_train)
    return x_train


def manipulation(stock, predict_gap, predict_price, tmp1, tmp2, trend):
    action = 0
    #tomorrow will be higher
    if predict_gap > 0:
        
        if trend <= 0:
            trend = 1
        else:
            trend = trend + 1
        
        if trend <= 1 and stock == 0: # 第一天漲
            action = 1
            stock = 1
            tmp1 = predict_price # 紀錄買價
        elif stock == 1 and predict_price > tmp1: # 明天價格比買價高
            action = -1
            stock = 0
            tmp1 = -1
        elif stock == 0 and trend >= 2:
            action = -1  # 賣空
            stock = -1
            tmp2 = predict_price
    #tommor will be lower
    if predict_gap <= 0:
    
        if trend >= 0:
            trend = -1
        else:
            trend = trend - 1
        
        if trend >= -1 and stock == 0: # 第一天跌
            action = -1  # 賣空
            stock = -1
            tmp2 = predict_price
        elif stock == -1 and predict_price < tmp2:
            action = 1   # 預測價格比賣空價低，買進
            stock = 0
            tmp2 = -1
        elif stock <= 0 and trend <= -2: # 連跌大於等於 2 天且未持股
            action = 1
            if stock == 0:
                tmp1 = predict_price
                stock = 1
            elif stock == -1:
                tmp2 = -1
                stock = 0
        
    if predict_gap == 0:
        action = 0
    return (action, stock, trend)


def manipulate(stock, predict_gap):
    action = 0
    if stock == 0:
        #predict will get higher
        if predict_gap > 0:
            action = 1
            stock = 1
        #predict lower and short one unit
        elif predict_gap < 0:
            action = -1
            stock = -1
        else:
            action = 0
            stock = 0
    elif stock == 1:
        #predict higher but nothing change
        if predict_gap > 0:
            action = 0
            stock = 1
        #predict lower and short one unit
        elif predict_gap < 0:
            action = -1
            stock = 0
        else:
            action = 0
            stock = 1
    elif stock == -1:
        #predict higher and buy one unit back
        if predict_gap > 0:
            action = 1
            stock = 0
        #predict lower and still short 
        elif predict_gap < 0:
            action = 0
            stock = -1
        else:
            action = 0
            stock = -1
    else:
        print("manipulate error")
        return None
    return (action, stock)

def lstm_model(x, y, model):
    err = 0
    result = []
    predicts = []
    for i,j in zip(x, y):
        i = i.reshape(1, i.shape[0], -1)
        predict = model.predict(i)
        predict = predict[0][0]
        j = j[0]
        predicts.append(predict)
        err = (predict-j)**2
        result.append(err)
        # print(math.sqrt(err))
        # print(j-predict)
    print("look before", x.shape[1], " avg loss ", math.sqrt(sum(result)/len(result)))
    return predicts

if __name__ == '__main__':
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    
    train = pd.read_csv(args.training, names=['open', 'high', 'low', 'close'])


    #training
    x_train, y_train = create_data(train,4, 1)
    regressor5 = buildManyToOneModel(x_train.shape)
    callback = EarlyStopping(monitor="loss", patience=400, verbose=1, mode="auto")
    regressor5.fit(x_train, y_train, epochs = 10000, callbacks=[callback])
    # regressor5 = keras.models.load_model('4')

    #testing
    test = pd.read_csv(args.testing, names=['open', 'high', 'low', 'close'])
    test_length = len(test)
    all_data = train.append(test)
    x_test, y_test = create_data(all_data, 4, 1)
    x_test = x_test[-test_length+1:]
    y_test = y_test[-test_length+1:]
    print(x_test.shape)
    print(test['open'].tolist())
    actions = []
    past = create_last_data(all_data, 4, 1)
    print(past)
    
    result = np.array([])
    result = np.append(result, lstm_model(x_test, y_test, regressor5))
    print(result)
    output_file = open(args.output, 'w')
    last_day = test['open'].tolist()[0]
    

    for i in range(test_length-1):
        #依上次做動作更新值
        if action == 1:
            tmp1 = test['open'].tolist()[i]
        elif action == -1:
            tmp2 = test['open'].tolist()[i]
        
        tmp = x_test[i].reshape(1, 4, -1)
        print(tmp)
        tmp = regressor5.predict(tmp)
        tmp = tmp[0][0]
        print(tmp)
        print(test['open'].tolist()[i+1])
        action, stock, trend = manipulation(stock, tmp-test['open'].tolist()[i], tmp, tmp1, tmp2, trend)
        actions.append(action)
        output_file.write(str(action))
        if i < test_length-2:
            output_file.write('\n')
    output_file.close()

    # regressor5.save('4')
