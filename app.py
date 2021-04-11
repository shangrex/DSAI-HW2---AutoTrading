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

def buildManyToOneModel(shape):
    model = Sequential()
    model.add(LSTM(50, input_length=shape[1], input_dim=shape[2], return_sequences=False))
    model.add(Dense(units = 10))
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

def manipulate(stock, predict_gap):
    action = 0
    if stock == 0:
        #predict will get higher
        if predict_gap > 0:
            action = 1
            stock = 1
        elif predict_gap < 0:
            action = -1
            stock = -1
        else:
            action = 0
            stock = 0
    elif stock == 1:
        if predict_gap > 0:
            action = 0
            stock = 1
        elif predict_gap < 0:
            action = -1
            stock = 0
        else:
            action = 0
            stock = 1
    elif stock == -1:
        if predict_gap > 0:
            action = 1
            stock = 0
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
    
    train = pd.read_csv('training.csv', names=['open', 'high', 'low', 'close'])


    #training
    x_train, y_train = create_data(train,4, 1)
    regressor5 = buildManyToOneModel(x_train.shape)
    callback = EarlyStopping(monitor="loss", patience=400, verbose=1, mode="auto")
    regressor5.fit(x_train, y_train, epochs = 10000, callbacks=[callback])

    # The following part is an example.
    # You can modify it at will.
    # training_data = load_data(args.training)
    # trader = Trader()
    # trader.train(training_data)
    
    test = pd.read_csv('testing.csv', names=['open', 'high', 'low', 'close'])
    test_length = len(test)
    print(test_length)
    all_data = train.append(test)
    x_test, y_test = create_data(all_data, 4, 1)
    x_test = x_test[-test_length:]
    y_test = y_test[-test_length:]
    # print(x_test)
    # print(y_test)
    actions = []
    stock = 0

    result = np.array([])
    result = np.append(result, lstm_model(x_test, y_test, regressor5))
    output_file = open(args.output, 'w')

    for i in range(test_length-1):
        action, stock = manipulate(stock, result[i+1]-result[i])
        actions.append(action)
        output_file.write(str(action))
        if i < test_length-2:
            output_file.write('\n')
    
    output_file.close()
    # with open(args.output, 'w') as output_file:
    #     for row in testing_data:
    #         # We will perform your action as the open price in the next day.
    #         action = trader.predict_action(row)
    #         output_file.write(action)

    #         # this is your option, you can leave it empty.
    #         trader.re_training(i)

    regressor5.save('4')
