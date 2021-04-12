---
tags: homework
---

# Stock Preidction

## Description
The sample data is NASDAQ:IBM. The data, called training_data.csv, contains more-than-five-year daily prices, whose line number corresponds to the time sequence. Another data, called testing_data.csv, contains one-year daily prices, which corresponds to the time period next to the final date in training_data.csv.

The action should be one of these three types:
1 → means to “Buy” the stock. If you short 1 unit, you will return to 0 as the open price in the next day. If you did not have any unit, you will have 1 unit as the open price in the next day. “If you already have 1 unit, your code will be terminated due to the invalid status.“

0 → means to “NoAction”. If you have 1-unit now, hold it. If your slot is available, the status continues. If you short 1 unit, the status continues.

-1 → means to “Sell” the stock. If you hold 1 unit, your will return to 0 as the open price in the next day. If you did not have any unit, we will short 1 unit as the open price in the next day. “If you already short 1 unit, your code will be terminated due to the invalid status.“

In the final day, if you hold/short the stock, we will force your slot empty as the close price of the final day in the testing period. Finally, your account will be settled and your profit will be calculated.


You can hold 1 unit at most. But of course, you can consider “sell short”, meaning that you can have “-1 unit”.
So that in any time, your slot status should be:
1 → means you hold 1 unit.
0 → means you don’t hold any unit.
-1 → means you short for 1 unit.

## Method
### data process
* 觀察起伏有一定的週期存在。
* 首先觀察所買的股票，為價值股，可以長期持有，所以在沒有資料個情況下，可以先購買，再來，股票的哲學是不是去預估最高點和最底點，而是觀察趨勢，所以不應即時做出反應，可以慢一點做出反應。
* 目標預估天數是20天，可以看出希望是在短期內做出反應，所以模型在training data的評估數字參考就好，甚至可以在評估時，使用短期評估。

### evaluation
* 實際操作看賺多少
```pyton
def reality(outputs, data):
    earn = 0.0
    stock = 0
    buy = 0
    for i, j in zip(outputs, data):
        if i == 1:
            stock += 1
            buy = j
            earn -= buy
        elif i == -1:
            stock -= 1
            buy = j
            earn += buy
    if stock == 1:
        earn += data[-1]
        stock -= 1
    elif stock == -1:
        earn -= data[-1]
        stock += 1
    return earn
# outputs dimension should same as data
print(reality(outputs, test['open'].tolist()))

```


### Rule bass
* model: 觀察，若n天都跌則short，若兩天都漲則買入，若都沒有則都不動。
```python 
outputs = []
def lookbefore(n, data):
    predicts = [data[0]]
    actions = [1]
    stock = 1
    for i in range(1, n):
        predicts.append(data[i])
        last = data[i]
        actions.append(0)
    trend = 0
    trend_day = 0
    for i in range(n , len(data)):
        #lowering
        if data[i] < last and trend >= 0:
            trend = -1 
            trend_day = 0
        elif data[i] < last:
            trend = -1
            trend_day += 1
        #highering
        elif data[i] > last and trend <= 0:
            trend = 1
            trend_day = 0
        elif data[i] > last:
            trend = 1
            trend_day += 1
        else:
            trend = 0
        last = data[i]
        #should short
        if trend_day >= n and trend < 0:
            predicts.append(-1)
            if stock >= 0:
                actions.append(-1)
                stock -= 1
            else:
                actions.append(0)
        #should buy
        elif trend_day >= n and trend > 0:
            predicts.append(1000000)
            if stock <= 0:
                actions.append(1)
                stock += 1
            else:
                actions.append(0)
        else:
            predicts.append(0)
            actions.append(0)
    actions = actions[:-1]
    return actions
outputs = lookbefore(3, test['open'].tolist())
print(len(outputs))
print(outputs)
print(reality(outputs, test['open'].tolist()))

```
在testing data不賺不賠
```
-0.39999999999997726
look berfore 2 days -0.39999999999997726
-0.8100000000000023
look berfore 3 days -0.8100000000000023
-154.4
look berfore 4 days -0.2300000000000182
-154.4
look berfore 5 days -0.2300000000000182
```
在training data整體上賺大約48
### ML

利用LSTM測試往前看幾天來預測未來
* model 
```python
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
```
* 2
	train
	```
	x_train, y_train = create_data(train,2, 1)
	regressor2 = buildManyToOneModel(x_train.shape)
	# 進行訓練
	regressor2.fit(x_train, y_train, epochs = 10000)
	```
* 3
	train
	```
	x_train, y_train = create_data(train,3, 1)
	regressor3 = buildManyToOneModel(x_train.shape)
	# 進行訓練
	regressor3.fit(x_train, y_train, epochs = 10000)
	```
* 4
	train
	```python 
	x_train, y_train = create_data(train,4, 1)
	regressor4 = buildManyToOneModel(x_train.shape)
	# 進行訓練
	regressor4.fit(x_train, y_train, epochs = 10000)
	```
* 5
	train
	```python 
	x_train, y_train = create_data(train,5, 1)
	regressor5 = buildManyToOneModel(x_train.shape)
	# 進行訓練
	regressor5.fit(x_train, y_train, epochs = 10000)
	```
* rmse loss in testing data
look before 2  avg  2.7006795462062856
look before 3  avg  1.6406683569004972
look before 4  avg  1.0537347771211647
look before 5  avg  1.2732146127677073
rmse loss in training data
look before 2  avg  2.664183938246895
look before 3  avg  2.7776563181195977
look before 4  avg  1.719911879952736
look before 5  avg  1.89430373496287
* evaltion
	在testing data上和training data上的結果
	* 2
	![](https://i.imgur.com/HKYuf5h.png)
	![](https://i.imgur.com/glu60JF.png)
	* 3
	![](https://i.imgur.com/86tj3A0.png)
	![](https://i.imgur.com/i4YES4P.png)
	* 4
	![](https://i.imgur.com/CpuCbsK.png)
	![](https://i.imgur.com/cHQmedk.png)

	* 5
	![](https://i.imgur.com/uJOrbeV.png)
	在trianning 上的結果
	![](https://i.imgur.com/KzzUg83.png)
	![](https://i.imgur.com/HcgcSZ3.png)
	* 6
	![](https://i.imgur.com/bzbBHjd.png)
	* 7
	![](https://i.imgur.com/ylbLlsC.png)

* earlystopping
400
```
look before 2  avg  17.95779577735981
look before 3  avg  1.2024779086183162
look before 4  avg  0.6934668794857886
look before 5  avg  1.6304687513010776
look before 6  avg  2.351419624735438
look before 7  avg  2.664411157927471

```
400
```
earn 2 7.379999999999967
earn 3 5.439999999999998
earn 4 7.129999999999967
earn 5 6.0999999999999375
earn 6 7.379999999999967
earn 7 6.0999999999999375
```
* manipulation 股票操作
若未來股票漲則購買，反之賣出，當然也要考慮現在所擁有的股票數目。
```python
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

```
觀察最後，決定使用往前看4天的預測後面的長遞來做操作。

## difficult point
這次作業再做訓練實驗時，遇到repreduce的問題，因為有的是因為saddle point或是遇到比較奇怪的下降點，下降的速度會非常非常慢訴，所以我使用抄大的epoch數以及early stop去做訓練，便且及大限度的限制repreduce的參數，但還是會不能重現實驗，只能這樣做。