'''
We use akshare to download stock data
'''
#!pip install akshare

import akshare as ak
## We first collect a list of stocks
#stockList = ak.stock_zh_a_spot()
#stockList.to_csv("stock_list.csv")

# We train on "千股千评" from eastmoney.com

'''
To download the data, please de-comment the following lines using ctrl + / or command + /
'''
## Data collection
training_data = ak.stock_comment_em()
## Also, we refine our data to make it price-independent on the level of same scale
training_data["主力成本"] = training_data["主力成本"] / training_data["最新价"]
training_data["市盈率"] = training_data["市盈率"] / 100
training_data["综合得分"] = training_data["综合得分"] / 100
training_data["关注指数"] = training_data["关注指数"] / 100
training_data["换手率"] = training_data["换手率"] / 100
## We rename accordingly
training_data.rename(columns = {"主力成本": "主力成本比", "市盈率": "市盈率/100", "综合得分": "综合得分/100", "关注指数": "关注指数/100", "换手率": "换手率/100"}, inplace = True)
## Also, we drop the date, index, and the current (latest) price
##  also the rankings
training_data.drop(columns = ["交易日", "序号", "最新价", "目前排名", "上升"], inplace = True)

## Save to csv to backup
training_data.to_csv("千股千评.csv", index = False)

'''
Please select the stocks you are going to train on, we will
be using the rest to test our model, and recreate a csv file
that containing all of the traing stocks and name it
'training_data.csv'
'''
## Training
number_of_stocks = 2000
import pandas as pd
training_data = pd.read_csv("千股千评.csv", header=None)
## We only use the top 100 stocks for training
training_data = training_data.drop(index = 0)
training_data = training_data.head(number_of_stocks)
training_data.to_csv("training_data.csv", index = False)

### Decomment this to create a SparkContext
# from pyspark import SparkContext

# sc = SparkContext()

import shutil, os

delta = 0.002
output_model = 'models/output'
if os.path.isdir(output_model):
    shutil.rmtree(output_model) # Remove the previous model to create a new one

## Helper function to parse the row
def parcing(row):
    L = row.split(",")
    if L[2] == '':
        return None
    percentage_change = float(L[2])
    F = L[3:]
    if '' in F:
        return None
    F = list(map(float, F))
    return (percentage_change, F)

def training(partition):
    model = []
    for part in partition:
        print(part)
        if part == None:
            continue
        t = part[0]
        F = part[1]
        ## Initialization
        if model == []:
            model = [0] * (len(F) + 1)
        ## We calculate y_exp - y_true here
        value = model[0]
        for i in range(len(F)):
            value += F[i] * model[i + 1]
        diff = value - t
        model[0] -= delta * diff
        for i in range(len(F)):
            model[i + 1] -= delta * F[i] * diff
    return model

training_data = sc.textFile('training_data.csv')

parsed = training_data.map(parcing).repartition(1)
result = parsed.mapPartitions(training)
result.saveAsTextFile(output_model)
