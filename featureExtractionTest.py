import pandas as pd
import numpy as np 
from sklearn import preprocessing
import sys
import os
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
import sklearn.cross_validation as cv
from scipy import stats

import prepareb


print("State Values are being read from file!!")
stateFile  = pd.read_csv('input/states.csv')

print("Train Values are being read from file!!")
trainFile  = pd.read_csv('input/train.csv')


print("Trends Values are being read from file!!")
trendFile  = pd.read_csv('input/trends.csv')

dmTrendFile = pd.read_csv('input/dmTrends.csv')

storeStates = prepareb.parseStateData(stateFile)


print("Train Values are being read from file!!")
testFile  = pd.read_csv('input/test.csv')

print("Store Values are being read from file!!")
storeFile  = pd.read_csv('input/store.csv')


medianList1 = prepareb.getMedians(trainFile,1) 
medianList2 = prepareb.getMedians(trainFile,2) 
medianList3 = prepareb.getMedians(trainFile,3) 

saturdayRatios, sundayRatios = prepareb.calculateWeekendRatios(medianList1,medianList2,medianList3);
storeFeats = prepareb.parseStoreInfo(storeFile)



testFile.drop('Id', axis=1,  inplace=True)
trainFile.drop('Customers', axis=1,  inplace=True)


trainFile = pd.concat([testFile,trainFile], ignore_index=True)


trainFeats,trainRatios = prepareb.parseTrainData(trainFile,storeFeats,medianList1,saturdayRatios, sundayRatios,storeStates,trendFile,dmTrendFile)


allTrainSales = trainFile.Sales.values
actuals = trainFile.Sales.values

allTrainSales = allTrainSales.astype(np.float32, copy=False)

for i in range(0,len(allTrainSales)):
    allTrainSales[i] = allTrainSales[i]*trainRatios[i]


trainId  = np.array(range(0,len(actuals)))

trainDataDic = {}
trainDataDic["ID"] = trainId

u = 0

for i in range(0,trainFeats.shape[1]):
    featureKey =  str("feat")+str(u)
    
    trainDataDic[featureKey] = trainFeats[:,i]
    
    u = u + 1

    
trainDataDic["target"]  = allTrainSales
trainDataDic["actuals"] = actuals
trainDataDic["ratios"]  = trainRatios[:,0]


print("Train Values are being written into file!!")

df = pd.DataFrame(trainDataDic).set_index('ID')
df.to_csv('input/trainFeatsTestSub.csv', header=True, index = True, sep = ',', float_format='%.8f')






