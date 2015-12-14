import pandas as pd
import numpy as np 
from sklearn import preprocessing
import sys
import os
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
import sklearn.cross_validation as cv
from scipy import stats
from datetime import datetime, date, time, timedelta
import scipy.stats as ss
import calendar
import holidays  

from scipy.stats.stats import pearsonr  


  
def extractLogFeatures(feat,nanValues,N):
    featTemp = feat.copy()
    
    featTemp[pd.isnull(featTemp)] = nanValues

    temp  =  featTemp[featTemp != nanValues]
    
    temp  = np.floor(np.divide(np.log(temp+1),np.log(N)))    
    
    featTemp[featTemp != nanValues] = temp

    return featTemp


def extractStatFeatures(feat,minRange,maxRange,nanReplace,countRefs):
    refCounts = []
    statVals  = [] 
    featTemp = feat.copy()
    total = 0 
    for i in range(0,len(countRefs)):
        countVal =  featTemp[featTemp == countRefs[i]]              
        refCounts.append(len(countVal)) 
        total = total + len(countVal)
    
    
    featTemp[pd.isnull(featTemp)] = nanReplace
    featTemp[featTemp < minRange] = nanReplace
    featTemp[featTemp > maxRange] = nanReplace

    temp = featTemp[featTemp != nanReplace]
    
    meanVal   = np.mean(temp) if len(temp) > 0 else 0
    stdVal    = np.std(temp) if len(temp) > 0 else 0
    maxVal    = np.max(temp) if len(temp) > 0 else 0
    skewVal   = stats.skew(temp) if len(temp) > 0 else 0
    kurtoVal  = stats.kurtosis(temp) if len(temp) > 0 else 0
    
    statVals.append(meanVal)
    statVals.append(stdVal)
    statVals.append(maxVal)
    statVals.append(skewVal)
    statVals.append(kurtoVal)
    statVals.append(0 if stdVal == 0 else meanVal/stdVal)
    statVals.append(0 if stdVal == 0 else maxVal/stdVal)
    statVals.append(0 if stdVal == 0 else skewVal/stdVal)
    statVals.append(0 if stdVal == 0 else kurtoVal/stdVal)
    statVals.append(0 if skewVal == 0 else kurtoVal/skewVal)
    statVals.append(0 if skewVal == 0 else meanVal/skewVal)
    statVals.append(0 if skewVal == 0 else maxVal/skewVal)
    statVals.append(0 if skewVal == 0 else meanVal/kurtoVal)    
    statVals.append(0 if skewVal == 0 else maxVal/kurtoVal)
    statVals.append(len(temp))
    statVals.append(len(temp)*meanVal)    
    
    return featTemp,refCounts,statVals
             
def extractStatFeatures2(feat,minRange,maxRange,nanReplace,countRefs):
    refCounts = []
    statVals  = [] 
    featTemp = feat.copy()
    total = 0 
    for i in range(0,len(countRefs)):
        countVal =  featTemp[featTemp == countRefs[i]]              
        refCounts.append(len(countVal)) 
        total = total + len(countVal)
    
    
    featTemp[pd.isnull(featTemp)] = nanReplace    
    featTemp[featTemp < minRange] = nanReplace
    featTemp[featTemp > maxRange] = nanReplace

    temp = featTemp[featTemp != nanReplace]
    
    meanVal   = np.mean(temp)
    stdVal    = np.std(temp)
    maxVal    = np.max(temp)
    skewVal   = stats.skew(temp)
    kurtoVal  = stats.kurtosis(temp)
    
    statVals.append(meanVal)
    statVals.append(stdVal)
    statVals.append(maxVal)
    statVals.append(skewVal)
    statVals.append(kurtoVal)
    statVals.append(0 if stdVal == 0 else meanVal/stdVal)
    statVals.append(0 if stdVal == 0 else maxVal/stdVal)
    statVals.append(0 if stdVal == 0 else skewVal/stdVal)
    statVals.append(0 if stdVal == 0 else kurtoVal/stdVal)
    statVals.append(0 if skewVal == 0 else kurtoVal/skewVal)
    statVals.append(0 if skewVal == 0 else meanVal/skewVal)
    statVals.append(0 if skewVal == 0 else maxVal/skewVal)
    statVals.append(0 if skewVal == 0 else meanVal/kurtoVal)    
    statVals.append(0 if skewVal == 0 else maxVal/kurtoVal)
    statVals.append(len(temp))
    statVals.append(len(temp)*meanVal)  
            
    return refCounts,statVals
    
    
def extractCategoricalFeatures(df):
    
    dataFrame = df.copy()
    dataFrame[pd.isnull(dataFrame)] = ""
    
    dataFrame = dataFrame.T.to_dict().values()
    
    vec = DictVectorizer(sparse = False)
    return vec.fit_transform(dataFrame), vec


def extractCategoricalFeatures2(df,vector):
    
    dataFrame = df.copy()
    dataFrame[pd.isnull(dataFrame)] = ""
    
    dataFrame = dataFrame.T.to_dict().values()
    
    return vector.transform(dataFrame)
    
    

    

def getDateFeatures(dateFeatList,MISSING):    

    sh1 = dateFeatList.shape[0]
    sh2 = dateFeatList.shape[1]
    
    featYear = np.ones((sh1,sh2),dtype = int)*MISSING
    featMonth = np.ones((sh1,sh2),dtype = int)*MISSING
    featDayMonth = np.ones((sh1,sh2),dtype = int)*MISSING
    featDayNumWeek = np.ones((sh1,sh2),dtype = int)*MISSING
    featDayNumYear = np.ones((sh1,sh2),dtype = int)*MISSING
    featDayTimeNumWeek = np.ones((sh1,sh2),dtype = int)*MISSING
    featDayHour = np.ones((sh1,sh2),dtype = int)*MISSING
    featDayMinute = np.ones((sh1,sh2),dtype = int)*MISSING
    featDayTime = np.ones((sh1,sh2),dtype = int)*MISSING
    featWeek = np.ones((sh1,sh2),dtype = int)*MISSING


    featDiff = np.ones((sh1,(sh2/2)*(sh2-1)),dtype = int)*MISSING
    featAbsDiff = np.ones((sh1,(sh2/2)*(sh2-1)),dtype = int)*MISSING

    for a in range(0,sh2):
        for i in range(0,sh1):
            if  pd.isnull(dateFeatList[i,a]):
                continue
            dt =  datetime.strptime(str(dateFeatList[i,a]), "%d%b%y:%H:%M:%S")
         
        
            tt = dt.timetuple()
        
            featYear[i,a] = tt[0]
            featMonth[i,a] = tt[1]
            featDayMonth[i,a] = tt[2]
            featDayHour[i,a] = tt[3]
            featDayMinute[i,a] = tt[4]
            featDayNumWeek[i,a] = tt[5]
            featDayNumYear[i,a] = tt[6]
            featDayTimeNumWeek[i,a] = featDayNumWeek[i,a]*24*60+featDayHour[i,a]*60+featDayMinute[i,a]
            featDayTime[i,a] = featDayHour[i,a]*60+featDayMinute[i,a]
            featWeek[i,a] = dt.isocalendar()[1]


    colIndex = 0 
    for i in range(0,sh2-1):
        for j in range(i+1,sh2):
            for k in range(0,sh1):

                if  pd.isnull(dateFeatList[k,i]): 
                    continue
        
                if  pd.isnull(dateFeatList[k,j]): 
                    continue
        
                dt1 =  datetime.strptime(str(dateFeatList[k,i]), "%d%b%y:%H:%M:%S")
                dt2 =  datetime.strptime(str(dateFeatList[k,j]), "%d%b%y:%H:%M:%S")
    
                featDiff[k,colIndex] = (dt1 - dt2).days
                featAbsDiff[k,colIndex] = np.abs(featDiff[k,colIndex])
        
            colIndex = colIndex+1
         

    return  np.concatenate((featYear,featMonth,featDayMonth,featDayNumWeek,featDayNumYear,featDayTimeNumWeek,featDayHour,featDayMinute,featDayTime,featWeek,featDiff,featAbsDiff),axis=1)   




def parseStoreInfo(storeFile):

#"Store","StoreType","Assortment","CompetitionDistance","CompetitionOpenSinceMonth","CompetitionOpenSinceYear","Promo2","Promo2SinceWeek","Promo2SinceYear","PromoInterval"
#3,"a","a",14130,12,2006,1,14,2011,"Jan,Apr,Jul,Oct"

    
    storeId = storeFile.Store.values # id   **
    storeType = storeFile.StoreType.values # categorical  **
    storeAssortment = storeFile.Assortment.values # categorical  ** 
    storeCompetitionDistance = storeFile.CompetitionDistance.values # numerical **1
    storeCompetitionOpenSinceMonth = storeFile.CompetitionOpenSinceMonth.values # numerical  **
    storeCompetitionOpenSinceYear = storeFile.CompetitionOpenSinceYear.values # categorical **
    storePromo2 = storeFile.Promo2.values # binary  **
    storePromo2SinceWeek = storeFile.Promo2SinceWeek.values # numerical week number **
    storePromo2SinceYear = storeFile.Promo2SinceYear.values # categorical 2011 **
    storePromoInterval = storeFile.PromoInterval.values #string need to be parsed  "Jan,Apr,Jul,Oct"   
    

    intFeat   = np.zeros((len(storePromoInterval),4),dtype=int)
  

    
    for i in range(0,len(storePromoInterval)):
        val = storePromoInterval[i]
        if pd.isnull(val):
            continue
        
        monthArr = val.split(",")

        first = monthArr[0]
        second =  monthArr[1]
        third =  monthArr[2]
        forth =  monthArr[3]
        
        intFeat[i,0] = setGetMonthInd(first) + 1    
        intFeat[i,1] = setGetMonthInd(second) + 1 
        intFeat[i,2] = setGetMonthInd(third) + 1
        intFeat[i,3] = setGetMonthInd(forth) + 1
        
                    
    storeType[pd.isnull(storeType)] = ""    
    uniqueVals = sorted(list(set(storeType)))
    storeTypeFeat = np.zeros((len(storeType),len(uniqueVals)),dtype= int)
    for i in range(0,len(uniqueVals)):
        feat = np.zeros((len(storeType),1),dtype= int)
        feat[storeType == uniqueVals[i]] = 1 
        storeTypeFeat[:,i] = feat[:,0]    
    
    
    storeAssortment[pd.isnull(storeAssortment)] = ""    
    uniqueVals = sorted(list(set(storeAssortment)))
    storeAssortmentFeat = np.zeros((len(storeAssortment),len(uniqueVals)),dtype= int)
    for i in range(0,len(uniqueVals)):
        feat = np.zeros((len(storeAssortment),1),dtype= int)
        feat[storeAssortment == uniqueVals[i]] = 1 
        storeAssortmentFeat[:,i] = feat[:,0]  
        
    
    storeTypeAssortConcat = storeType+storeAssortment
    uniqueVals = sorted(list(set(storeTypeAssortConcat)))
    storeTypeAssortConcatFeat = np.zeros((len(storeTypeAssortConcat),len(uniqueVals)),dtype= int)
    for i in range(0,len(uniqueVals)):
        feat = np.zeros((len(storeTypeAssortConcat),1),dtype= int)
        feat[storeTypeAssortConcat == uniqueVals[i]] = 1 
        storeTypeAssortConcatFeat[:,i] = feat[:,0]              
            
            
    storeCompetitionDistance[pd.isnull(storeCompetitionDistance)] = -999 
    storeCompetitionOpenSinceMonth[pd.isnull(storeCompetitionOpenSinceMonth)] = -999 
    storeCompetitionOpenSinceYear[pd.isnull(storeCompetitionOpenSinceYear)] = -999     
    storePromo2[pd.isnull(storePromo2)] = -999   
    storePromo2SinceWeek[pd.isnull(storePromo2SinceWeek)] = -999  
    storePromo2SinceYear[pd.isnull(storePromo2SinceYear)] = -999  
    
    storeCompetitionDistance = storeCompetitionDistance.reshape(len(storeId),1)
    storeCompetitionOpenSinceMonth = storeCompetitionOpenSinceMonth.reshape(len(storeId),1)
    storeCompetitionOpenSinceYear = storeCompetitionOpenSinceYear.reshape(len(storeId),1)
    storePromo2 = storePromo2.reshape(len(storeId),1)
    storePromo2SinceWeek = storePromo2SinceWeek.reshape(len(storeId),1)
    storePromo2SinceYear = storePromo2SinceYear.reshape(len(storeId),1)
    
 
    
    storePromo2SinceMonth = np.zeros((len(storeId),1),dtype= int)
    storeCompetitionOpenSinceWeek = np.zeros((len(storeId),1),dtype= int)
    
    dateList1 = []
    dateList2 = []
    
    for i in range(0,len(storeId)):
        skip = False
        
        competitionOpenSinceMonth = int(storeCompetitionOpenSinceMonth[i,0])
        competitionOpenSinceYear = int(storeCompetitionOpenSinceYear[i,0])
        promo2SinceWeek = int(storePromo2SinceWeek[i,0])
        promo2SinceYear = int(storePromo2SinceYear[i,0])
        
        if competitionOpenSinceMonth == -999 or competitionOpenSinceYear == -999:
            dateList1.append(None)
        else:
            dateList1.append(datetime(competitionOpenSinceYear, competitionOpenSinceMonth, 1))
         
         
        if promo2SinceWeek == -999 or promo2SinceYear == -999:
            dateList2.append(None)
        else:
            dateList2.append(datetime.strptime('%d %d 1' % (promo2SinceYear, promo2SinceWeek), '%Y %W %w'))
                    
                    
        if  dateList1[i] != None:
            storeCompetitionOpenSinceWeek[i] = dateList1[i].isocalendar()[1] 
            
        if  dateList2[i] != None:
            tt = dateList2[i].timetuple()
            storePromo2SinceMonth[i] = tt[1] 
        
        
    
    storePromo2SinceWeek = storePromo2SinceWeek.reshape(len(storeId),1)
    storePromo2SinceYear = storePromo2SinceYear.reshape(len(storeId),1)    
    
        
    storeFeats = {}
    storeFeats["distance"]      = storeCompetitionDistance
    storeFeats["comWeek"]       = storeCompetitionOpenSinceWeek
    storeFeats["comMonth"]      = storeCompetitionOpenSinceMonth
    storeFeats["comYear"]       = storeCompetitionOpenSinceYear
    storeFeats["promoState"]    = storePromo2
    storeFeats["promoWeek"]     = storePromo2SinceWeek
    storeFeats["promoMonth"]    = storePromo2SinceMonth
    storeFeats["promoYear"]     = storePromo2SinceYear
    storeFeats["storeType"]     = storeTypeFeat
    storeFeats["assortment"]    = storeAssortmentFeat
    storeFeats["compDate"]      = dateList1
    storeFeats["promoDate"]     = dateList2
    storeFeats["promoInterval"] = intFeat
    storeFeats["typeAssortment"] = storeTypeAssortConcatFeat
                         
    return storeFeats

    
def setGetMonthInd(monthVal):
    
    if monthVal ==  "Jan":
        return 0
    elif monthVal == "Feb":
        return 1     
    elif monthVal ==  "Mar":
        return 2       
    elif monthVal ==  "Apr":
        return 3       
    elif monthVal ==  "May":
        return 4       
    elif monthVal ==  "Jun":
        return 5     
    elif monthVal ==  "Jul":
        return 6       
    elif monthVal ==  "Aug":
        return 7       
    elif monthVal ==  "Sept":
        return 8    
    elif monthVal ==  "Oct":
        return  9    
    elif monthVal ==  "Nov":
        return 10       
    elif monthVal ==  "Dec":
        return 11           
    
    
def parseTrainData(trainFile,storeFeats,medianList,saturdayRatios,sundayRatios,storeStates,trendFile,dmTrendFile):    


    medians0 = getMediansPromo(trainFile, 0)
    medians1 = getMediansPromo(trainFile, 1)
    means0   = getMeansPromo(trainFile, 0)
    means1   =  getMeansPromo(trainFile, 1)

    ext1File = getExt1Data()
    ext2FileMap = getExt2Data() 
    ext3File = getExt3Data()    
    ext4File = getExt4Data() 

    wheatherFileMap = getWheatherData()
    
    startDateList, trendMap = getExternalData(trendFile)
    dmStartDateList, dmTrendMap = getExternalData(dmTrendFile)
    
    print("Feature Extraction Started")
    
    trainStoreId = trainFile.Store.values #key
    
    
    trainDayOfWeek = trainFile.DayOfWeek.values #num(can be categorical)
    trainDate = trainFile.Date.values  #date
    trainOpen = trainFile.Open.values
    trainPromo = trainFile.Promo.values
    trainStateHoliday = trainFile.StateHoliday.values #categorical
    trainSchoolHoliday = trainFile.SchoolHoliday.values #categorical


    trainStateHoliday[trainStateHoliday == 0] = '0'

    trainStateHoliday[pd.isnull(trainStateHoliday)] = ""    
    
    
    uniqueVals = ['a', 'c', '0', 'b']
    
    trainStateHolidayFeat = np.zeros((len(trainStateHoliday),len(uniqueVals)),dtype= int)
    for i in range(0,len(uniqueVals)):
        feat = np.zeros((len(trainStateHoliday),1),dtype= int)
        feat[trainStateHoliday == uniqueVals[i]] = 1 
        trainStateHolidayFeat[:,i] = feat[:,0] 


    trainSchoolHoliday[pd.isnull(trainSchoolHoliday)] = -999    
    uniqueVals = sorted(list(set(trainSchoolHoliday)))
    trainSchoolHolidayFeat = np.zeros((len(trainSchoolHoliday),len(uniqueVals)),dtype= int)
    for i in range(0,len(uniqueVals)):
        feat = np.zeros((len(trainSchoolHoliday),1),dtype= int)
        feat[trainSchoolHoliday == uniqueVals[i]] = 1 
        trainSchoolHolidayFeat[:,i] = feat[:,0] 

    
    uniqueVals = ['RH','BD','SCA','BY','TH','SH','HE','SC','NRW','BR','HM','BE']
    trainStoreState = np.zeros((len(storeStates),len(uniqueVals)),dtype= int)
    for i in range(0,len(uniqueVals)):
        feat = np.zeros((len(storeStates),1),dtype= int)
        feat[storeStates == uniqueVals[i]] = 1 
        trainStoreState[:,i] = feat[:,0] 
        
    
        
    trainDateWeekNum  = np.zeros((len(trainDate),1),dtype= int) 
    trainDateDayNum = np.zeros((len(trainDate),1),dtype= int) 
    trainDateMonthNum = np.zeros((len(trainDate),1),dtype= int) 
    trainDateYearNum  = np.zeros((len(trainDate),1),dtype= int) 
    
    trainDateDiff1 = np.ones((len(trainDate),1),dtype= int)*(-10000) 
    trainDateDiff2 = np.ones((len(trainDate),1),dtype= int)*(-10000) 
    promoIntervalFeat     = np.ones((len(trainDate),2),dtype= int)*(-999)

    dateList1 = storeFeats["compDate"]
    dateList2 = storeFeats["promoDate"]
    intFeat = storeFeats["promoInterval"] 
    storeCompetitionDistance = storeFeats["distance"] 
    
    refDate = datetime(2013, 1, 1)
    orderFeat  = np.zeros((len(trainDate),1),dtype= int) 
    dateNumFeat  = np.zeros((len(trainDate),1),dtype= int) 
    dowCodeFeat  = np.zeros((len(trainDate),7),dtype= int) 
    distanceFeat          = np.zeros((len(trainDate),1),dtype= int)
    
    
    trendValCur           = np.ones((len(trainDate),1),dtype= int)*(-999)
    dmTrendValCur           = np.ones((len(trainDate),1),dtype= int)*(-999)
    trendValPrev          = np.ones((len(trainDate),1),dtype= int)*(-999)
    
    for i in range(0,len(trainDate)):
        if (pd.isnull(trainDate[i])):
            continue
        
        dateVal =  datetime.strptime(trainDate[i], '%Y-%m-%d')
        
        
        
        tr = getTrendsList(startDateList,trendMap,dateVal,0,storeStates[trainStoreId[i]-1])
            
        trendValCur[i] = tr + trainPromo[i]*100
        
        
        tr = getTrendsList(dmStartDateList,dmTrendMap,dateVal,0,storeStates[trainStoreId[i]-1])
        
        
        dmTrendValCur[i] =   trainPromo[i]*100 + 100 - tr      
          
        trainDateWeekNum[i] = dateVal.isocalendar()[1]
        trainDateDayNum[i] = dateVal.day     
        trainDateMonthNum[i] = dateVal.month
        trainDateYearNum[i] = dateVal.year
        
        if  dateList1[trainStoreId[i]-1] is not None:
            trainDateDiff1[i] = (dateVal-dateList1[trainStoreId[i]-1]).days
                    
        if  dateList2[trainStoreId[i]-1] is not None:
            trainDateDiff2[i] = (dateVal-dateList2[trainStoreId[i]-1]).days    

            intDate1 = datetime(trainDateYearNum[i], intFeat[trainStoreId[i]-1,0], 1)
            intDate2 = datetime(trainDateYearNum[i], intFeat[trainStoreId[i]-1,1], 1)
            intDate3 = datetime(trainDateYearNum[i], intFeat[trainStoreId[i]-1,2], 1)
            intDate4 = datetime(trainDateYearNum[i], intFeat[trainStoreId[i]-1,3], 1)
        
            promoList = []
            
            promoList = np.zeros((4,1), dtype = int)
            
            
            promoList[0] = (dateVal-intDate1).days
            promoList[1] = (dateVal-intDate2).days
            promoList[2] = (dateVal-intDate3).days
            promoList[3] = (dateVal-intDate4).days
            
            minVal1 = np.min(promoList > 0)
            minVal2 = -1*np.max(promoList < 0)            
        
            promoIntervalFeat[i,0] = minVal1
            promoIntervalFeat[i,1] = minVal2
            
            
        
        orderFeat[i] = (dateVal-refDate).days 

        firstDay = datetime(trainDateYearNum[i], 1, 1)
        
        dateNumFeat[i] = (dateVal-firstDay).days
         
        dowCodeFeat[i,trainDayOfWeek[i]-1] = 1
        

    
    storeCompetitionOpenSinceWeek = storeFeats["comWeek"]       
    storeCompetitionOpenSinceMonth = storeFeats["comMonth"]      
    storeCompetitionOpenSinceYear = storeFeats["comYear"]       
    storePromo2 = storeFeats["promoState"]    
    storePromo2SinceWeek = storeFeats["promoWeek"]     
    storePromo2SinceMonth = storeFeats["promoMonth"]    
    storePromo2SinceYear = storeFeats["promoYear"]       
    storeType = storeFeats["storeType"]     
    storeAssortment = storeFeats["assortment"]    
    storeTypeAssortConcat = storeFeats["typeAssortment"] 
    
    storeTypeEl     = storeType[0,:]
    assortmentEl    = storeAssortment[0,:]
    typeAssortEl    = storeTypeAssortConcat[0,:]
    
    rankList =  ss.rankdata(medianList)
    
    combinedDiff          = np.zeros((len(trainDate),1),dtype= int)    
    diffDaystoEnd         = np.zeros((len(trainDate),1),dtype= int)
    diffDaystoMid         = np.zeros((len(trainDate),1),dtype= int)
    medianFeat            = np.zeros((len(trainDate),1),dtype= float)
    meanFeat            = np.zeros((len(trainDate),1),dtype= float)
    rankFeat              = np.zeros((len(trainDate),1),dtype= int)
    comWeekFeat           = np.zeros((len(trainDate),1),dtype= int) 
    comMonthFeat          = np.zeros((len(trainDate),1),dtype= int) 
    comYearFeat           = np.zeros((len(trainDate),1),dtype= int) 
    promoStateFeat        = np.zeros((len(trainDate),1),dtype= int) 
    promoWeekFeat         = np.zeros((len(trainDate),1),dtype= int) 
    promoMonthFeat        = np.zeros((len(trainDate),1),dtype= int) 
    promoYearFeat         = np.zeros((len(trainDate),1),dtype= int) 
    storeTypeFeat         = np.zeros((len(trainDate),len(storeTypeEl)),dtype= int) 
    assortmentFeat        = np.zeros((len(trainDate),len(assortmentEl)),dtype= int) 
    storeTypeAssortConcatFeat  = np.zeros((len(trainDate),len(typeAssortEl)),dtype= int) 
    promoIntervalFeatCond = np.zeros((len(trainDate),1),dtype= int)


    weekModFeat           = np.zeros((len(trainDate),4),dtype= int)
    weekYearModFeat       = np.zeros((len(trainDate),4),dtype= int) 
    monthModFeat          = np.zeros((len(trainDate),3),dtype= int)
    monthFloorFeat        = np.zeros((len(trainDate),4),dtype= int)      
    dOMFloorFeat          = np.zeros((len(trainDate),5),dtype= int) 
    dOMModFeat            = np.zeros((len(trainDate),5),dtype= int)
    
    
    uniqueVals =  list(set(trainStoreId))
    
    
    ratioFeat             = np.zeros((len(trainDate),1),dtype= float)
    trFeat                = np.zeros((len(trainDate),1),dtype= float)
    sp1                   = np.zeros((len(trainDate),1),dtype= float)
    sp2                   = np.zeros((len(trainDate),1),dtype= float)    
    sp3                   = np.zeros((len(trainDate),1),dtype= float)
    sp4                   = np.zeros((len(trainDate),1),dtype= float)  
    sp5                   = np.zeros((len(trainDate),1),dtype= float)
    sp6                   = np.zeros((len(trainDate),1),dtype= float)     
    sp7                   = np.zeros((len(trainDate),1),dtype= float)         
           
    trainStoreStateFeat            = np.zeros((len(trainDate),trainStoreState.shape[1]),dtype= int)    
        
    trainWeekNumFeat      = np.zeros((len(trainDate),48),dtype= int)
        
        
         
#    retVal[0,0] =  chrisStartDiff
#    retVal[0,1] =  chrisEndDiff 
#    retVal[0,2] =  chrisRatio
#    retVal[0,3] =  chrisLength
#    retVal[0,4] =  winterStartDiff 
#    retVal[0,5] =  winterEndDiff 
#    retVal[0,6] =  winterRatio
#    retVal[0,7] =  winterLength 
#    retVal[0,8] =  easterStartDiff 
#    retVal[0,9] =  easterEndDiff
#    retVal[0,10] = easterRatio 
#    retVal[0,11] = easterLength
#    retVal[0,12] = witsunCount
#    retVal[0,13] = witsun1StartDiff 
#    retVal[0,14] = witsun1EndDiff
#    retVal[0,15] = witsun1Ratio
#    retVal[0,16] = witsun1Length  
#    retVal[0,17] = witsun2StartDiff 
#    retVal[0,18] = witsun2EndDiff
#    retVal[0,19] = witsun2Ratio 
#    retVal[0,20] = witsun2Length 
#    retVal[0,21] = summerStartDiff
#    retVal[0,22] = summerEndDiff 
#    retVal[0,23] = summerRatio    
#    retVal[0,24] = summerLength
#    retVal[0,25] = autumnStartDiff 
#    retVal[0,26] = autumnEndDiff
#    retVal[0,27] = autumnRatio
#    retVal[0,28] = autumnLength 
#    retVal[0,29] = nextChrisStartDiff 
#    retVal[0,30] = nextChrisEndDiff
#    retVal[0,31] = nextChrisRatio 
#    retVal[0,32] = nextChrisLength 
    
    
#    indices = np.array([0,4,8,13,17,21,25,29])
    
    indices = np.array(range(0,33))
    indices2 = np.array(range(0,36))
    
    
    trainHolidayFeat            = np.zeros((len(trainDate),len(indices)),dtype= float)
    trainCommonHolidayFeat      = np.zeros((len(trainDate),len(indices2)),dtype= float)
    
    
    ext1Feats                   = np.zeros((len(trainDate),4),dtype= float)
    ext2Feats                   = np.zeros((len(trainDate),3),dtype= float)
    ext3Feats                   = np.zeros((len(trainDate),1),dtype= float)
    ext4Feats                   = np.zeros((len(trainDate),4),dtype= float)
    
    wheatherFeats                   = np.zeros((len(trainDate),14),dtype= float)
    
    ext4RatiosFeats             = np.zeros((len(trainDate),4),dtype= float)
       
#    diffFeat1 = generateDeltaWeekSales(trainFile,1)
#    diffFeat2 = getMondaySales(trainFile, 0)
#    diffFeat3 = getPrevWeekDaySales(trainFile, 1,1)
#    diffFeat4 = getPrevWeekDaySales(trainFile, 1,5)
#    diffFeat5 = getPrevMidWeekDaySales(trainFile, 1)       
#    diffFeat6 = getPrevWeekendSales(trainFile, 1) 
    
    
    for i in range(0,len(trainStoreId)):    
        
#        if ((i+1) % 10000) == 0:
#            print(i+1)
            
            
        commonHolidayFeats = holidays.getCommonHolidays(trainDate[i])    
        holidayFeats = holidays.getSchoolHolidayFeats(trainDate[i], storeStates[trainStoreId[i]-1])
        
        
        trainHolidayFeat[i,:] = holidayFeats[0,indices]
        trainCommonHolidayFeat[i,:] = commonHolidayFeats[0,indices2]
        
        
        trainStoreStateFeat[i,:] = trainStoreState[trainStoreId[i]-1,:]
        
        dow = trainDayOfWeek[i]
        dom =  trainDateDayNum[i]
        
        
        if trainDateDayNum[i] == 15:
            indice1 = (trainDateMonthNum[i]-1)*4 + 1
            indice2 = (trainDateMonthNum[i]-1)*4 + 2
            trainWeekNumFeat[i,indice1] = 1
            trainWeekNumFeat[i,indice2] = 1
        elif trainDateDayNum[i] > 28:
            indice1 = (trainDateMonthNum[i]-1)*4 + 3
              
            if trainDateMonthNum[i] < 12:
                indice2 = (trainDateMonthNum[i])*4
            else:
                indice2 = 0
              
            trainWeekNumFeat[i,indice1] = 1
            trainWeekNumFeat[i,indice2] = 1
            
        else:
            indice = (trainDateMonthNum[i]-1)*4+int((trainDateDayNum[i]-1)/7)
            trainWeekNumFeat[i,indice] = 1
        
        if dow == 1:
            trVal = 10
        elif  dow == 2:
            trVal = 6
        elif dow == 3:
            trVal = 4
        elif dow == 4:
            trVal = 4
        elif dow == 5:
            trVal = 9 
        elif dow == 6:
            trVal = 10
        elif dow == 7:
            trVal = 10
        
        refDate = datetime(trainDateYearNum[i], trainDateMonthNum[i], 1)
        
        x = refDate.strftime('%Y-%m-%d')

        ext1Feats[i,0] = ext1File.A.values[ext1File.Date.values == x]
        ext1Feats[i,1] = ext1File.B.values[ext1File.Date.values == x]
        ext1Feats[i,2] = ext1File.C.values[ext1File.Date.values == x]
        ext1Feats[i,3] = ext1File.D.values[ext1File.Date.values == x]
        
        
        myFile = ext2FileMap[storeStates[trainStoreId[i]-1]]
        
        ext2Feats[i,0] = myFile.A.values[myFile.Date.values == x]
        ext2Feats[i,1] = myFile.B.values[myFile.Date.values == x]
        ext2Feats[i,2] = myFile.C.values[myFile.Date.values == x]
        
        
        
        ext3Feats[i,0] = ext3File.A.values[ext3File.Date.values == x]
        
        
        stateName = storeStates[trainStoreId[i]-1]


        ext4Feats[i,0] = ext4File.Area.values[ext4File.State.values == stateName]
        ext4Feats[i,1] = ext4File.Population.values[ext4File.State.values == stateName]
        ext4Feats[i,2] = ext4File.Density.values[ext4File.State.values == stateName]
        ext4Feats[i,3] = ext4File.GDP.values[ext4File.State.values == stateName]
        
        
        tempList = storeStates[storeStates == stateName]
        
        numOfStores = len(tempList)
        
        ext4RatiosFeats[i,0] =  float(float(ext4Feats[i,0])/float(numOfStores))
        ext4RatiosFeats[i,1] =  float(float(ext4Feats[i,1])/float(numOfStores))
        ext4RatiosFeats[i,2] =  float(float(ext4Feats[i,2])/float(numOfStores))
        ext4RatiosFeats[i,3] =  float(float(ext4Feats[i,3])/float(numOfStores))
        
             
             
        myFile = wheatherFileMap[storeStates[trainStoreId[i]-1]]            
            
        x = refDate.strftime('%Y%mWW%d')

        wheatherFeats[i,0] = myFile.A.values[myFile.Date.values == x]
        wheatherFeats[i,1] = myFile.B.values[myFile.Date.values == x]
        wheatherFeats[i,2] = myFile.C.values[myFile.Date.values == x]
        wheatherFeats[i,3] = myFile.D.values[myFile.Date.values == x]
        wheatherFeats[i,4] = myFile.E.values[myFile.Date.values == x]
        wheatherFeats[i,5] = myFile.F.values[myFile.Date.values == x]
        wheatherFeats[i,6] = myFile.G.values[myFile.Date.values == x]
        wheatherFeats[i,7] = myFile.H.values[myFile.Date.values == x]
        wheatherFeats[i,8] = myFile.I.values[myFile.Date.values == x]
        wheatherFeats[i,9] = myFile.J.values[myFile.Date.values == x]
        wheatherFeats[i,10] = myFile.K.values[myFile.Date.values == x]
        wheatherFeats[i,11] = myFile.L.values[myFile.Date.values == x]
        wheatherFeats[i,12] = myFile.M.values[myFile.Date.values == x]
        wheatherFeats[i,13] = myFile.N.values[myFile.Date.values == x]
                        
        
#        if (dom  > 13 and dom < 20) or (dom < 7)  or (dom > 28):
#            if dow == 1:
#                trVal = trVal + 12
#            elif  dow == 2:
#                trVal = trVal + 10
#            elif dow == 3:
#                trVal = trVal + 7 
#            elif dow == 4:
#                trVal = trVal + 7
#            elif dow == 5:
#                trVal = trVal + 12
#            elif dow == 6:
#                trVal = trVal + 12
#            elif dow == 7:
#                trVal = trVal

        if trainPromo[i] == 1:
            if dow == 1:
                trVal = trVal + 5
                sp1[i] = 1
                if (dom  > 13 and dom < 20):
                    sp3[i] = 1
                if (dom < 7)  or (dom > 28):
                    sp4[i] = 1                    
            elif  dow == 2:
                sp7[i] = 1
            elif dow == 3:
                sp7[i] = 1
            elif dow == 4:
                sp7[i] = 1
            elif dow == 5:
                sp2[i] = 1
                if (dom  > 13 and dom < 20):
                    sp5[i] = 1
                if (dom < 7)  or (dom > 28):
                    sp6[i] = 1                
            elif dow == 6:
                trVal = trVal + 4
            elif dow == 7:
                trVal = trVal
                
                                
        trFeat[i] = trVal
        
        if trainDayOfWeek[i] < 6:
            ratioFeat[i] = 1
        elif trainDayOfWeek[i] ==  6:
            ratioFeat[i] = saturdayRatios[trainStoreId[i]-1]
        elif trainDayOfWeek[i] ==  7:
            ratioFeat[i] = sundayRatios[trainStoreId[i]-1]
            
        totalDayInMonth = calendar.monthrange(trainDateYearNum[i],trainDateMonthNum[i])[1]
        
        diffToEnd = totalDayInMonth - trainDateDayNum[i]
        diffToMid =  15 - trainDateDayNum[i]   
        
        if diffToEnd < 2 or trainDateDayNum[i] < 2:
            diffDaystoEnd[i] = 1
            combinedDiff[i] = 1
            
        if  np.abs(diffToMid) < 2:
            diffDaystoMid[i] = 1
            combinedDiff[i] = 1
                  
#        diffFeat1[i]           = np.log(diffFeat1[i]*0.5+1)
#        diffFeat2[i]           = np.log(diffFeat2[i]*0.5+1)
#        diffFeat3[i]           = np.log(diffFeat3[i]*0.5+1)
#        diffFeat4[i]           = np.log(diffFeat4[i]*0.5+1)
#        diffFeat5[i]           = np.log(diffFeat5[i]*0.5+1)
#        diffFeat6[i]           = np.log(diffFeat6[i]*0.5+1)
        
        
        if trainPromo[i] == 0:
            medianFeat[i]          = np.log(float(medians0[trainStoreId[i]-1]*0.5+1))
            meanFeat[i]          = np.log(float(means0[trainStoreId[i]-1]*0.5+1))
        elif trainPromo[i] == 1:
            medianFeat[i]          = np.log(float(medians1[trainStoreId[i]-1]*0.5+1))  
            meanFeat[i]          = np.log(float(means1[trainStoreId[i]-1]*0.5+1))      
        
        
        trendValCur[i] =  np.log(trendValCur[i]*0.5+1)
        dmTrendValCur[i] =  np.log(dmTrendValCur[i]*0.5+1)
        
        
        
        rankFeat[i]            = rankList[trainStoreId[i]-1]
        distanceFeat[i]        = storeCompetitionDistance[trainStoreId[i]-1] 
        comWeekFeat[i]         = storeCompetitionOpenSinceWeek[trainStoreId[i]-1]
        comMonthFeat[i]        = storeCompetitionOpenSinceMonth[trainStoreId[i]-1]
        comYearFeat[i]         = storeCompetitionOpenSinceYear[trainStoreId[i]-1]
        promoStateFeat[i]      = storePromo2[trainStoreId[i]-1]
        promoWeekFeat[i]       = storePromo2SinceWeek[trainStoreId[i]-1]
        promoMonthFeat[i]      = storePromo2SinceMonth[trainStoreId[i]-1]
        promoYearFeat[i]       = storePromo2SinceYear[trainStoreId[i]-1]
         
        storeTypeFeat[i,:]     = storeType[trainStoreId[i]-1,:]
        assortmentFeat[i,:]    = storeAssortment[trainStoreId[i]-1,:]
        storeTypeAssortConcatFeat[i,:] = storeTypeAssortConcat[trainStoreId[i]-1,:]
        
        if trainDateDiff2[i] >= 0 :
            if ( trainDateMonthNum[i] == intFeat[trainStoreId[i]-1,0]):
                promoIntervalFeatCond[i] = 1    
            elif (trainDateMonthNum[i] == intFeat[trainStoreId[i]-1,1]):
                promoIntervalFeatCond[i] = 1         
            elif (trainDateMonthNum[i] == intFeat[trainStoreId[i]-1,2]):
                promoIntervalFeatCond[i] = 1         
            elif (trainDateMonthNum[i] == intFeat[trainStoreId[i]-1,3]):    
                promoIntervalFeatCond[i] = 1 
    
        
        
        weekModFeat[i,0] =  trainDateWeekNum[i]%2
        weekModFeat[i,1] =  trainDateWeekNum[i]%3
        weekModFeat[i,2] =  trainDateWeekNum[i]%4
        weekModFeat[i,3] =  trainDateWeekNum[i]%5

        weekYearModFeat[i,0] =  ((trainDateYearNum[i]%2)+trainDateWeekNum[i])%2
        weekYearModFeat[i,1] =  ((trainDateYearNum[i]%2)+trainDateWeekNum[i])%3
        weekYearModFeat[i,2] =  ((trainDateYearNum[i]%2)+trainDateWeekNum[i])%4
        weekYearModFeat[i,3] =  ((trainDateYearNum[i]%2)+trainDateWeekNum[i])%5
        
        
        monthModFeat[i,0] =  trainDateMonthNum[i]%2
        monthModFeat[i,1] =  trainDateMonthNum[i]%3
        monthModFeat[i,2] =  trainDateMonthNum[i]%4    
        
        monthFloorFeat[i,0] =  np.floor(trainDateMonthNum[i]/2)
        monthFloorFeat[i,1] =  np.floor(trainDateMonthNum[i]/3)
        monthFloorFeat[i,2] =  np.floor(trainDateMonthNum[i]/4) 
        monthFloorFeat[i,3] =  np.floor(trainDateMonthNum[i]/6)    
        
        dOMFloorFeat[i,0] =  np.floor(trainDateDayNum[i]/2)
        dOMFloorFeat[i,1] =  np.floor(trainDateDayNum[i]/3)
        dOMFloorFeat[i,2] =  np.floor(trainDateDayNum[i]/4) 
        dOMFloorFeat[i,3] =  np.floor(trainDateDayNum[i]/7) 
        dOMFloorFeat[i,4] =  np.floor(trainDateDayNum[i]/15)     
            
            
        dOMModFeat[i,0] =  trainDateDayNum[i]%2
        dOMModFeat[i,1] =  trainDateDayNum[i]%3
        dOMModFeat[i,2] =  trainDateDayNum[i]%4 
        dOMModFeat[i,3] =  trainDateDayNum[i]%7 
        dOMModFeat[i,4] =  trainDateDayNum[i]%15
    
    
    
    trainOpen = trainOpen.reshape(len(trainStoreId),1)
    trainDayOfWeek = trainDayOfWeek.reshape(len(trainStoreId),1)
    trainPromo = trainPromo.reshape(len(trainStoreId),1)
    trainStoreId = trainStoreId.reshape(len(trainStoreId),1)
    

    
    c1,c2 = getConsolidatedPromoFeats(trainFile)
    
    
    
#    mediansFeat2 = getMedianFeats(trainFile,240)
    
#    for i in range(0,len(mediansFeat2)):
#        mediansFeat2[i] = np.log(float(mediansFeat2[i]*0.5+1))
    
    
    stateOpenFeat = getStateOpenNumbers(trainFile,storeStates)
        
    featBefore,featAfter,consClosedBefore,consClosedAfter = getOpenFeats(trainFile)
    
    statePromoFeat = getStatePromoNumbers(trainFile,storeStates)

#    statePromoCondFeat = getStatePromoConditions(trainFile,storeStates)
    
    
    openDaysFeat1 = getOpenDays(trainFile,0)
    
    closedDaysFeat = getClosedCondition(trainFile,0)
    
    trainFeats = np.concatenate((trainDayOfWeek,trendValCur,trainStateHolidayFeat,trainSchoolHolidayFeat),axis=1)    
    trainFeats = np.concatenate((trainFeats,trainDateWeekNum,trainDateDayNum,trainDateMonthNum,trainDateYearNum,trainDateDiff1,trainDateDiff2),axis=1)  
    trainFeats = np.concatenate((trainFeats,medianFeat,rankFeat,distanceFeat,comWeekFeat,comMonthFeat,comYearFeat),axis=1) 
    trainFeats = np.concatenate((trainFeats,promoStateFeat,promoWeekFeat,promoMonthFeat,promoYearFeat,storeTypeFeat,assortmentFeat,promoIntervalFeat,promoIntervalFeatCond),axis=1) 
    trainFeats = np.concatenate((trainFeats,dateNumFeat,dowCodeFeat,trainHolidayFeat,trainStoreId,trainCommonHolidayFeat,c1,c2,sp1,sp2,sp3,sp4,sp5,sp6,sp7),axis=1) 
    trainFeats = np.concatenate((trainFeats,trainStoreStateFeat,stateOpenFeat,featBefore,featAfter,consClosedBefore,consClosedAfter,statePromoFeat,wheatherFeats),axis=1)     


#    trainFeats = np.concatenate((trainFeats,ext1Feats,ext2Feats,ext3Feats,ext4Feats,ext4RatiosFeats),axis=1)

    trainFeats = np.concatenate((trainFeats,closedDaysFeat),axis=1)    
    
#    trainFeats = np.concatenate((trainFeats,diffFeat1,diffFeat2,diffFeat3,diffFeat4,diffFeat5,diffFeat6),axis=1) 
    
    return trainFeats,ratioFeat                        
    
    
def getMedians(trainFile, cond):    

    trainStoreId = trainFile.Store.values #key
    trainSales = trainFile.Sales.values
    trainDayOfWeek = trainFile.DayOfWeek.values #num(can be categorical)
    trainStateHoliday = trainFile.StateHoliday.values #categorical
    trainSchoolHoliday = trainFile.SchoolHoliday.values #categorical
    trainDate = trainFile.Date.values  #date

    trainDateWeekNum  = np.zeros((len(trainDate),1),dtype= int) 
    trainDateDayNum = np.zeros((len(trainDate),1),dtype= int) 
    trainDateMonthNum = np.zeros((len(trainDate),1),dtype= int) 
    trainDateYearNum  = np.zeros((len(trainDate),1),dtype= int) 
    
    trainDateDiff1 = np.zeros((len(trainDate),1),dtype= int) 
    trainDateDiff2 = np.zeros((len(trainDate),1),dtype= int) 
        
    for i in range(0,len(trainDate)):
        if (pd.isnull(trainDate[i])):
            continue
        
        dateVal =  datetime.strptime(trainDate[i], '%Y-%m-%d')
        
        trainDateWeekNum[i] = dateVal.isocalendar()[1]
        trainDateDayNum[i] = dateVal.day     
        trainDateMonthNum[i] = dateVal.month
        trainDateYearNum[i] = dateVal.year
        
            
    myMap = {}
    dateMap = {}
    weekMap = {}
    for i in range(0,len(trainStoreId)):
        key = trainStoreId[i]
        dayOfWeek = trainDayOfWeek[i]
        label = trainSales[i]
        
        
        if label == 0: 
            continue
        
        if cond == 1 and dayOfWeek > 5:
            continue
            
        if cond == 2 and dayOfWeek != 6:
            continue
            
        if cond == 3 and dayOfWeek != 7:
            continue
            
        if key not in myMap:
            myMap[key] = []
            dateMap[key] = []
            weekMap[key] = []
            
        myMap[key].append(label)
        dateMap[key].append(trainDate[i])
        weekMap[key].append(trainDateWeekNum[i])
        
    storeIds = sorted(list(set(trainStoreId)))
    
    medianList = []
    for i in range(0,len(storeIds)):
        
        key = storeIds[i]
        
        if key not in myMap:
            medianList.append(0)
            continue
            
        myList =  myMap[key]
        
        meanVal = np.mean(myList)
        minVal = np.min(myList)        
        maxVal = np.max(myList)
        medianVal = np.median(myList)
        stdVal = np.std(myList)
        
        medianList.append(medianVal)
        
#        print('id : %d  mean: %d min: %d max: %d median: %d std: %d' % (key,meanVal,minVal,maxVal,medianVal,stdVal))    
    
    
    return medianList    
    


def getMediansPromo(trainFile, cond):    

    trainStoreId = trainFile.Store.values #key
    trainSales = trainFile.Sales.values
    
    trainPromo = trainFile.Promo.values  #date

            
    myMap = {}
    for i in range(0,len(trainStoreId)):
        key = trainStoreId[i]
        label = trainSales[i]
        
        if label == 0: 
            continue
        
        if trainPromo[i] != cond :
            continue
            
            
        if key not in myMap:
            myMap[key] = []
            
        myMap[key].append(label)
        
    storeIds = sorted(list(set(trainStoreId)))
    
    medianList = []
    for i in range(0,len(storeIds)):
        
        key = storeIds[i]
        
        if key not in myMap:
            medianList.append(0)
            continue
            
        myList =  myMap[key]
        
        medianVal = np.median(myList)
        
        medianList.append(medianVal)
        
#        print('id : %d  mean: %d min: %d max: %d median: %d std: %d' % (key,meanVal,minVal,maxVal,medianVal,stdVal))    
    
    
    return medianList    
    
    
def getMeansPromo(trainFile, cond):    

    trainStoreId = trainFile.Store.values #key
    trainSales = trainFile.Sales.values
    
    trainPromo = trainFile.Promo.values  #date

            
    myMap = {}
    for i in range(0,len(trainStoreId)):
        key = trainStoreId[i]
        label = trainSales[i]
        
        if label == 0: 
            continue
        
        if trainPromo[i] != cond :
            continue
            
            
        if key not in myMap:
            myMap[key] = []
            
        myMap[key].append(label)
        
    storeIds = sorted(list(set(trainStoreId)))
    
    medianList = []
    for i in range(0,len(storeIds)):
        
        key = storeIds[i]
        
        if key not in myMap:
            medianList.append(0)
            continue
            
        myList =  myMap[key]
        
        medianVal = np.mean(myList)
        
        medianList.append(medianVal)
        
    return medianList   
    
        
def calculateWeekendRatios(listWeekdays, listSaturdays, listSundays):
    
    saturdayRatios = []
    sundayRatios = []
    
    for i in range(0,len(listWeekdays)):
        weekDaysValue = listWeekdays[i]
        saturdayValue = listSaturdays[i]
        sundayValue = listSundays[i]
   
        if saturdayValue == 0:
            saturdayRatios.append(0) 
        else:
            saturdayRatios.append(weekDaysValue / saturdayValue) 
            
        if sundayValue == 0:
            sundayRatios.append(0) 
        else:
            sundayRatios.append(weekDaysValue / sundayValue) 
            
    return saturdayRatios,sundayRatios
    
                    
def parseTrainData2(trainFile):    

    trainStoreId = trainFile.Store.values #key
    trainSales = trainFile.Sales.values
    trainDayOfWeek = trainFile.DayOfWeek.values #num(can be categorical)
    trainStateHoliday = trainFile.StateHoliday.values #categorical
    trainSchoolHoliday = trainFile.SchoolHoliday.values #categorical
    trainDate = trainFile.Date.values  #date
    trainOpen    = trainFile.Open.values
    trainPromo    = trainFile.Promo.values
    
#    diffSales = generateDeltaWeekSales(trainFile, trainSales, 0)

    trainDateWeekNum  = np.zeros((len(trainDate),1),dtype= int) 
    trainDateDayNum = np.zeros((len(trainDate),1),dtype= int) 
    trainDateMonthNum = np.zeros((len(trainDate),1),dtype= int) 
    trainDateYearNum  = np.zeros((len(trainDate),1),dtype= int) 
    
    trainDateDiff1 = np.zeros((len(trainDate),1),dtype= int) 
    trainDateDiff2 = np.zeros((len(trainDate),1),dtype= int) 
        
    for i in range(0,len(trainDate)):
        if (pd.isnull(trainDate[i])):
            continue
        
        dateVal =  datetime.strptime(trainDate[i], '%Y-%m-%d')
        
        trainDateWeekNum[i] = dateVal.isocalendar()[1]
        trainDateDayNum[i] = dateVal.day     
        trainDateMonthNum[i] = dateVal.month
        trainDateYearNum[i] = dateVal.year
        
            
    myMap = {}
    dateMap = {}
    weekMap = {}
    meanMap = {}
    dowMap = {}
    for i in range(0,len(trainStoreId)):
        key = trainStoreId[i]
        dayOfWeek = trainDayOfWeek[i]
        label = trainSales[i]
        
        if label == 0 or trainOpen[i] == 0 :
            continue
            
        if key not in myMap:
            myMap[key] = []
            dateMap[key] = []
            weekMap[key] = []
            meanMap[key] = []
            dowMap[key] = []
            
        myMap[key].append(label)
        dateMap[key].append(trainDate[i])
        weekMap[key].append(trainDateWeekNum[i])
        meanMap[key].append(trainPromo[i])
        dowMap[key].append(trainDayOfWeek[i])
    storeIds = sorted(list(set(trainStoreId)))
    
    for i in range(0,len(storeIds)):
        key = storeIds[i]
        myList =  myMap[key]
        
        meanVal = np.mean(myList)
        minVal = np.min(myList)        
        maxVal = np.max(myList)
        medianVal = np.median(myList)
        stdVal = np.std(myList)
        
#        print('id : %d  mean: %d min: %d max: %d median: %d std: %d' % (key,meanVal,minVal,maxVal,medianVal,stdVal))
                                
                                
    myListVal = myMap[10]
    myListDate = dateMap[10]
    trainDateWeekNum_ = weekMap[10]
    meanVals = meanMap[10]
    dowVals = dowMap[10]
    for i in range(0,len(myListVal)):
        print(myListDate[i] +"  " + str(myListVal[i]) + "  " + str(meanVals[i])+ "  " + str(dowVals[i]))        

    return trainFeats        
    
def getExternalData(externalFile):
    startDates = externalFile.start.values
    endDates = externalFile.end.values
    externalFile.drop('start', axis=1,  inplace=True)
    externalFile.drop('end', axis=1,  inplace=True)
    
    
    stateList = ['RH','BD','SCA','BY','TH','SH','HE','SC','NRW','BR','HM','BE', 'GE']
    
    trendMap = {}
    

    for i in range(0,len(stateList)):  
        trendMap[stateList[i]] = externalFile[stateList[i]].values
        
        
    startDateList = []
    for i in range(0,len(startDates)):
        dateVal =  datetime.strptime(startDates[i], '%Y-%m-%d')
        startDateList.append(dateVal)
    
    return startDateList, trendMap
    
    
def getTrendsList(trendDateList,trendMap,dateVal,delta,state):
   
    trends = trendMap[state]
    
    weekStart = dateVal - timedelta(days=dateVal.weekday()+1)
 
    for i in range(0,len(trendDateList)):
        diffDay = (weekStart-trendDateList[i]).days
     
        if diffDay == 0:
            trendsData =  trends[i-delta]
            break

    return trendsData         

def parseStateData(stateFile):    

    storeId = stateFile.store.values #key
    state = stateFile.state.values      
    
    retValStoreStates = np.ndarray((len(storeId)),dtype=object)
    
    for i in range(0,len(storeId)):
        retValStoreStates[storeId[i]-1] = state[i]
    
    return retValStoreStates
                 
                 
def generateDeltaWeekSales(trainFile, trainSales, deltaVal):     

    trainStoreId = trainFile.Store.values 
    trainDate    = trainFile.Date.values
    trainOpen    = trainFile.Open.values
    
    trainDayOfWeek = trainFile.DayOfWeek.values
    
    refDate = datetime(2012, 12, 31)                
    
    storeMap = {}
    
    for i in range(0,len(trainStoreId)):   

        if   trainSales[i] == 0 or trainOpen[i] == 0 or trainDayOfWeek[i] > 5:
            continue
                    
        dateVal =  datetime.strptime(trainDate[i], '%Y-%m-%d')
        
        if trainStoreId[i] not in storeMap:
            storeMap[trainStoreId[i]]  = {}   
        
        diffDays = (dateVal-refDate).days 
                                
        dateKey = int(diffDays/7)                        
 
        myMap = storeMap[trainStoreId[i]]      
            
        if dateKey not in myMap:
            myMap[dateKey] = []     
                 
        myMap[dateKey].append(trainSales[i])           


    diffFeat = np.ones((len(trainStoreId),1),dtype=float)*(-999)

    for i in range(0,len(trainStoreId)):            
        dateVal =  datetime.strptime(trainDate[i], '%Y-%m-%d')
        
        if trainStoreId[i] not in storeMap:
            diffFeat[i]  = -999   
            continue
            
        diffDays = (dateVal-refDate).days 
        dateKey = int(diffDays/7) - deltaVal       
                        
        myMap = storeMap[trainStoreId[i]]      
            
        if dateKey not in myMap:
            diffFeat[i]  = -999 
            continue    
                 
        salesVals = myMap[dateKey]
        
        diffFeat[i]  =  np.mean(np.array(salesVals))
        
#        if trainStoreId[i] == 2 and trainSales[i] != 0:
#            print(str(trainDayOfWeek[i]) + "  " + str(trainDate[i])+"  "+str(trainSales[i])+"  "+ str(diffFeat[i]))
        
    return diffFeat[:,0]                 
                 
                 

def getMondaySales(trainFile,trainSales, deltaVal):     

    trainStoreId = trainFile.Store.values 
    trainDate    = trainFile.Date.values
    trainOpen    = trainFile.Open.values
        
    trainDayOfWeek = trainFile.DayOfWeek.values
    
    refDate = datetime(2012, 12, 31)                
    
    storeMap = {}
    
    for i in range(0,len(trainStoreId)):   

        if  trainSales[i] == 0 or trainOpen[i] == 0 or trainDayOfWeek[i] != 1:
            continue
                    
        dateVal =  datetime.strptime(trainDate[i], '%Y-%m-%d')
        
        if trainStoreId[i] not in storeMap:
            storeMap[trainStoreId[i]]  = {}   
        
        diffDays = (dateVal-refDate).days 
                                
        dateKey = int(diffDays/7)                        
         
        myMap = storeMap[trainStoreId[i]]      
            
        if dateKey not in myMap:
            myMap[dateKey] = []     
                 
        myMap[dateKey].append(trainSales[i])           


    diffFeat = np.ones((len(trainStoreId),1),dtype=float)*(-999)

    for i in range(0,len(trainStoreId)):            
        dateVal =  datetime.strptime(trainDate[i], '%Y-%m-%d')
        
        if trainStoreId[i] not in storeMap or trainDayOfWeek[i] == 1:
            diffFeat[i]  = -999   
            continue
            
        diffDays = (dateVal-refDate).days 
        dateKey = int(diffDays/7) - deltaVal                       
        myMap = storeMap[trainStoreId[i]]      
            
        if dateKey not in myMap:
            diffFeat[i]  = -999 
            continue    
                 
        salesVals = myMap[dateKey]
        
        diffFeat[i]  =  np.mean(np.array(salesVals))
        
#        if trainStoreId[i] == 2 and trainSales[i] != 0:
#            print(str(trainDayOfWeek[i]) + "  " + str(trainDate[i])+"  "+str(trainSales[i])+"  "+ str(diffFeat[i]))
        
    return diffFeat[:,0]     
                 
                 
                 
def getPrevWeekDaySales(trainFile,trainSales, deltaVal, dayNum):     

    trainStoreId = trainFile.Store.values 
    trainDate    = trainFile.Date.values
    trainOpen    = trainFile.Open.values
        
    trainDayOfWeek = trainFile.DayOfWeek.values
    
    refDate = datetime(2012, 12, 31)                
    
    storeMap = {}
    
    for i in range(0,len(trainStoreId)):   

        if  trainSales[i] == 0 or trainOpen[i] == 0 or trainDayOfWeek[i] != dayNum:
            continue
                    
        dateVal =  datetime.strptime(trainDate[i], '%Y-%m-%d')
        
        if trainStoreId[i] not in storeMap:
            storeMap[trainStoreId[i]]  = {}   
        
        diffDays = (dateVal-refDate).days 
                                
        dateKey = int(diffDays/7)                        
         
        myMap = storeMap[trainStoreId[i]]      
            
        if dateKey not in myMap:
            myMap[dateKey] = []     
                 
        myMap[dateKey].append(trainSales[i])           


    diffFeat = np.ones((len(trainStoreId),1),dtype=float)*(-999)

    for i in range(0,len(trainStoreId)):            
        dateVal =  datetime.strptime(trainDate[i], '%Y-%m-%d')
        
        if trainStoreId[i] not in storeMap :
            diffFeat[i]  = -999   
            continue
            
        diffDays = (dateVal-refDate).days 
        dateKey = int(diffDays/7) - deltaVal                       
        myMap = storeMap[trainStoreId[i]]      
            
        if dateKey not in myMap:
            diffFeat[i]  = -999 
            continue    
                 
        salesVals = myMap[dateKey]
        
        diffFeat[i]  =  np.mean(np.array(salesVals))
        
#        if trainStoreId[i] == 2 and trainSales[i] != 0:
#            print(str(trainDayOfWeek[i]) + "  " + str(trainDate[i])+"  "+str(trainSales[i])+"  "+ str(diffFeat[i]))
        
    return diffFeat[:,0]     
                                  
def getPrevMidWeekDaySales(trainFile,trainSales, deltaVal):     

    trainStoreId = trainFile.Store.values 
    trainDate    = trainFile.Date.values
    trainOpen    = trainFile.Open.values
        
    trainDayOfWeek = trainFile.DayOfWeek.values
    
    refDate = datetime(2012, 12, 31)                
    
    storeMap = {}
    
    for i in range(0,len(trainStoreId)):   

        if  trainSales[i] == 0 or trainOpen[i] == 0 or ( trainDayOfWeek[i] == 1 or   trainDayOfWeek[i] > 4):
            continue
                    
        dateVal =  datetime.strptime(trainDate[i], '%Y-%m-%d')
        
        if trainStoreId[i] not in storeMap:
            storeMap[trainStoreId[i]]  = {}   
        
        diffDays = (dateVal-refDate).days 
                                
        dateKey = int(diffDays/7)                        
         
        myMap = storeMap[trainStoreId[i]]      
            
        if dateKey not in myMap:
            myMap[dateKey] = []     
                 
        myMap[dateKey].append(trainSales[i])           


    diffFeat = np.ones((len(trainStoreId),1),dtype=float)*(-999)

    for i in range(0,len(trainStoreId)):            
        dateVal =  datetime.strptime(trainDate[i], '%Y-%m-%d')
        
        if trainStoreId[i] not in storeMap :
            diffFeat[i]  = -999   
            continue
            
        diffDays = (dateVal-refDate).days 
        dateKey = int(diffDays/7) - deltaVal                       
        myMap = storeMap[trainStoreId[i]]      
            
        if dateKey not in myMap:
            diffFeat[i]  = -999 
            continue    
                 
        salesVals = myMap[dateKey]
        
        diffFeat[i]  =  np.mean(np.array(salesVals))
        
#        if trainStoreId[i] == 2 and trainSales[i] != 0:
#            print(str(trainDayOfWeek[i]) + "  " + str(trainDate[i])+"  "+str(trainSales[i])+"  "+ str(diffFeat[i]))
        
    return diffFeat[:,0]  
    
def getPrevWeekendSales(trainFile,trainSales, deltaVal):     

    trainStoreId = trainFile.Store.values 
    trainDate    = trainFile.Date.values
    trainOpen    = trainFile.Open.values
        
    trainDayOfWeek = trainFile.DayOfWeek.values
    
    refDate = datetime(2012, 12, 31)                
    
    storeMap = {}
    
    for i in range(0,len(trainStoreId)):   

        if  trainSales[i] == 0 or trainOpen[i] == 0 or  trainDayOfWeek[i] < 6:
            continue
                    
        dateVal =  datetime.strptime(trainDate[i], '%Y-%m-%d')
        
        if trainStoreId[i] not in storeMap:
            storeMap[trainStoreId[i]]  = {}   
        
        diffDays = (dateVal-refDate).days 
                                
        dateKey = int(diffDays/7)                        
         
        myMap = storeMap[trainStoreId[i]]      
            
        if dateKey not in myMap:
            myMap[dateKey] = []     
                 
        myMap[dateKey].append(trainSales[i])           


    diffFeat = np.ones((len(trainStoreId),1),dtype=float)*(-999)

    for i in range(0,len(trainStoreId)):            
        dateVal =  datetime.strptime(trainDate[i], '%Y-%m-%d')
        
        if trainStoreId[i] not in storeMap :
            diffFeat[i]  = -999   
            continue
            
        diffDays = (dateVal-refDate).days 
        dateKey = int(diffDays/7) - deltaVal                       
        myMap = storeMap[trainStoreId[i]]      
            
        if dateKey not in myMap:
            diffFeat[i]  = -999 
            continue    
                 
        salesVals = myMap[dateKey]
        
        diffFeat[i]  =  np.mean(np.array(salesVals))
        
#        if trainStoreId[i] == 2 and trainSales[i] != 0:
#            print(str(trainDayOfWeek[i]) + "  " + str(trainDate[i])+"  "+str(trainSales[i])+"  "+ str(diffFeat[i]))
        
    return diffFeat[:,0]         
    
    
def getConsolidatedPromoFeats(trainFile):     

    trainStoreId = trainFile.Store.values 
    trainDate    = trainFile.Date.values
    trainOpen    = trainFile.Open.values
    trainPromo  = trainFile.Promo.values
    
    
    trainDayOfWeek = trainFile.DayOfWeek.values
    
    refDate = datetime(2012, 12, 31)                
    
    storeMap = {}
    
    for i in range(0,len(trainStoreId)):   

        if trainOpen[i] == 0 or trainDayOfWeek[i] > 5:
            continue
                    
        dateVal =  datetime.strptime(trainDate[i], '%Y-%m-%d')
        
        if trainStoreId[i] not in storeMap:
            storeMap[trainStoreId[i]]  = {}   
        
        if trainPromo[i] == 1:
            diffDays = (dateVal-refDate).days 
                                
            dateKey = int(diffDays/7)                        
 
            myMap = storeMap[trainStoreId[i]]      
            
            if dateKey not in myMap:
                myMap[dateKey] = 1     
                 

    diffFeat = np.ones((len(trainStoreId),6),dtype=float)*(-999)
    diffType = np.zeros((len(trainStoreId),4),dtype=int)
    
    
    for i in range(0,len(trainStoreId)):            
        dateVal =  datetime.strptime(trainDate[i], '%Y-%m-%d')
        
        if trainStoreId[i] not in storeMap or trainOpen[i] == 0:
            continue
      
                        
        diffDays = (dateVal-refDate).days 
        dateKey = int(diffDays/7)            
                        
        myMap = storeMap[trainStoreId[i]]      
             
                
        promoLen  = -999
        dateIndex = -999
        remainingDays = -999
        passedDays = -999
        nextPromoLen = -999
        prevPromoLen = -999
                          
        if (dateKey in myMap) and (dateKey + 1 in myMap):
            diffType[i,0] = 1
            promoLen = 12
            dateIndex = trainDayOfWeek[i]
            remainingDays = 12 - dateIndex
            
        elif  (dateKey  in myMap) and (dateKey - 1 in myMap) :
            diffType[i,1] = 1
            promoLen = 12
            dateIndex = trainDayOfWeek[i] + 7
            remainingDays = 6 - dateIndex
            
        elif (dateKey  in myMap):
            diffType[i,2] = 1
            promoLen = 5
            dateIndex = trainDayOfWeek[i]
            remainingDays = 6 - dateIndex    
                    
        elif (dateKey + 1  in myMap):
            diffType[i,3] = 1
            remainingDays = 8 - trainDayOfWeek[i]       
            nextPromoLen = 1
            
            if (dateKey + 2  in myMap):
                nextPromoLen = 2    
            
            if dateKey - 1  in myMap and  dateKey - 2  in myMap:
                prevPromoLen = 2
                passedDays = trainDayOfWeek[i] + 2
            elif  dateKey - 1  in myMap:
                prevPromoLen = 1
                passedDays = trainDayOfWeek[i] + 2
                   
        diffFeat[i,0] = promoLen         
        diffFeat[i,1] = dateIndex  
        diffFeat[i,2] = remainingDays 
        diffFeat[i,3] = nextPromoLen 
        diffFeat[i,4] = prevPromoLen 
        diffFeat[i,5] = passedDays
      
    return diffFeat,diffType
    
    
def getOpenFeats(trainFile):     

    trainStoreId = trainFile.Store.values 
    trainDate    = trainFile.Date.values
    trainOpen    = trainFile.Open.values
    trainPromo  = trainFile.Promo.values
    
    
    trainDayOfWeek = trainFile.DayOfWeek.values
    
    refDate = datetime(2012, 12, 31)                
    
    storeMap = {}
    
    for i in range(0,len(trainStoreId)):   
 
        dateVal =  datetime.strptime(trainDate[i], '%Y-%m-%d')
        
        if trainStoreId[i] not in storeMap:
            storeMap[trainStoreId[i]]  = {}   
        
        if trainOpen[i] == 0:
            diffDays = (dateVal-refDate).days 
                                
            dateKey = diffDays                        
 
            myMap = storeMap[trainStoreId[i]]      
            
            if dateKey not in myMap:
                myMap[dateKey] = 1     
                 

    featBefore = np.ones((len(trainStoreId),3),dtype=float)*(-999)
    featAfter  = np.ones((len(trainStoreId),3),dtype=float)*(-999)
        
    consClosedBefore  = np.ones((len(trainStoreId),5),dtype=float)*(-999)
    consClosedAfter  = np.ones((len(trainStoreId),5),dtype=float)*(-999)
        
    for i in range(0,len(trainStoreId)):            
        dateVal =  datetime.strptime(trainDate[i], '%Y-%m-%d')
        
        if trainStoreId[i] not in storeMap or trainOpen[i] == 0:
            continue
        
                        
        diffDays = (dateVal-refDate).days 
        dateKey  =  diffDays            
                        
        myMap = storeMap[trainStoreId[i]]      
             
        dow = trainDayOfWeek[i]     
        
        if  ((dateKey -1) in myMap) and ((dow-1)%7) != 0:
            featBefore[i,0] = 1                     
        if  ((dateKey -1) in myMap) and ((dateKey -2) in myMap) and ((dow-1)%7) != 0 and ((dow-2)%7) != 0:
            featBefore[i,1] = 1   
        if  ((dateKey -1) in myMap) and  ((dateKey -2) in myMap) and ((dateKey -3) in myMap) and ((dow-1)%7) != 0 and ((dow-2)%7) != 0 and ((dow-3)%7) != 0:
            featBefore[i,2] = 1               


        if ((dateKey +1) in myMap) and ((dow+1)%7) != 0:
            featAfter[i,0] = 1                     
        if ((dateKey+1) in myMap) and ((dateKey+2) in myMap) and ((dow+1)%7) != 0 and ((dow+2)%7) != 0:
            featAfter[i,1] = 1   
        if ((dateKey+1) in myMap) and  ((dateKey+2) in myMap) and ((dateKey+3) in myMap) and ((dow+1)%7) != 0 and ((dow+2)%7) != 0 and ((dow+3)%7) != 0:
            featAfter[i,2] = 1 
                                      
       
        for k in range(0,5):       
            t = 0
            ind = -1*k
           
            while True:
                ind = ind-1
               
                if ((dow+ind)%7) == 0:
                    continue
               
                if ((dateKey+ind) in myMap):
                    t = t + 1
                else:
                    break
           
            if t != 0:
                consClosedBefore[i,k] = t
                      
        for k in range(0,5):       
            t = 0
            ind = k
           
            while True:
                ind = ind+1
               
                if ((dow+ind)%7) == 0:
                    continue
               
                if ((dateKey+ind) in myMap):
                    t = t + 1
                else:
                    break
           
            if t != 0:
                consClosedAfter[i,k] = t
               
            
    return featBefore,featAfter,consClosedBefore,consClosedAfter
    
    
    
def getStateOpenNumbers(trainFile,storeStates):

    trainStoreId = trainFile.Store.values 
    trainDate    = trainFile.Date.values
    trainOpen    = trainFile.Open.values
    trainPromo  = trainFile.Promo.values
    
    
    trainDayOfWeek = trainFile.DayOfWeek.values
    
    storeOpen = {}
    storeClosed = {}
    storeTotal = {}
    
    stateOpenFeat = np.ones((len(trainStoreId),5),dtype=float)*(-999)
    
    for i in range(0,len(trainStoreId)):  
    
        stateVal =  storeStates[trainStoreId[i]-1]
        
        if stateVal not in storeOpen:
            storeOpen[stateVal]  = {}
            storeClosed[stateVal]  = {} 
            storeTotal[stateVal]  = {} 
               
        map1 = storeOpen[stateVal]
        map2 = storeClosed[stateVal]
        map3 = storeTotal[stateVal]
               
        if trainDate[i] not in map1:
            map1[trainDate[i]] = []    
            map2[trainDate[i]] = []  
            map3[trainDate[i]] = []  

        if trainOpen[i] == 0:
            map2[trainDate[i]].append(1)
        else:
            map1[trainDate[i]].append(1)
        
        map3[trainDate[i]].append(1)


    for i in range(0,len(trainStoreId)):  
    
        stateVal =  storeStates[trainStoreId[i]-1]
        
        if stateVal not in storeOpen:
            continue
               
        map1 = storeOpen[stateVal]
        map2 = storeClosed[stateVal]
        map3 = storeTotal[stateVal]
               
        if trainDate[i] not in map1:
            continue  

        numOfOpen = np.sum(np.array(map1[trainDate[i]]))
        numOfClosed = np.sum(np.array(map2[trainDate[i]]))
        numOfTotal = np.sum(np.array(map3[trainDate[i]]))
        ratio1 = float(numOfOpen/numOfTotal)
        
        if numOfOpen > 0:
            ratio2 = float(numOfClosed/numOfOpen)
        else:
            ratio2 = -999
       
        stateOpenFeat[i,0] =  numOfOpen    
        stateOpenFeat[i,1] =  numOfClosed 
        stateOpenFeat[i,2] =  numOfTotal 
        stateOpenFeat[i,3] =  ratio1 
        stateOpenFeat[i,4] =  ratio2
       
       
    return stateOpenFeat       
    

def getMedianFeats(trainFile,windowSize):     
    print("Medians Started")
    
    trainStoreId = trainFile.Store.values 
    trainDate    = trainFile.Date.values
    trainOpen    = trainFile.Open.values
    trainSales   = trainFile.Sales.values    
    
    trainDayOfWeek = trainFile.DayOfWeek.values
    
    refDate = datetime(2012, 12, 31)                
    
    storeMap = {}
    
    storeIds = sorted(list(set(trainStoreId)))
    
    for i in range(0,len(trainStoreId)):   

        if  trainSales[i] == 0 or trainOpen[i] == 0:
            continue
                    
        dateVal =  datetime.strptime(trainDate[i], '%Y-%m-%d')
        
        if trainStoreId[i] not in storeMap:
            storeMap[trainStoreId[i]]  = {}   
        
        dateKey = (dateVal-refDate).days 
                                
        myMap = storeMap[trainStoreId[i]]      
            
        myMap[dateKey] =  trainSales[i]    
                 

    medianFeat = np.ones((len(trainStoreId),1),dtype=float)*(-999)

    for i in range(0,len(trainStoreId)): 
                  
        dateVal =  datetime.strptime(trainDate[i], '%Y-%m-%d')
        
        if trainStoreId[i] not in storeMap :
            diffFeat[i]  = -999   
            continue
            
        dateKey = (dateVal-refDate).days 
        
        myMap = storeMap[trainStoreId[i]]      
            
        offset = 0    
        
        while True:
            myList = []    
            for j in range(45,windowSize+45):
                if dateKey+offset-j in myMap:
                    val = myMap[dateKey+offset-j]
                    if val > 0:
                        myList.append(val)        
               
            
            if len(myList) >= 0.5*windowSize:
                break
                   
            offset = offset + 30
            
            if offset > 2*windowSize:
                break
        
        if offset > 2*windowSize:
            continue
            
        medianFeat[i] = np.median(np.array(myList))
#        print(str(i) +"  " + str(medianFeat[i]))    
    return medianFeat        
    
    
def getStatePromoNumbers(trainFile,storeStates):

    trainStoreId = trainFile.Store.values 
    trainDate    = trainFile.Date.values
    trainPromo  = trainFile.Promo.values
    
    
    trainDayOfWeek = trainFile.DayOfWeek.values
    
    storePromoRun = {}
    storeNonPromoRun = {}
    storeTotal = {}
    
    statePromoFeat = np.ones((len(trainStoreId),5),dtype=float)*(-999)
    
    for i in range(0,len(trainStoreId)):  
    
        stateVal =  storeStates[trainStoreId[i]-1]
        
        if stateVal not in storePromoRun:
            storePromoRun[stateVal]  = {}
            storeNonPromoRun[stateVal]  = {} 
            storeTotal[stateVal]  = {} 
               
        map1 = storePromoRun[stateVal]
        map2 = storeNonPromoRun[stateVal]
        map3 = storeTotal[stateVal]
               
        if trainDate[i] not in map1:
            map1[trainDate[i]] = []    
            map2[trainDate[i]] = []  
            map3[trainDate[i]] = []  

        if trainPromo[i] == 0:
            map2[trainDate[i]].append(1)
        else:
            map1[trainDate[i]].append(1)
        
        map3[trainDate[i]].append(1)


    for i in range(0,len(trainStoreId)):  
    
        stateVal =  storeStates[trainStoreId[i]-1]
        
        if stateVal not in storePromoRun:
            continue
               
        map1 = storePromoRun[stateVal]
        map2 = storeNonPromoRun[stateVal]
        map3 = storeTotal[stateVal]
               
        if trainDate[i] not in map1:
            continue  

        numOfOpen = np.sum(np.array(map1[trainDate[i]]))
        numOfClosed = np.sum(np.array(map2[trainDate[i]]))
        numOfTotal = np.sum(np.array(map3[trainDate[i]]))
        ratio1 = float(numOfOpen/numOfTotal)
        
        if numOfOpen > 0:
            ratio2 = float(numOfClosed/numOfOpen)
        else:
            ratio2 = -999
       
        statePromoFeat[i,0] =  numOfOpen    
        statePromoFeat[i,1] =  numOfClosed 
        statePromoFeat[i,2] =  numOfTotal 
        statePromoFeat[i,3] =  ratio1 
        statePromoFeat[i,4] =  ratio2
       
       
    return statePromoFeat     
    
    
def getStatePromoConditions(trainFile,storeStates):

    trainStoreId = trainFile.Store.values 
    trainDate    = trainFile.Date.values
    trainPromo  = trainFile.Promo.values
    
    
    trainDayOfWeek = trainFile.DayOfWeek.values
    
    storePromoRun = {}
    
    storeIds = sorted(list(set(trainStoreId)))
    statePromoFeat = np.ones((len(trainStoreId),len(storeIds)),dtype=float)*(-999)
    
    for i in range(0,len(trainStoreId)):  
    
        stateVal =  storeStates[trainStoreId[i]-1]
        
        if stateVal not in storePromoRun:
            storePromoRun[stateVal]  = {}
               
        map1 = storePromoRun[stateVal]
               
        if trainDate[i] not in map1:
            map1[trainDate[i]] = np.ones((1,len(storeIds)),dtype=float)*(-999) 

        listVal = map1[trainDate[i]]
        
        if trainPromo[i] == 0:
            listVal[0,trainStoreId[i]-1] = 0
        else:
            listVal[0,trainStoreId[i]-1] = 1
        
        map1[trainDate[i]] = listVal


    for i in range(0,len(trainStoreId)):  
    
        stateVal =  storeStates[trainStoreId[i]-1]
        
        if stateVal not in storePromoRun:
            continue
               
        map1 = storePromoRun[stateVal]
               
        if trainDate[i] not in map1:
            continue  
            
        listVal = map1[trainDate[i]]
        
        statePromoFeat[i,:] = listVal[0,:]
        
        
       
    return statePromoFeat     
    

def examineTrends(trainFile,storeStates,trendFile):    

    startDateList, trendMap = getExternalData(trendFile)
    
    refDate = datetime(2012, 12, 31)   

            
    trainStoreId = trainFile.Store.values #key
    trainSales = trainFile.Sales.values
    trainDayOfWeek = trainFile.DayOfWeek.values #num(can be categorical)
    trainStateHoliday = trainFile.StateHoliday.values #categorical
    trainSchoolHoliday = trainFile.SchoolHoliday.values #categorical
    trainDate = trainFile.Date.values  #date
    trainPromo = trainFile.Promo.values  #date
    
    trainDateWeekNum  = np.zeros((len(trainDate),1),dtype= int) 
    trainDateDayNum = np.zeros((len(trainDate),1),dtype= int) 
    trainDateMonthNum = np.zeros((len(trainDate),1),dtype= int) 
    trainDateYearNum  = np.zeros((len(trainDate),1),dtype= int) 
    
        
    for i in range(0,len(trainDate)):
        if (pd.isnull(trainDate[i])):
            continue
        
        dateVal =  datetime.strptime(trainDate[i], '%Y-%m-%d')
        
        trainDateWeekNum[i] = dateVal.isocalendar()[1]
        trainDateDayNum[i] = dateVal.day     
        trainDateMonthNum[i] = dateVal.month
        trainDateYearNum[i] = dateVal.year
        
            
    myMap = {}
    dateMap = {}
    yearMap = {}
    promoMap = {}
    
    for i in range(0,len(trainStoreId)):
        key = trainStoreId[i]
        dayOfWeek = trainDayOfWeek[i]
        label = trainSales[i]
        
        
        if label < 1200 or trainDayOfWeek[i] > 5:
            continue
            
        if key not in myMap:
            myMap[key] = {}
            dateMap[key] = []
            yearMap[key] = []
            promoMap[key] = []
            

        dateVal =  datetime.strptime(trainDate[i], '%Y-%m-%d')
        
        diffDays = (dateVal-refDate).days 
                            
        dateKey = int(diffDays/7) 
        
        tempMap = myMap[key]
        
        if dateKey not in tempMap:
            myMap[key][dateKey] = [] 
            
            dateMap[key].append(trainDate[i])
            yearMap[key].append(trainDateYearNum[i])
            promoMap[key].append(trainPromo[i])
            
        myMap[key][dateKey].append(label)
        
        
    storeIds = sorted(list(set(trainStoreId)))
                                  
      
    for k in range(0,len(storeIds)):  
        storeKey =  storeIds[k]    
                             
        myListValMap = myMap[storeKey]
        myListDate = dateMap[storeKey]
        yearVal = yearMap[storeKey]
        promoVal = promoMap[storeKey]
        
        trData = []
        valData = []
        
        for i in range(0,len(myListDate)):
            dateVal =  datetime.strptime(myListDate[i], '%Y-%m-%d')

            diffDays = (dateVal-refDate).days 
                            
            dateKey = int(diffDays/7) 
        
            if dateKey < 120:
                continue
                    
            tr = getTrendsList(startDateList,trendMap,dateVal,0,storeStates[trainStoreId[i]-1])
            
            tr = tr + promoVal[i]*100
              
            valList = myListValMap[dateKey]
            valMean = np.mean(valList)
            
            trData.append(tr*0.5+1)
            valData.append(valMean*0.5+1)
                
        pList = pearsonr(np.log(valData),np.log(trData))      
        print(str(storeKey)+"  "+str(pList[0]))






def getOpenDays(trainFile,deltaVal):     

    trainStoreId = trainFile.Store.values 
    trainDate    = trainFile.Date.values
    trainOpen    = trainFile.Open.values
        
    trainDayOfWeek = trainFile.DayOfWeek.values
    
    refDate = datetime(2012, 12, 31)                
    
    storeMap = {}
    
    for i in range(0,len(trainStoreId)):   

        if  trainOpen[i] == 0 :
            continue
                    
        dateVal =  datetime.strptime(trainDate[i], '%Y-%m-%d')
        
        if trainStoreId[i] not in storeMap:
            storeMap[trainStoreId[i]]  = {}   
        
        diffDays = (dateVal-refDate).days 
                                
        dateKey = int(diffDays/7)                        
         
        myMap = storeMap[trainStoreId[i]]      
            
        if dateKey not in myMap:
            myMap[dateKey] = []     
                 
        myMap[dateKey].append(1)           


    diffFeat = np.ones((len(trainStoreId),1),dtype=float)*(-999)

    for i in range(0,len(trainStoreId)):            
        dateVal =  datetime.strptime(trainDate[i], '%Y-%m-%d')
        
        if trainStoreId[i] not in storeMap :
            diffFeat[i]  = -999   
            continue
            
        diffDays = (dateVal-refDate).days 
        dateKey = int(diffDays/7) - deltaVal                       
        myMap = storeMap[trainStoreId[i]]      
            
        if dateKey not in myMap:
            diffFeat[i]  = -999 
            continue    
                 
        openVals = myMap[dateKey]
        
        diffFeat[i]  =  np.sum(np.array(openVals))
        
#        if trainStoreId[i] == 2 and trainSales[i] != 0:
#            print(str(trainDayOfWeek[i]) + "  " + str(trainDate[i])+"  "+str(trainSales[i])+"  "+ str(diffFeat[i]))
        
    return diffFeat
    
    


def getClosedCondition(trainFile,deltaVal):     

    trainStoreId = trainFile.Store.values 
    trainDate    = trainFile.Date.values
    trainOpen    = trainFile.Open.values
        
    trainDayOfWeek = trainFile.DayOfWeek.values
    
    refDate = datetime(2012, 12, 31)          
    
    sundayStores = [85,122,209,259,262,274,299,310,335,353,423,433,453,494,512,524,530,562,578,676,682,732,733,769,863,867,877,931,948,1045,1081,1097,1099]      
    
    storeMap = {}
    
    for i in range(0,len(trainStoreId)):   
                    
        dateVal =  datetime.strptime(trainDate[i], '%Y-%m-%d')
        
        if trainStoreId[i] not in storeMap:
            storeMap[trainStoreId[i]]  = {}   
        
        diffDays = (dateVal-refDate).days 
                                
        dateKey = int(diffDays/7)                        
         
        myMap = storeMap[trainStoreId[i]]      
            
        if dateKey not in myMap:
            myMap[dateKey] = [0 , 0 , 0 ,0 ,0 , 0, 0]     
                 
        myList = myMap[dateKey]

        
        if trainDayOfWeek[i] != 7:        
            myList[trainDayOfWeek[i]-1] = trainOpen[i]
        elif trainStoreId[i] in sundayStores:  
            myList[trainDayOfWeek[i]-1] = trainOpen[i]
        else:
            myList[trainDayOfWeek[i]-1] = -999
            
        myMap[dateKey] = myList
      

    diffFeat = np.ones((len(trainStoreId),7),dtype=float)*(-999)

    for i in range(0,len(trainStoreId)):            
        dateVal =  datetime.strptime(trainDate[i], '%Y-%m-%d')
        
        if trainStoreId[i] not in storeMap :
            diffFeat[i,:]  = np.ones((1,7),dtype=float)*(-999)   
            continue
            
        diffDays = (dateVal-refDate).days 
        dateKey = int(diffDays/7) - deltaVal                       
        myMap = storeMap[trainStoreId[i]]      
            
        if dateKey not in myMap:
            diffFeat[i,:]  = np.ones((1,7),dtype=float)*(-999)
            continue    
                 
        diffFeat[i,:]  =  np.array(myMap[dateKey])
        
#        if trainStoreId[i] == 2 and trainSales[i] != 0:
#            print(str(trainDayOfWeek[i]) + "  " + str(trainDate[i])+"  "+str(trainSales[i])+"  "+ str(diffFeat[i]))
        
    return diffFeat
    
    
    

def getExt1Data(): 
    ext1File  = pd.read_csv('input/ext1.csv')
    return ext1File 
 
def getExt2Data(): 
    stateList = ['RH','BD','SCA','BY','TH','SH','HE','SC','NRW','BR','HM','BE']
    fileMap = {}
    
    for i in range(0,len(stateList)):
        myFile  = pd.read_csv('input/ext2'+str(stateList[i])+'.csv') 
        fileMap[stateList[i]] = myFile   
        
    return fileMap
         
 
def getExt3Data(): 
    ext3File  = pd.read_csv('input/ext3.csv')    
    return ext3File 
    
def getExt4Data(): 
    ext4File  = pd.read_csv('input/ext4.csv') 
    return ext4File 
               
      
      
def getWheatherData(): 
    stateList = ['RH','BD','SCA','BY','TH','SH','HE','SC','NRW','BR','HM','BE']
    fileMap = {}
    
    for i in range(0,len(stateList)):
        myFile  = pd.read_csv('wheather/'+str(stateList[i])+'.csv') 
        fileMap[stateList[i]] = myFile   
        
    return fileMap      
                
