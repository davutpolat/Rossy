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





def getCommonHolidays(dateStr):
    dateVal =  datetime.strptime(dateStr, '%Y-%m-%d')
    
    day    = dateVal.day     
    month  = dateVal.month
    year   = dateVal.year
    
    if year == 2013:
        return getCommonHolidays2013(dateVal,month,day)
    elif year == 2014:
        return getCommonHolidays2014(dateVal,month,day)
    elif year == 2015:
        return getCommonHolidays2015(dateVal,month,day)
        
                    
def getCommonHolidays2013(dateVal,month,day):    
 
    epiphanyDay     = datetime(2013, 1, 6)
    shroveMonday    = datetime(2013, 2, 11)
    shroveTuesday   = datetime(2013, 2, 12)
    ashWednesday    = datetime(2013, 2, 13)
    valentinesDay   = datetime(2013, 2, 14)
    womenCarnaval   = None  
    maundyThursday  = datetime(2013, 3, 28)
    goodFriday      = datetime(2013, 3, 29)
    easterSunday    = datetime(2013, 3, 31)
    easterMonday    = datetime(2013, 4, 1) 
    palmSunday      = datetime(2013, 3, 24)
    labourDay       = datetime(2013, 5, 1)
    ascensionDay    = datetime(2013, 5, 9)
    europeDay       = None
    mothersDay      = datetime(2013, 5, 12)
    whitsun         = datetime(2013, 5, 19)
    whitMonday      = datetime(2013, 5, 24)
    corpus          = datetime(2013, 5, 30)
    ausburg         = datetime(2013, 8, 8)
    assumptionDay   = datetime(2013, 8, 15)
    reformationDay  = datetime(2013, 10, 31)
    allSaintsDay    = datetime(2013, 11, 1)
    germanUnity     = datetime(2013, 10, 3) 
    thanksGiving    = datetime(2013, 10, 6) 
    remembarenceDay = datetime(2013, 11, 17)
    dayOfPrayer     = datetime(2013, 11, 20)
    deadSunday      = datetime(2013, 11, 24)
    firstAdvent     = datetime(2013, 12, 1)
    saintNicholas   = datetime(2013, 12, 6)
    secondAdvent    = datetime(2013, 12, 8)
    thirdAdvent     = datetime(2013, 12, 15)
    forthAdvent     = datetime(2013, 12, 22)
    octoberfestStart  = datetime(2013, 9, 21)
    octoberfestEnd    = datetime(2013, 10, 6)
    ramadanStart   = datetime(2013, 8, 8)
    
    epiphanyDayFeat     = -999
    shroveMondayFeat    = -999
    shroveTuesdayFeat   = -999
    ashWednesdayFeat    = -999
    valentinesDayFeat   = -999  
    womenCarnavalFeat   = -999  
    maundyThursdayFeat  = -999
    goodFridayFeat      = -999
    easterSundayFeat    = -999
    easterMondayFeat    = -999
    palmSundayFeat      = -999
    ascensionDayFeat    = -999
    mothersDayFeat      = -999
    whitsunFeat         = -999
    whitMondayFeat      = -999 
    corpusFeat          = -999     
    ausburgFeat         = -999
    assumptionDayFeat   = -999
    reformationDayFeat  = -999 
    allSaintsDayFeat    = -999
    germanUnityFeat     = -999
    thanksGivingFeat    = -999
    remembarenceDayFeat = -999
    dayOfPrayerFeat     = -999
    deadSundayFeat      = -999
    firstAdventFeat     = -999 
    saintNicholasFeat   = -999
    secondAdventFeat    = -999
    thirdAdventFeat     = -999
    forthAdventFeat     = -999
    europeDayFeat       = -999
    labourDayFeat       = -999
    octoberStartDiff    = -999  
    octoberEndDiff      = -999 
    octoberRatio        = -999 
    ramadanStartDiff    = -999
        
    if month == 1 and day > 3 and day < 8:
        epiphanyDayFeat = (dateVal-epiphanyDay).days 
    
    if month == 2 and day > 10 and day < 18:
        shroveMondayFeat = (dateVal-shroveMonday).days  
        shroveTuesdayFeat = (dateVal-shroveTuesday).days 
        ashWednesdayFeat = (dateVal-ashWednesday).days 
        valentinesDayFeat = (dateVal-valentinesDay).days 
    
    if month == 3 and day > 22 and day < 26:
        palmSundayFeat = (dateVal-palmSunday).days 
        
        
    if (month == 3 and day > 24) or (month == 5 and day < 6):
        easterMondayFeat = (dateVal-easterMonday).days
        goodFridayFeat = (dateVal-goodFriday).days 
        maundyThursdayFeat = (dateVal-maundyThursday).days 
        easterSundayFeat = (dateVal-easterSunday).days 
        
          
    if (month == 4 and day > 27 ) or (month == 5 and day < 4):
        labourDayFeat = (dateVal-labourDay).days
               
    if month == 5 and day > 5 and day < 13:
        ascensionDayFeat = (dateVal-ascensionDay).days 
        mothersDayFeat = (dateVal-mothersDay).days 
    
    if month == 5 and day > 16 and day < 22:
        whitsunFeat         = (dateVal-whitsun).days
        whitMondayFeat      = (dateVal-whitMonday).days     

    if month == 5 and day > 26 :
        corpusFeat = (dateVal-corpus).days
    
    if month == 8 and day > 4 and day < 12:
        ausburgFeat = (dateVal-ausburg).days
        
    if month == 7 or  month == 8:
        ramadanStartDiff = (dateVal-ramadanStart).days        
    
    if month == 8 and day > 11 and day < 19:
        assumptionDayFeat = (dateVal-assumptionDay).days
    
    if (month == 10 and day > 27) or (month == 11 and day < 4):
        reformationDayFeat = (dateVal-reformationDay).days 
        allSaintsDayFeat = (dateVal-allSaintsDay).days
    
    if month == 10 and day < 5:
        germanUnityFeat = (dateVal-germanUnity).days 
    
    if month == 10 and day >  3 and day < 8:
        thanksGivingFeat = (dateVal-thanksGiving).days 
    
    if month == 11 and day > 14 and day < 19:
        remembarenceDayFeat = (dateVal-remembarenceDay).days 
    
    if month == 11 and day > 17 and day < 23:
        dayOfPrayerFeat = (dateVal-dayOfPrayer).days
        
    if month == 11 and day > 21 and day < 26:
        deadSundayFeat = (dateVal-deadSunday).days
    
    if  (month == 11 and day > 28) or (month == 12 and day < 3):
        firstAdventFeat = (dateVal-firstAdvent).days
    
    if month == 12 and day > 4 and day < 9:    
        saintNicholasFeat = (dateVal-saintNicholas).days
    
    if month == 12 and day > 5 and day < 10:    
        secondAdventFeat = (dateVal-secondAdvent).days    
    
    if month == 12 and day > 12 and day < 17:    
        thirdAdventFeat = (dateVal-thirdAdvent).days
        
    if month == 12 and day > 19 and day < 24:    
        forthAdventFeat = (dateVal-forthAdvent).days        
            
    if (month == 9 and day > 8) or (month == 10 and day < 7):
        octoberStartDiff = (dateVal-octoberfestStart).days     
        octoberEndDiff   = (dateVal-octoberfestEnd).days
        easterLength    = 16
        if octoberEndDiff*octoberStartDiff <= 0:
            octoberRatio    = octoberStartDiff/easterLength 
                 
    retVal = np.zeros((1,36),dtype=float)
    
    
    retVal[0,0] = epiphanyDayFeat     
    retVal[0,1] = shroveMondayFeat    
    retVal[0,2] = shroveTuesdayFeat   
    retVal[0,3] = ashWednesdayFeat    
    retVal[0,4] = valentinesDayFeat     
    retVal[0,5] = womenCarnavalFeat     
    retVal[0,6] = maundyThursdayFeat  
    retVal[0,7] = goodFridayFeat      
    retVal[0,8] = easterSundayFeat    
    retVal[0,9] = easterMondayFeat    
    retVal[0,10] = palmSundayFeat      
    retVal[0,11] = ascensionDayFeat    
    retVal[0,12] = mothersDayFeat      
    retVal[0,13] = whitsunFeat         
    retVal[0,14] = whitMondayFeat       
    retVal[0,15] = corpusFeat               
    retVal[0,16] = ausburgFeat         
    retVal[0,17] = assumptionDayFeat   
    retVal[0,18] = reformationDayFeat   
    retVal[0,19] = allSaintsDayFeat    
    retVal[0,20] = germanUnityFeat     
    retVal[0,21] = thanksGivingFeat    
    retVal[0,22] = remembarenceDayFeat 
    retVal[0,23] = dayOfPrayerFeat     
    retVal[0,24] = deadSundayFeat      
    retVal[0,25] = firstAdventFeat     
    retVal[0,26] = saintNicholasFeat   
    retVal[0,27] = secondAdventFeat    
    retVal[0,28] = thirdAdventFeat     
    retVal[0,29] = forthAdventFeat     
    retVal[0,30] = europeDayFeat
    retVal[0,31] = labourDayFeat
    retVal[0,32] = octoberStartDiff
    retVal[0,33] = octoberEndDiff
    retVal[0,34] = octoberRatio
    retVal[0,35] = ramadanStartDiff
    
    return retVal  
    
    
            

def getCommonHolidays2014(dateVal,month,day):    
 
    epiphanyDay     = datetime(2014, 1, 6) 
    shroveMonday    = datetime(2014, 3, 3) 
    shroveTuesday   = datetime(2014, 3, 4) 
    ashWednesday    = datetime(2014, 3, 5) 
    valentinesDay   = datetime(2014, 2, 14) 
    womenCarnaval   = datetime(2014, 2, 27) 
    maundyThursday  = datetime(2014, 4, 17)
    goodFriday      = datetime(2014, 4, 18)
    easterSunday    = datetime(2014, 4, 20) 
    easterMonday    = datetime(2014, 4, 21)  
    palmSunday      = datetime(2014, 4, 13)
    labourDay       = datetime(2014, 5, 1) 
    europeDay       = datetime(2014, 5, 9) 
    ascensionDay    = datetime(2014, 5, 29) 
    mothersDay      = datetime(2014, 5, 11) 
    whitsun         = datetime(2014, 6, 8) 
    whitMonday      = datetime(2014, 6, 9)  
    corpus          = datetime(2014, 6, 19)
    ausburg         = datetime(2014, 8, 8) 
    assumptionDay   = datetime(2014, 8, 15)
    reformationDay  = datetime(2014, 10, 31) 
    allSaintsDay    = datetime(2014, 11, 1) 
    germanUnity     = datetime(2014, 10, 3) 
    thanksGiving    = datetime(2014, 10, 5) 
    remembarenceDay = datetime(2014, 11, 16)
    dayOfPrayer     = datetime(2014, 11, 19)
    deadSunday      = datetime(2014, 11, 23)
    firstAdvent     = datetime(2014, 11, 30)
    saintNicholas   = datetime(2014, 12, 6)
    secondAdvent    = datetime(2014, 12, 7)
    thirdAdvent     = datetime(2014, 12, 14)
    forthAdvent     = datetime(2014, 12, 21)
    octoberfestStart  = datetime(2014, 9, 20)
    octoberfestEnd    = datetime(2014, 10, 5)    
    ramadanStart   = datetime(2014, 7, 28)
        
    epiphanyDayFeat     = -999
    shroveMondayFeat    = -999
    shroveTuesdayFeat   = -999
    ashWednesdayFeat    = -999
    valentinesDayFeat   = -999  
    womenCarnavalFeat   = -999  
    maundyThursdayFeat  = -999
    goodFridayFeat      = -999
    easterSundayFeat    = -999
    easterMondayFeat    = -999
    palmSundayFeat      = -999
    ascensionDayFeat    = -999
    mothersDayFeat      = -999
    whitsunFeat         = -999
    whitMondayFeat      = -999 
    corpusFeat          = -999     
    ausburgFeat         = -999
    assumptionDayFeat   = -999
    reformationDayFeat  = -999 
    allSaintsDayFeat    = -999
    germanUnityFeat     = -999
    thanksGivingFeat    = -999
    remembarenceDayFeat = -999
    dayOfPrayerFeat     = -999
    deadSundayFeat      = -999
    firstAdventFeat     = -999 
    saintNicholasFeat   = -999
    secondAdventFeat    = -999
    thirdAdventFeat     = -999
    forthAdventFeat     = -999
    europeDayFeat       = -999
    labourDayFeat       = -999
    octoberStartDiff    = -999  
    octoberEndDiff      = -999 
    octoberRatio        = -999 
    ramadanStartDiff    = -999
            
    if month == 1 and day > 3 and day < 8:
        epiphanyDayFeat = (dateVal-epiphanyDay).days 
    
    if month == 2 and day > 9 and day < 17:
        valentinesDayFeat = (dateVal-valentinesDay).days 
    
    if month == 2 and day > 23:
        womenCarnavalFeat =  (dateVal-womenCarnaval).days  
        
    if month == 3 and  day < 8:
        shroveMondayFeat = (dateVal-shroveMonday).days  
        shroveTuesdayFeat = (dateVal-shroveTuesday).days 
        ashWednesdayFeat = (dateVal-ashWednesday).days 

    if month == 4 and day > 10 and day < 15:
        palmSundayFeat = (dateVal-palmSunday).days 
        
    if  month == 4 and day > 13 and day < 25 :
        easterMondayFeat = (dateVal-easterMonday).days
        goodFridayFeat = (dateVal-goodFriday).days 
        maundyThursdayFeat = (dateVal-maundyThursday).days 
        easterSundayFeat = (dateVal-easterSunday).days
        
    if (month == 4 and day > 27 ) or (month == 5 and day < 4):
        labourDayFeat = (dateVal-labourDay).days        
          
    if month == 5 and day > 5 and day < 13:
        europeDayFeat = (dateVal-europeDay).days 
        mothersDayFeat = (dateVal-mothersDay).days 
    
    if month == 5 and day > 26 :
        ascensionDayFeat = (dateVal-ascensionDay).days
    
    if month == 6 and day > 6 and day < 14:
        whitsunFeat         = (dateVal-whitsun).days
        whitMondayFeat      = (dateVal-whitMonday).days     

    if month == 6 and day > 17 and day < 22 :
        corpusFeat = (dateVal-corpus).days
        
    if month == 8 and day > 4 and day < 12:
        ausburgFeat = (dateVal-ausburg).days
    
    if (month == 6 and day > 20) or month == 7 or  (month == 8 and day < 20):
        ramadanStartDiff = (dateVal-ramadanStart).days 
            
    if month == 8 and day > 11 and day < 19:
        assumptionDayFeat = (dateVal-assumptionDay).days
    
    if (month == 10 and day > 27) or (month == 11 and day < 4):
        reformationDayFeat = (dateVal-reformationDay).days 
        allSaintsDayFeat = (dateVal-allSaintsDay).days
    
    if month == 10 and day < 5:
        germanUnityFeat = (dateVal-germanUnity).days 
    
    if month == 10 and day >  2 and day < 7:
        thanksGivingFeat = (dateVal-thanksGiving).days 
    
    if month == 11 and day > 13 and day < 18:
        remembarenceDayFeat = (dateVal-remembarenceDay).days 
    
    if month == 11 and day > 16 and day < 22:
        dayOfPrayerFeat = (dateVal-dayOfPrayer).days
        
    if month == 11 and day > 20 and day < 25:
        deadSundayFeat = (dateVal-deadSunday).days
    
    if  (month == 11 and day > 27) or (month == 12 and day < 2):
        firstAdventFeat = (dateVal-firstAdvent).days
    
    if month == 12 and day > 4 and day < 8:    
        saintNicholasFeat = (dateVal-saintNicholas).days
    
    if month == 12 and day > 4 and day < 9:    
        secondAdventFeat = (dateVal-secondAdvent).days    
    
    if month == 12 and day > 11 and day < 16:    
        thirdAdventFeat = (dateVal-thirdAdvent).days
        
    if month == 12 and day > 18 and day < 23:    
        forthAdventFeat = (dateVal-forthAdvent).days        
            

    if (month == 9 and day > 7) or (month == 10 and day < 6):
        octoberStartDiff = (dateVal-octoberfestStart).days     
        octoberEndDiff   = (dateVal-octoberfestEnd).days
        easterLength    = 16
        if octoberEndDiff*octoberStartDiff <= 0:
            octoberRatio    = octoberStartDiff/easterLength 
                        
    retVal = np.zeros((1,36),dtype=float)
    
    
    retVal[0,0] = epiphanyDayFeat     
    retVal[0,1] = shroveMondayFeat    
    retVal[0,2] = shroveTuesdayFeat   
    retVal[0,3] = ashWednesdayFeat    
    retVal[0,4] = valentinesDayFeat     
    retVal[0,5] = womenCarnavalFeat     
    retVal[0,6] = maundyThursdayFeat  
    retVal[0,7] = goodFridayFeat      
    retVal[0,8] = easterSundayFeat    
    retVal[0,9] = easterMondayFeat    
    retVal[0,10] = palmSundayFeat      
    retVal[0,11] = ascensionDayFeat    
    retVal[0,12] = mothersDayFeat      
    retVal[0,13] = whitsunFeat         
    retVal[0,14] = whitMondayFeat       
    retVal[0,15] = corpusFeat               
    retVal[0,16] = ausburgFeat         
    retVal[0,17] = assumptionDayFeat   
    retVal[0,18] = reformationDayFeat   
    retVal[0,19] = allSaintsDayFeat    
    retVal[0,20] = germanUnityFeat     
    retVal[0,21] = thanksGivingFeat    
    retVal[0,22] = remembarenceDayFeat 
    retVal[0,23] = dayOfPrayerFeat     
    retVal[0,24] = deadSundayFeat      
    retVal[0,25] = firstAdventFeat     
    retVal[0,26] = saintNicholasFeat   
    retVal[0,27] = secondAdventFeat    
    retVal[0,28] = thirdAdventFeat     
    retVal[0,29] = forthAdventFeat     
    retVal[0,30] = europeDayFeat
    retVal[0,31] = labourDayFeat
    retVal[0,32] = octoberStartDiff
    retVal[0,33] = octoberEndDiff
    retVal[0,34] = octoberRatio
    retVal[0,35] = ramadanStartDiff
        
    return retVal  
    
            
def getCommonHolidays2015(dateVal,month,day):    
 
    epiphanyDay     = datetime(2015, 1, 6) 
    shroveMonday    = datetime(2015, 2, 16) 
    shroveTuesday   = datetime(2015, 2, 17) 
    ashWednesday    = datetime(2015, 2, 18) 
    valentinesDay   = datetime(2015, 2, 14) 
    womenCarnaval   = datetime(2015, 2, 12) 
    maundyThursday  = datetime(2015, 4, 2) 
    goodFriday      = datetime(2015, 4, 3)
    easterSunday    = datetime(2015, 4, 5) 
    easterMonday    = datetime(2015, 4, 6)  
    palmSunday      = datetime(2015, 3, 29) 
    labourDay       = datetime(2015, 5, 1) 
    europeDay       = datetime(2015, 5, 9) 
    ascensionDay    = datetime(2015, 5, 14)  
    mothersDay      = datetime(2015, 5, 10)  
    whitsun         = datetime(2015, 5, 24) 
    whitMonday      = datetime(2015, 5, 25)   
    corpus          = datetime(2015, 6, 4) 
    ausburg         = datetime(2015, 8, 8) 
    assumptionDay   = datetime(2015, 8, 15)
    reformationDay  = datetime(2015, 10, 31)  
    allSaintsDay    = datetime(2015, 11, 1) 
    germanUnity     = datetime(2015, 10, 3) 
    thanksGiving    = datetime(2015, 10, 4) 
    remembarenceDay = datetime(2015, 11, 15) 
    dayOfPrayer     = datetime(2015, 11, 18)
    deadSunday      = datetime(2015, 11, 22) 
    firstAdvent     = datetime(2015, 11, 29) 
    saintNicholas   = datetime(2015, 12, 6) 
    secondAdvent    = None
    thirdAdvent     = datetime(2015, 12, 13)
    forthAdvent     = datetime(2015, 12, 20)
    octoberfestStart  = datetime(2015, 9, 19)
    octoberfestEnd    = datetime(2015, 10, 4)    
    ramadanStart   = datetime(2015, 7, 17)
    
    
    epiphanyDayFeat     = -999
    shroveMondayFeat    = -999
    shroveTuesdayFeat   = -999
    ashWednesdayFeat    = -999
    valentinesDayFeat   = -999  
    womenCarnavalFeat   = -999  
    maundyThursdayFeat  = -999
    goodFridayFeat      = -999
    easterSundayFeat    = -999
    easterMondayFeat    = -999
    palmSundayFeat      = -999
    ascensionDayFeat    = -999
    mothersDayFeat      = -999
    whitsunFeat         = -999
    whitMondayFeat      = -999 
    corpusFeat          = -999     
    ausburgFeat         = -999
    assumptionDayFeat   = -999
    reformationDayFeat  = -999 
    allSaintsDayFeat    = -999
    germanUnityFeat     = -999
    thanksGivingFeat    = -999
    remembarenceDayFeat = -999
    dayOfPrayerFeat     = -999
    deadSundayFeat      = -999
    firstAdventFeat     = -999 
    saintNicholasFeat   = -999
    secondAdventFeat    = -999
    thirdAdventFeat     = -999
    forthAdventFeat     = -999
    europeDayFeat       = -999
    labourDayFeat       = -999
    octoberStartDiff    = -999  
    octoberEndDiff      = -999 
    octoberRatio        = -999      
    ramadanStartDiff    = -999
    
    if month == 1 and day > 4 and day < 9:
        epiphanyDayFeat = (dateVal-epiphanyDay).days 
    
    if month == 2 and day > 8 and day < 17:
        valentinesDayFeat = (dateVal-valentinesDay).days 
        womenCarnavalFeat =  (dateVal-womenCarnaval).days 
    
    if month == 2 and  day < 23 and day > 12:
        shroveMondayFeat = (dateVal-shroveMonday).days  
        shroveTuesdayFeat = (dateVal-shroveTuesday).days 
        ashWednesdayFeat = (dateVal-ashWednesday).days 

    if month == 3 and day > 26 :
        palmSundayFeat = (dateVal-palmSunday).days 
        
    if  (month == 3 and day > 29 ) or (month == 4 and  day < 8) :
        easterMondayFeat = (dateVal-easterMonday).days
        goodFridayFeat = (dateVal-goodFriday).days 
        maundyThursdayFeat = (dateVal-maundyThursday).days 
        easterSundayFeat = (dateVal-easterSunday).days
        
    if (month == 4 and day > 27 ) or (month == 5 and day < 5):
        labourDayFeat = (dateVal-labourDay).days        
          
    if month == 5 and day > 7 and day < 12:
        europeDayFeat = (dateVal-europeDay).days 
        mothersDayFeat = (dateVal-mothersDay).days 

    if month == 5 and day > 12 and day < 18:
        ascensionDayFeat = (dateVal-ascensionDay).days
    
    if month == 5 and day > 22 and day < 28:
        whitsunFeat         = (dateVal-whitsun).days
        whitMondayFeat      = (dateVal-whitMonday).days     

    if month == 6 and  day < 8 :
        corpusFeat = (dateVal-corpus).days
        
    if month == 8 and day > 6 and day < 10:
        ausburgFeat = (dateVal-ausburg).days

    if (month == 6 and day > 10) or month == 7 or  (month == 8 and day < 15):
        ramadanStartDiff = (dateVal-ramadanStart).days 
        
    if month == 8 and day > 13 and day < 18:
        assumptionDayFeat = (dateVal-assumptionDay).days
    
    if (month == 10 and day > 29) or (month == 11 and day < 3):
        reformationDayFeat = (dateVal-reformationDay).days 
        allSaintsDayFeat = (dateVal-allSaintsDay).days
    
    if month == 10 and day < 6:
        germanUnityFeat = (dateVal-germanUnity).days 
        thanksGivingFeat = (dateVal-thanksGiving).days 
    
    if month == 11 and day > 12 and day < 17:
        remembarenceDayFeat = (dateVal-remembarenceDay).days 
    
    if month == 11 and day > 15 and day < 21:
        dayOfPrayerFeat = (dateVal-dayOfPrayer).days
        
    if month == 11 and day > 19 and day < 24:
        deadSundayFeat = (dateVal-deadSunday).days
    
    if  month == 11 and day > 26:
        firstAdventFeat = (dateVal-firstAdvent).days
    
    if month == 12 and day > 3 and day < 8:    
        saintNicholasFeat = (dateVal-saintNicholas).days
    
    if month == 12 and day > 10 and day < 15:    
        thirdAdventFeat = (dateVal-thirdAdvent).days
        
    if month == 12 and day > 17 and day < 22:    
        forthAdventFeat = (dateVal-forthAdvent).days        

    if (month == 9 and day > 6) or (month == 10 and day < 5):
        octoberStartDiff = (dateVal-octoberfestStart).days     
        octoberEndDiff   = (dateVal-octoberfestEnd).days
        easterLength    = 16
        if octoberEndDiff*octoberStartDiff <= 0:
            octoberRatio    = octoberStartDiff/easterLength             
            
    retVal = np.zeros((1,36),dtype=float)
    
    
    retVal[0,0] = epiphanyDayFeat     
    retVal[0,1] = shroveMondayFeat    
    retVal[0,2] = shroveTuesdayFeat   
    retVal[0,3] = ashWednesdayFeat    
    retVal[0,4] = valentinesDayFeat     
    retVal[0,5] = womenCarnavalFeat     
    retVal[0,6] = maundyThursdayFeat  
    retVal[0,7] = goodFridayFeat      
    retVal[0,8] = easterSundayFeat    
    retVal[0,9] = easterMondayFeat    
    retVal[0,10] = palmSundayFeat      
    retVal[0,11] = ascensionDayFeat    
    retVal[0,12] = mothersDayFeat      
    retVal[0,13] = whitsunFeat         
    retVal[0,14] = whitMondayFeat       
    retVal[0,15] = corpusFeat               
    retVal[0,16] = ausburgFeat         
    retVal[0,17] = assumptionDayFeat   
    retVal[0,18] = reformationDayFeat   
    retVal[0,19] = allSaintsDayFeat    
    retVal[0,20] = germanUnityFeat     
    retVal[0,21] = thanksGivingFeat    
    retVal[0,22] = remembarenceDayFeat 
    retVal[0,23] = dayOfPrayerFeat     
    retVal[0,24] = deadSundayFeat      
    retVal[0,25] = firstAdventFeat     
    retVal[0,26] = saintNicholasFeat   
    retVal[0,27] = secondAdventFeat    
    retVal[0,28] = thirdAdventFeat     
    retVal[0,29] = forthAdventFeat     
    retVal[0,30] = europeDayFeat
    retVal[0,31] = labourDayFeat
    retVal[0,32] = octoberStartDiff
    retVal[0,33] = octoberEndDiff
    retVal[0,34] = octoberRatio
    retVal[0,35] = ramadanStartDiff
        
    return retVal  
    
    
    
    

def getSchoolHolidayFeats(dateStr, state):
    dateVal =  datetime.strptime(dateStr, '%Y-%m-%d')
    
    day    = dateVal.day     
    month  = dateVal.month
    year   = dateVal.year
    
    
    
    if state == 'BD':
        if year == 2013:
            return getBaden2013(dateVal,month,day)
        elif year == 2014:
            return getBaden2014(dateVal,month,day)        
        elif year == 2015:
            return getBaden2015(dateVal,month,day)    
    elif state == 'BE':
        if year == 2013:
            return getBerlin2013(dateVal,month,day)
        elif year == 2014:
            return getBerlin2014(dateVal,month,day)        
        elif year == 2015: 
            return getBerlin2015(dateVal,month,day)    
    elif state == 'BY':
        if year == 2013:
            return getBayern2013(dateVal,month,day)
        elif year == 2014:
            return getBayern2014(dateVal,month,day)        
        elif year == 2015: 
            return getBayern2015(dateVal,month,day)           
    elif state == 'BR':
        if year == 2013:
            return getBremen2013(dateVal,month,day)
        elif year == 2014:
            return getBremen2014(dateVal,month,day)        
        elif year == 2015: 
            return getBremen2015(dateVal,month,day)   
    elif state == 'HE':
        if year == 2013:
            return getHessen2013(dateVal,month,day)
        elif year == 2014:
            return getHessen2014(dateVal,month,day)        
        elif year == 2015: 
            return getHessen2015(dateVal,month,day)
    elif state == 'HM':
        if year == 2013:
            return getHamburg2013(dateVal,month,day)
        elif year == 2014:
            return getHamburg2014(dateVal,month,day)        
        elif year == 2015: 
            return getHamburg2015(dateVal,month,day)   
    elif state == 'NRW':
        if year == 2013:
            return getNRW2013(dateVal,month,day)
        elif year == 2014:
            return getNRW2014(dateVal,month,day)        
        elif year == 2015: 
            return getNRW2015(dateVal,month,day)   
    elif state == 'RH':
        if year == 2013:
            return getRheinland2013(dateVal,month,day)
        elif year == 2014:
            return getRheinland2014(dateVal,month,day)        
        elif year == 2015: 
            return getRheinland2015(dateVal,month,day)    
    elif state == 'SC':
        if year == 2013:
            return getSachsen2013(dateVal,month,day)
        elif year == 2014:
            return getSachsen2014(dateVal,month,day)        
        elif year == 2015: 
            return getSachsen2015(dateVal,month,day)    
    elif state == 'SCA':
        if year == 2013:
            return getSachsenAnhalt2013(dateVal,month,day)
        elif year == 2014:
            return getSachsenAnhalt2014(dateVal,month,day)        
        elif year == 2015: 
            return getSachsenAnhalt2015(dateVal,month,day)   
    elif state == 'SH':
        if year == 2013:
            return getScheleswig2013(dateVal,month,day)
        elif year == 2014:
            return getScheleswig2014(dateVal,month,day)        
        elif year == 2015: 
            return getScheleswig2015(dateVal,month,day)     
    elif state == 'TH':
        if year == 2013:
            return getThuringen2013(dateVal,month,day)
        elif year == 2014:
            return getThuringen2014(dateVal,month,day)        
        elif year == 2015: 
            return getThuringen2015(dateVal,month,day)    
    
    return None


############## BADEN SCHOOL VACATIONS ####################
def getBaden2013(dateVal,month,day):

    chrisStart      = datetime(2012, 12, 24)
    chrisEnd        = datetime(2013, 1, 5)  
    winterStart     = None 
    winterEnd       = None
    easterStart     = datetime(2013, 3, 25)
    easterEnd       = datetime(2013, 4, 5)
    witsun1Start    = datetime(2013, 5, 21)
    witsun1End      = datetime(2013, 6, 1)
    witsun2Start    = datetime(2013, 5, 21)
    witsun2End      = datetime(2013, 6, 1)
    summerStart     = datetime(2013, 7, 25)
    summerEnd       = datetime(2013, 9, 7)
    autumnStart     = datetime(2013, 10, 28)
    autumnEnd       = datetime(2013, 10, 30)
    nextChrisStart  = datetime(2013, 12, 23)
    nextChrisEnd    = datetime(2014, 1, 4) 
    
     
    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 13:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 13
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength
              
    if (day > 17 and month == 3) or (month == 4 and day < 10):
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 12
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if (month == 5 and day > 13) or (month == 6 and day < 5): 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 12
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if (month == 5 and day > 13) or (month == 6 and day < 5):  
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 12
        witsunCount      = 1
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (day > 17 and month == 7) or month == 8 or (month == 9 and  day < 15): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 45 
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if (day > 20 and month == 10) or (day < 3 and month == 11): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 3
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if day > 15 and month == 12: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 13
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal
    
            
def getBaden2014(dateVal,month,day):

    chrisStart    = datetime(2013, 12, 23)
    chrisEnd      = datetime(2014, 1, 4)  
    winterStart   = None 
    winterEnd     = None
    easterStart   = datetime(2014, 4, 14)
    easterEnd     = datetime(2014, 4, 25)
    witsun1Start  = datetime(2014, 6, 10)
    witsun1End    = datetime(2014, 6, 21)
    witsun2Start  = datetime(2014, 6, 10)
    witsun2End    = datetime(2014, 6, 21)
    summerStart   = datetime(2014, 7, 31)
    summerEnd     = datetime(2014, 9, 13)
    autumnStart   = datetime(2014, 10, 27)
    autumnEnd     = datetime(2014, 10, 30)
    nextChrisStart  = datetime(2014, 12, 22)
    nextChrisEnd    = datetime(2015, 1, 5)     

    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 12:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 13
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength
              
    if  month == 4 and day > 6  :
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 12
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if month == 6 and day > 2 and day < 26 : 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 12
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if month == 6 and day > 2 and day < 26 : 
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 12
        witsunCount      = 1
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (month == 7 and day > 23) or month == 8 or (month == 9 and day < 21): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 45 
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if ( day > 19 and month == 10) or ( month == 11 and day < 7): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 4
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if month == 12 and day > 14: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 15
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal    
    
def getBaden2015(dateVal,month,day):

    chrisStart    = datetime(2014, 12, 22)
    chrisEnd      = datetime(2015, 1, 5)  
    winterStart   = None 
    winterEnd     = None
    easterStart   = datetime(2015, 3, 30)
    easterEnd     = datetime(2015, 4, 12)
    witsun1Start  = datetime(2015, 5, 26)
    witsun1End    = datetime(2015, 6, 6)
    witsun2Start  = datetime(2015, 5, 26)
    witsun2End    = datetime(2015, 6, 6)
    summerStart   = datetime(2015, 7, 30)
    summerEnd     = datetime(2015, 9, 12)
    autumnStart   = datetime(2015, 11, 2)
    autumnEnd     = datetime(2015, 11, 6)
    nextChrisStart  = datetime(2015, 12, 23)
    nextChrisEnd    = datetime(2016, 1, 9)     



    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 13:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 15
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength
              
    if (month == 3 and day > 22) or (month == 4 and day < 20):
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 12
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if (month == 5 and day > 18 ) or (month == 6 and day < 14): 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 12
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if (month == 5 and day > 18 ) or (month == 6 and day < 14): 
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 12
        witsunCount      = 1
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (month == 7  and day > 22 )or month == 8 or (month == 9 and day < 20): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 45 
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if (month == 10 and day > 25)or (month == 11 and day < 14): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 5
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if month == 12 and day > 15: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 18
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal
    
############## BAYERN SCHOOL VACATIONS ####################
def getBayern2013(dateVal,month,day):

    chrisStart      = datetime(2012, 12, 24)
    chrisEnd        = datetime(2013, 1, 5)  
    winterStart     = datetime(2013, 2, 11) 
    winterEnd       = datetime(2013, 2, 15)
    easterStart     = datetime(2013, 3, 25)
    easterEnd       = datetime(2013, 4, 6)
    witsun1Start    = datetime(2013, 5, 21)
    witsun1End      = datetime(2013, 5, 31)
    witsun2Start    = datetime(2013, 5, 21)
    witsun2End      = datetime(2013, 5, 31)
    summerStart     = datetime(2013, 7, 31)
    summerEnd       = datetime(2013, 9, 11)
    autumnStart     = datetime(2013, 10, 28)
    autumnEnd       = datetime(2013, 10, 31)
    nextChrisStart  = datetime(2013, 12, 23)
    nextChrisEnd    = datetime(2014, 1, 4) 
    
     
    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 13:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 13
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength

    if month == 2 and day < 23 and day > 3:
        winterStartDiff = (dateVal-winterStart).days     
        winterEndDiff   = (dateVal-winterStart).days
        winterLength     = 5
        if winterStartDiff*winterEndDiff <= 0:
            winterRatio    = winterStartDiff/winterLength
            
                          
    if (day > 17 and month == 3) or (month == 4 and day < 14):
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 13
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if (month == 5 and day > 13) or (month == 6 and day < 8): 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 11
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if (month == 5 and day > 13) or (month == 6 and day < 8):  
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 11
        witsunCount      = 1
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (day > 23 and month == 7) or month == 8 or (month == 9 and  day < 19): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 43 
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if (day > 20 and month == 10) or (day < 8 and month == 11): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 4
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if day > 15 and month == 12: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 13
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal
    
            
def getBayern2014(dateVal,month,day):

    chrisStart    = datetime(2013, 12, 23)
    chrisEnd      = datetime(2014, 1, 4)  
    winterStart   = datetime(2014, 3, 3) 
    winterEnd     = datetime(2014, 3, 7)
    easterStart   = datetime(2014, 4, 14)
    easterEnd     = datetime(2014, 4, 26)
    witsun1Start  = datetime(2014, 6, 10)
    witsun1End    = datetime(2014, 6, 21)
    witsun2Start  = datetime(2014, 6, 10)
    witsun2End    = datetime(2014, 6, 21)
    summerStart   = datetime(2014, 7, 30)
    summerEnd     = datetime(2014, 9, 15)
    autumnStart   = datetime(2014, 10, 27)
    autumnEnd     = datetime(2014, 10, 31)
    nextChrisStart  = datetime(2014, 12, 24)
    nextChrisEnd    = datetime(2015, 1, 5)     

    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 12:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 13
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength
            
    if (month == 2 and day > 23) or ( month == 3 and day < 15):
        winterStartDiff = (dateVal-winterStart).days     
        winterEndDiff   = (dateVal-winterStart).days
        winterLength     = 5
        if winterStartDiff*winterEndDiff <= 0:
            winterRatio    = winterStartDiff/winterLength
                          
    if  (month == 4 and day > 6)  or (month == 5 and day < 4) :
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 13
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if month == 6 and day > 2 and day < 29 : 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 12
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if month == 6 and day > 2 and day < 29 : 
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 12
        witsunCount      = 1
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (month == 7 and day > 22) or month == 8 or (month == 9 and day < 23): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 48
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if ( day > 19 and month == 10) or ( month == 11 and day < 8): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 5
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if month == 12 and day > 16: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 13
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal    
    
def getBayern2015(dateVal,month,day):

    chrisStart    = datetime(2014, 12, 24)
    chrisEnd      = datetime(2015, 1, 5)  
    winterStart   = datetime(2015, 2, 16) 
    winterEnd     = datetime(2015, 2, 20)
    easterStart   = datetime(2015, 3, 30)
    easterEnd     = datetime(2015, 4, 11)
    witsun1Start  = datetime(2015, 5, 26)
    witsun1End    = datetime(2015, 6, 5)
    witsun2Start  = datetime(2015, 5, 26)
    witsun2End    = datetime(2015, 6, 5)
    summerStart   = datetime(2015, 8, 1)
    summerEnd     = datetime(2015, 9, 14)
    autumnStart   = datetime(2015, 11, 2)
    autumnEnd     = datetime(2015, 11, 7)
    nextChrisStart  = datetime(2015, 12, 24)
    nextChrisEnd    = datetime(2016, 1, 5)     


    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 13:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 13
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength

    if month == 2 and day < 28 and day > 8:
        winterStartDiff = (dateVal-winterStart).days     
        winterEndDiff   = (dateVal-winterStart).days
        winterLength     = 5
        if winterStartDiff*winterEndDiff <= 0:
            winterRatio    = winterStartDiff/winterLength
                          
    if (month == 3 and day > 22) or (month == 4 and day < 19):
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 13
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if (month == 5 and day > 18 ) or (month == 6 and day < 13): 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 11
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if (month == 5 and day > 18 ) or (month == 6 and day < 13): 
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 11
        witsunCount      = 1
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (month == 7  and day > 24 ) or month == 8 or (month == 9 and day < 22): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 45 
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if (month == 10 and day > 25) or (month == 11 and day < 15): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 6
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if month == 12 and day > 16: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 13
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal
    
############## BERLIN SCHOOL VACATIONS #################### 
def getBerlin2013(dateVal,month,day):

    chrisStart      = datetime(2012, 12, 24)
    chrisEnd        = datetime(2013, 1, 4)  
    winterStart     = datetime(2013, 2, 4) 
    winterEnd       = datetime(2013, 2, 9)
    easterStart     = datetime(2013, 3, 25)
    easterEnd       = datetime(2013, 4, 6)
    witsun1Start    = datetime(2013, 5, 10)
    witsun1End      = datetime(2013, 5, 10)
    witsun2Start    = datetime(2013, 5, 21)
    witsun2End      = datetime(2013, 5, 21)
    summerStart     = datetime(2013, 6, 19)
    summerEnd       = datetime(2013, 8, 2)
    autumnStart     = datetime(2013, 9, 30)
    autumnEnd       = datetime(2013, 10, 12)
    nextChrisStart  = datetime(2013, 12, 23)
    nextChrisEnd    = datetime(2014, 1, 3) 
    
     
    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 12:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 12
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength

    if (month == 1 and day > 27) or (month == 2 and day < 17):
        winterStartDiff = (dateVal-winterStart).days     
        winterEndDiff   = (dateVal-winterStart).days
        winterLength     = 6
        if winterStartDiff*winterEndDiff <= 0:
            winterRatio    = winterStartDiff/winterLength
            
                          
    if (day > 17 and month == 3) or (month == 4 and day < 14):
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 13
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if month == 5 and day > 7 and day < 12: 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 1
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if  month == 5 and day > 18 and day < 24:  
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 1
        witsunCount      = 2
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (day > 11 and month == 6) or month == 7 or (month == 8 and  day < 10): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 45
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if (day > 22 and month == 9) or (day < 20 and month == 10): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 13
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if day > 15 and month == 12: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 12
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal
    
            
def getBerlin2014(dateVal,month,day):

    chrisStart      = datetime(2013, 12, 23)
    chrisEnd        = datetime(2014, 1, 3)  
    winterStart     = datetime(2014, 2, 3) 
    winterEnd       = datetime(2014, 2, 7)
    easterStart     = datetime(2014, 4, 14)
    easterEnd       = datetime(2014, 4, 26)
    witsun1Start    = datetime(2014, 5, 2)
    witsun1End      = datetime(2014, 5, 2)
    witsun2Start    = datetime(2014, 5, 30)
    witsun2End      = datetime(2014, 5, 30)
    summerStart     = datetime(2014, 7, 9)
    summerEnd       = datetime(2014, 8, 22)
    autumnStart     = datetime(2014, 10, 20)
    autumnEnd       = datetime(2014, 11, 1)
    nextChrisStart  = datetime(2014, 12, 22)
    nextChrisEnd    = datetime(2015, 1, 2)     

    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 11:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 12
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength
            
    if (month == 1 and day > 26) or ( month == 2 and day < 16):
        winterStartDiff = (dateVal-winterStart).days     
        winterEndDiff   = (dateVal-winterStart).days
        winterLength     = 6
        if winterStartDiff*winterEndDiff <= 0:
            winterRatio    = winterStartDiff/winterLength
                          
    if  (month == 4 and day > 6)  or (month == 5 and day < 4) :
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 13
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if (month == 4 and day > 29 ) or (month == 5 and day < 5 ): 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 1
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if (month == 5 and day > 27 ) or (month == 6 and day < 2 ): 
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 1
        witsunCount      = 2
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (month == 7 and day > 1) or  (month == 8 and day < 30): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 45
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if ( day > 12 and month == 10) or ( month == 11 and day < 8): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 13
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if month == 12 and day > 14: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 12
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal    
    
def getBerlin2015(dateVal,month,day):

    chrisStart      = datetime(2014, 12, 22)
    chrisEnd        = datetime(2015, 1, 2)  
    winterStart     = datetime(2015, 2, 2) 
    winterEnd       = datetime(2015, 2, 7)
    easterStart     = datetime(2015, 3, 30)
    easterEnd       = datetime(2015, 4, 11)
    witsun1Start    = datetime(2015, 5, 15)
    witsun1End      = datetime(2015, 5, 15)
    witsun2Start    = datetime(2015, 5, 15)
    witsun2End      = datetime(2015, 5, 15)
    summerStart     = datetime(2015, 7, 15)
    summerEnd       = datetime(2015, 8, 28)
    autumnStart     = datetime(2015, 10, 19)
    autumnEnd       = datetime(2015, 10, 31)
    nextChrisStart  = datetime(2015, 12, 23)
    nextChrisEnd    = datetime(2016, 1, 2)     


    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 10:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 12
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength

    if (month == 1 and day > 25 ) or (month == 2 and day < 16) :
        winterStartDiff = (dateVal-winterStart).days     
        winterEndDiff   = (dateVal-winterStart).days
        winterLength     = 6
        if winterStartDiff*winterEndDiff <= 0:
            winterRatio    = winterStartDiff/winterLength
                          
    if (month == 3 and day > 22) or (month == 4 and day < 19):
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 13
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if month == 5 and day > 12  and day < 18: 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 1
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if month == 5 and day > 12  and day < 18:  
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 1
        witsunCount      = 1
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (month == 7  and day > 7 ) or month == 8 or (month == 9 and day < 5): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 45 
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if (month == 10 and day > 25)or (month == 11 and day < 15): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 6
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if month == 12 and day > 15: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 12
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal
    
############## BREMEN SCHOOL VACATIONS #################### 
def getBremen2013(dateVal,month,day):

    chrisStart      = datetime(2012, 12, 24)
    chrisEnd        = datetime(2013, 1, 5)  
    winterStart     = datetime(2013, 1, 31) 
    winterEnd       = datetime(2013, 2, 1)
    easterStart     = datetime(2013, 3, 16)
    easterEnd       = datetime(2013, 4, 2)
    witsun1Start    = datetime(2013, 5, 21)
    witsun1End      = datetime(2013, 5, 21)
    witsun2Start    = datetime(2013, 5, 21)
    witsun2End      = datetime(2013, 5, 21)
    summerStart     = datetime(2013, 6, 27)
    summerEnd       = datetime(2013, 8, 7)
    autumnStart     = datetime(2013, 10, 4)
    autumnEnd       = datetime(2013, 10, 18)
    nextChrisStart  = datetime(2013, 12, 23)
    nextChrisEnd    = datetime(2014, 1, 3) 
    
     
    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 13:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 13
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength

    if (month == 1 and day > 27) or (month == 2 and day < 4):
        winterStartDiff = (dateVal-winterStart).days     
        winterEndDiff   = (dateVal-winterStart).days
        winterLength     = 2
        if winterStartDiff*winterEndDiff <= 0:
            winterRatio    = winterStartDiff/winterLength
            
                          
    if (day > 8 and month == 3) or (month == 4 and day < 10):
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 18
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if  month == 5 and day > 17 and day < 25:
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 1
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if  month == 5 and day > 17 and day < 25:  
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 1
        witsunCount      = 1
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (day > 19 and month == 6) or month == 7 or (month == 8 and  day < 15): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 42
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if (day > 26 and month == 9) or (day < 26 and month == 10): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 15
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if day > 15 and month == 12: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 10
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal
    
            
def getBremen2014(dateVal,month,day):

    chrisStart      = datetime(2013, 12, 23)
    chrisEnd        = datetime(2014, 1, 3)  
    winterStart     = datetime(2014, 1, 30) 
    winterEnd       = datetime(2014, 1, 31)
    easterStart     = datetime(2014, 4, 3)
    easterEnd       = datetime(2014, 4, 22)
    witsun1Start    = datetime(2014, 6, 10)
    witsun1End      = datetime(2014, 6, 10)
    witsun2Start    = datetime(2014, 6, 10)
    witsun2End      = datetime(2014, 6, 10)
    summerStart     = datetime(2014, 7, 31)
    summerEnd       = datetime(2014, 9, 10)
    autumnStart     = datetime(2014, 10, 27)
    autumnEnd       = datetime(2014, 11, 8)
    nextChrisStart  = datetime(2014, 12, 22)
    nextChrisEnd    = datetime(2015, 1, 5)     

    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 11:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 10
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength
            
    if (month == 1 and day > 26) or ( month == 2 and day < 5):
        winterStartDiff = (dateVal-winterStart).days     
        winterEndDiff   = (dateVal-winterStart).days
        winterLength     = 2
        if winterStartDiff*winterEndDiff <= 0:
            winterRatio    = winterStartDiff/winterLength
                          
    if  (month == 3 and day > 25)  or (month == 4 and day < 30) :
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 21
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if month == 6 and day > 8  and day < 12 : 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 1
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if month == 6 and day > 8  and day < 12 : 
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 1
        witsunCount      = 2
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (month == 7 and day > 23) or month == 8 or (month == 9 and day < 18): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 42
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if ( day > 19 and month == 10) or ( month == 11 and day < 16): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 13
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if month == 12 and day > 14: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 15
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal    
    
def getBremen2015(dateVal,month,day):

    chrisStart      = datetime(2014, 12, 22)
    chrisEnd        = datetime(2015, 1, 5)  
    winterStart     = datetime(2015, 2, 2) 
    winterEnd       = datetime(2015, 2, 3)
    easterStart     = datetime(2015, 3, 25)
    easterEnd       = datetime(2015, 4, 10)
    witsun1Start    = datetime(2015, 5, 26)
    witsun1End      = datetime(2015, 5, 26)
    witsun2Start    = datetime(2015, 5, 26)
    witsun2End      = datetime(2015, 5, 26)
    summerStart     = datetime(2015, 7, 23)
    summerEnd       = datetime(2015, 9, 2)
    autumnStart     = datetime(2015, 10, 19)
    autumnEnd       = datetime(2015, 10, 31)
    nextChrisStart  = datetime(2015, 12, 23)
    nextChrisEnd    = datetime(2016, 1, 6)     


    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 13:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 15
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength

    if (month == 1 and day > 25 ) or (month == 2 and day < 9) :
        winterStartDiff = (dateVal-winterStart).days     
        winterEndDiff   = (dateVal-winterStart).days
        winterLength     = 2
        if winterStartDiff*winterEndDiff <= 0:
            winterRatio    = winterStartDiff/winterLength
                          
    if (month == 3 and day > 17) or (month == 4 and day < 18):
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 17
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if month == 5 and day > 23  and day < 29: 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 1
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if month == 5 and day > 23  and day < 29: 
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 1
        witsunCount      = 1
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (month == 7  and day > 15 ) or month == 8 or (month == 9 and day < 10): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 42 
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if (month == 10 and day > 11)or (month == 11 and day < 8): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 13
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if month == 12 and day > 15: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 15
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal


############## HAMBURG SCHOOL VACATIONS ####################
def getHamburg2013(dateVal,month,day):

    chrisStart      = datetime(2012, 12, 21)
    chrisEnd        = datetime(2013, 1, 4)  
    winterStart     = datetime(2012, 2, 1) 
    winterEnd       = datetime(2012, 2, 1)
    easterStart     = datetime(2013, 3, 4)
    easterEnd       = datetime(2013, 4, 15)
    witsun1Start    = datetime(2013, 5, 2)
    witsun1End      = datetime(2013, 5, 10)
    witsun2Start    = datetime(2013, 5, 2)
    witsun2End      = datetime(2013, 5, 10)
    summerStart     = datetime(2013, 6, 20)
    summerEnd       = datetime(2013, 7, 31)
    autumnStart     = datetime(2013, 9, 30)
    autumnEnd       = datetime(2013, 10,11)
    nextChrisStart  = datetime(2013, 12, 19)
    nextChrisEnd    = datetime(2014, 1, 3) 
    
     
    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 12:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 15
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength

    if (month == 1 and day > 29 ) or (month == 2 and day < 4) :
        winterStartDiff = (dateVal-winterStart).days     
        winterEndDiff   = (dateVal-winterStart).days
        winterLength     = 1
        if winterStartDiff*winterEndDiff <= 0:
            winterRatio    = winterStartDiff/winterLength
                          
    if (day > 24 and month == 2) or (month == 3 and day < 23):
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 12
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if (month == 4 and day > 24) or (month == 5 and day < 18): 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 9
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if (month == 4 and day > 24) or (month == 5 and day < 18): 
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 9
        witsunCount      = 1
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (day > 12 and month == 6) or month == 7 or (month == 8 and  day < 8): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 42 
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if (day > 23 and month == 9) or (day < 19 and month == 10): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 11
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if day > 11 and month == 12: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 16
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal
    
            
def getHamburg2014(dateVal,month,day):

    chrisStart    = datetime(2013, 12, 19)
    chrisEnd      = datetime(2014, 1, 3)  
    winterStart   = datetime(2012, 1, 31) 
    winterEnd     = datetime(2012, 1, 31)
    easterStart   = datetime(2014, 3, 3)
    easterEnd     = datetime(2014, 3, 14)
    witsun1Start  = datetime(2014, 4, 28)
    witsun1End    = datetime(2014, 5, 2)
    witsun2Start  = datetime(2014, 5, 30)
    witsun2End    = datetime(2014, 5, 30)
    summerStart   = datetime(2014, 7, 10)
    summerEnd     = datetime(2014, 8, 20)
    autumnStart   = datetime(2014, 10, 13)
    autumnEnd     = datetime(2014, 10, 24)
    nextChrisStart  = datetime(2014, 12, 22)
    nextChrisEnd    = datetime(2015, 1, 6)     

    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 11:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 16
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength

    if (month == 1 and day > 26 ) or (month == 2 and day < 4) :
        winterStartDiff = (dateVal-winterStart).days     
        winterEndDiff   = (dateVal-winterStart).days
        winterLength     = 1
        if winterStartDiff*winterEndDiff <= 0:
            winterRatio    = winterStartDiff/winterLength
                          
    if  (month == 2 and day > 23 ) or (month == 3 and day < 22):
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 12
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if (month == 4 and day > 20) or (month == 5 and day < 8) : 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 5
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if (month == 5 and day > 25 ) or (month == 6 and day < 3)  : 
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 1
        witsunCount      = 2
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (month == 7 and day > 2) or (month == 8 and day < 28): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 42
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if  day > 5 and month == 10: 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 12
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if month == 12 and day > 14: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 16
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal    
    
def getHamburg2015(dateVal,month,day):

    chrisStart    = datetime(2014, 12, 22)
    chrisEnd      = datetime(2015, 1, 6)  
    winterStart   = datetime(2012, 1, 30) 
    winterEnd     = datetime(2012, 1, 30)
    easterStart   = datetime(2015, 3, 2)
    easterEnd     = datetime(2015, 3, 13)
    witsun1Start  = datetime(2015, 5, 11)
    witsun1End    = datetime(2015, 6, 15)
    witsun2Start  = datetime(2015, 5, 11)
    witsun2End    = datetime(2015, 6, 15)
    summerStart   = datetime(2015, 7, 16)
    summerEnd     = datetime(2015, 8, 26)
    autumnStart   = datetime(2015, 10, 19)
    autumnEnd     = datetime(2015, 10, 30)
    nextChrisStart  = datetime(2015, 12, 21)
    nextChrisEnd    = datetime(2016, 1, 1)     



    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 13:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 16
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength

    if (month == 1 and day > 25 ) or (month == 2 and day < 4) :
        winterStartDiff = (dateVal-winterStart).days     
        winterEndDiff   = (dateVal-winterStart).days
        winterLength     = 1
        if winterStartDiff*winterEndDiff <= 0:
            winterRatio    = winterStartDiff/winterLength
                          
    if (month == 2 and day > 22) or (month == 4 and day < 21):
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 12
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if (month == 5 and day > 3   and day < 23): 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 5
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if (month == 5 and day > 3   and day < 23): 
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 5
        witsunCount      = 1
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (month == 7  and day > 8 )or month == 8 or (month == 9 and day < 3): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 42
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if (month == 10 and day > 11 )or (month == 11 and day < 6): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 13
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if month == 12 and day > 13: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 12
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal    

############## HESSEN SCHOOL VACATIONS ####################
def getHessen2013(dateVal,month,day):

    chrisStart      = datetime(2012, 12, 24)
    chrisEnd        = datetime(2013, 1, 12)  
    winterStart     = None
    winterEnd       = None
    easterStart     = datetime(2013, 3, 25)
    easterEnd       = datetime(2013, 4, 6)
    witsun1Start    = None
    witsun1End      = None
    witsun2Start    = None
    witsun2End      = None
    summerStart     = datetime(2013, 7, 8)
    summerEnd       = datetime(2013, 8, 16)
    autumnStart     = datetime(2013, 10,14)
    autumnEnd       = datetime(2013, 10,26)
    nextChrisStart  = datetime(2013, 12,23)
    nextChrisEnd    = datetime(2014, 1, 11) 
    
     
    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 20:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 20
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength


                          
    if (day > 17 and month == 3) or (month == 4 and day < 14):
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 13
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
            
    if (month == 7) or (month == 8 and  day < 24): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 40
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if (day > 6 and month == 10) or (day < 4 and month == 11): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 13
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if day > 15 and month == 12: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 20
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal
    
            
def getHessen2014(dateVal,month,day):

    chrisStart    = datetime(2013, 12, 23)
    chrisEnd      = datetime(2014, 1, 11)  
    winterStart   = None
    winterEnd     = None
    easterStart   = datetime(2014, 4, 14)
    easterEnd     = datetime(2014, 4, 26)
    witsun1Start  = None
    witsun1End    = None
    witsun2Start  = None
    witsun2End    = None
    summerStart   = datetime(2014, 7, 28)
    summerEnd     = datetime(2014, 9, 5)
    autumnStart   = datetime(2014, 10, 20)
    autumnEnd     = datetime(2014, 11, 1)
    nextChrisStart  = datetime(2014, 12, 22)
    nextChrisEnd    = datetime(2015, 1, 10)     

    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 19:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 20
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength

    if  month == 4 and day > 6 :
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 13
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if (month == 7 and day > 20) or month == 8 or (month == 9 and day < 13): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 40
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if  (day > 12 and month == 10) or (month == 11 and day < 8): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 13
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if month == 12 and day > 14: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 20
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal    
    
def getHessen2015(dateVal,month,day):

    chrisStart    = datetime(2014, 12, 22)
    chrisEnd      = datetime(2015, 1, 10)  
    winterStart   = None 
    winterEnd     = None
    easterStart   = datetime(2015, 3, 30)
    easterEnd     = datetime(2015, 4, 11)
    witsun1Start  = None
    witsun1End    = None
    witsun2Start  = None
    witsun2End    = None
    summerStart   = datetime(2015, 7, 27)
    summerEnd     = datetime(2015, 9, 4)
    autumnStart   = datetime(2015, 10, 19)
    autumnEnd     = datetime(2015, 10, 31)
    nextChrisStart  = datetime(2015, 12, 23)
    nextChrisEnd    = datetime(2016, 1, 9)     



    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 18:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 20
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength

    if (month == 3 and day > 22) or (month == 4 and day < 19):
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 13
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
            
    if (month == 7  and day > 19 )or month == 8 or (month == 9 and day < 12): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 40
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if (month == 10 and day > 11 )or (month == 11 and day < 8): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 13
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if month == 12 and day > 15: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 18
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal     
    
############## NRW SCHOOL VACATIONS ####################
def getNRW2013(dateVal,month,day):

    chrisStart      = datetime(2012, 12, 21)
    chrisEnd        = datetime(2013, 1, 4)  
    winterStart     = None 
    winterEnd       = None
    easterStart     = datetime(2013, 3, 25)
    easterEnd       = datetime(2013, 4, 6)
    witsun1Start    = datetime(2013, 5, 21)
    witsun1End      = datetime(2013, 5, 21)
    witsun2Start    = datetime(2013, 5, 21)
    witsun2End      = datetime(2013, 5, 21)
    summerStart     = datetime(2013, 7, 22)
    summerEnd       = datetime(2013, 9, 3)
    autumnStart     = datetime(2013, 10, 21)
    autumnEnd       = datetime(2013, 11, 2)
    nextChrisStart  = datetime(2013, 12, 23)
    nextChrisEnd    = datetime(2014, 1, 7) 
    
     
    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 12:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 15
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength
              
    if (day > 17 and month == 3) or (month == 4 and day < 14):
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 13
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if month == 5 and day > 18 and day < 24:  
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 1
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if month == 5 and day > 18 and day < 24:  
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 1
        witsunCount      = 1
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (day > 14 and month == 7) or month == 8 or (month == 9 and  day < 11): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 44
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if (day > 13 and month == 10) or (day < 10 and month == 11): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 13
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if day > 15 and month == 12: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 16
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal
    
            
def getNRW2014(dateVal,month,day):

    chrisStart    = datetime(2013, 12, 23)
    chrisEnd      = datetime(2014, 1, 7)  
    winterStart   = None 
    winterEnd     = None
    easterStart   = datetime(2014, 4, 14)
    easterEnd     = datetime(2014, 4, 26)
    witsun1Start  = datetime(2014, 6, 10)
    witsun1End    = datetime(2014, 6, 10)
    witsun2Start  = datetime(2014, 6, 10)
    witsun2End    = datetime(2014, 6, 10)
    summerStart   = datetime(2014, 7, 7)
    summerEnd     = datetime(2014, 8, 19)
    autumnStart   = datetime(2014, 10, 6)
    autumnEnd     = datetime(2014, 10, 18)
    nextChrisStart  = datetime(2014, 12, 22)
    nextChrisEnd    = datetime(2015, 1, 6)     

    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 15:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 16
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength
              
    if  month == 4 and day > 6  :
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 12
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if month == 6 and day > 8 and day < 12 : 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 1
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if month == 6 and day > 8 and day < 12 : 
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 1
        witsunCount      = 1
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if month == 7  or (month == 8  and day < 27): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 44
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if  day <  26 and month == 10: 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 4
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if month == 12 and day > 14: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 16
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal    
    
def getNRW2015(dateVal,month,day):

    chrisStart    = datetime(2014, 12, 22)
    chrisEnd      = datetime(2015, 1, 6)  
    winterStart   = None 
    winterEnd     = None
    easterStart   = datetime(2015, 3, 30)
    easterEnd     = datetime(2015, 4, 11)
    witsun1Start  = datetime(2015, 5, 26)
    witsun1End    = datetime(2015, 5, 26)
    witsun2Start  = datetime(2015, 5, 26)
    witsun2End    = datetime(2015, 5, 26)
    summerStart   = datetime(2015, 6, 29)
    summerEnd     = datetime(2015, 8, 11)
    autumnStart   = datetime(2015, 10, 5)
    autumnEnd     = datetime(2015, 10, 17)
    nextChrisStart  = datetime(2015, 12, 23)
    nextChrisEnd    = datetime(2016, 1, 6)     



    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 14:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 16
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength
              
    if (month == 3 and day > 22) or (month == 4 and day < 19):
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 13
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if (month == 5 and day > 24 and day < 28): 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 1
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if (month == 5 and day > 24 and day < 28):  
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 1
        witsunCount      = 1
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (month == 6  and day > 21 )or month == 7 or (month == 8 and day < 19): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 44
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if (month == 9 and day > 27)or (month == 10 and day < 25): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 13
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if month == 12 and day > 15: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 15
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal

############## RHEINLAND SCHOOL VACATIONS ####################
def getRheinland2013(dateVal,month,day):

    chrisStart      = datetime(2012, 12, 20)
    chrisEnd        = datetime(2013, 1, 4)  
    winterStart     = None 
    winterEnd       = None
    easterStart     = datetime(2013, 3, 20)
    easterEnd       = datetime(2013, 4, 5)
    witsun1Start    = None
    witsun1End      = None
    witsun2Start    = None
    witsun2End      = None
    summerStart     = datetime(2013, 7, 8)
    summerEnd       = datetime(2013, 8, 16)
    autumnStart     = datetime(2013, 10, 4)
    autumnEnd       = datetime(2013, 10, 18)
    nextChrisStart  = datetime(2013, 12, 23)
    nextChrisEnd    = datetime(2014, 1, 7) 
    
     
    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 12:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 16
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength
              
    if (day > 13 and month == 3) or (month == 4 and day < 13):
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 17
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if  month == 7 or (month == 8 and  day < 24): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 40
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if (day > 25 and month == 9) or (day < 26 and month == 10): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 15
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if day > 15 and month == 12: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 16
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal
    
            
def getRheinland2014(dateVal,month,day):

    chrisStart    = datetime(2013, 12, 23)
    chrisEnd      = datetime(2014, 1, 7)  
    winterStart   = None 
    winterEnd     = None
    easterStart   = datetime(2014, 4, 11)
    easterEnd     = datetime(2014, 4, 25)
    witsun1Start  = datetime(2014, 5, 30)
    witsun1End    = datetime(2014, 5, 30)
    witsun2Start  = datetime(2014, 6, 20)
    witsun2End    = datetime(2014, 6, 20)
    summerStart   = datetime(2014, 7, 28)
    summerEnd     = datetime(2014, 9, 5)
    autumnStart   = datetime(2014, 10, 20)
    autumnEnd     = datetime(2014, 10, 31)
    nextChrisStart  = datetime(2014, 12, 22)
    nextChrisEnd    = datetime(2015, 1, 7)     

    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 15:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 16
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength
              
    if  month == 4 and day > 3  :
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 15
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if month == 5 and day > 28  : 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 1
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if month == 6 and day > 18 and day < 23 : 
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 1
        witsunCount      = 1
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (month == 7  and day > 20) or month == 8 or (month == 9  and day < 13): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 40
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if (month == 10 and day > 12 ) or (day <  8 and month == 11): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 12
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if month == 12 and day > 14: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 17
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal    
    
def getRheinland2015(dateVal,month,day):

    chrisStart    = datetime(2014, 12, 22)
    chrisEnd      = datetime(2015, 1, 7)  
    winterStart   = None 
    winterEnd     = None
    easterStart   = datetime(2015, 3, 26)
    easterEnd     = datetime(2015, 4, 10)
    witsun1Start  = datetime(2015, 5, 15)
    witsun1End    = datetime(2015, 5, 15)
    witsun2Start  = datetime(2015, 6, 5)
    witsun2End    = datetime(2015, 6, 5)
    summerStart   = datetime(2015, 7, 27)
    summerEnd     = datetime(2015, 9, 4)
    autumnStart   = datetime(2015, 10, 19)
    autumnEnd     = datetime(2015, 10, 30)
    nextChrisStart  = datetime(2015, 12, 23)
    nextChrisEnd    = datetime(2016, 1, 8)     



    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 15:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 17
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength
              
    if (month == 3 and day > 18) or (month == 4 and day < 18):
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 16
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if (month == 5 and day > 12 and day < 18): 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 1
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if (month == 6 and day > 3 and day < 8):  
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 1
        witsunCount      = 1
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (month == 7  and day > 19 ) or month == 8 or (month == 9 and day < 12): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 40
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if (month == 10 and day > 11 )or (month == 11 and day < 7): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 12
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if month == 12 and day > 15: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 18
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal
        
############## SACHSEN SCHOOL VACATIONS ####################
def getSachsen2013(dateVal,month,day):

    chrisStart      = datetime(2012, 12, 22)
    chrisEnd        = datetime(2013, 1, 2)  
    winterStart     = datetime(2013, 2, 3) 
    winterEnd       = datetime(2013, 2, 15)
    easterStart     = datetime(2013, 3, 29)
    easterEnd       = datetime(2013, 4, 6)
    witsun1Start    = datetime(2013, 5, 10)
    witsun1End      = datetime(2013, 5, 10)
    witsun2Start    = datetime(2013, 5, 18)
    witsun2End      = datetime(2013, 5, 22)
    summerStart     = datetime(2013, 7, 15)
    summerEnd       = datetime(2013, 8, 23)
    autumnStart     = datetime(2013, 10, 21)
    autumnEnd       = datetime(2013, 11, 1)
    nextChrisStart  = datetime(2013, 12, 21)
    nextChrisEnd    = datetime(2014, 1, 3) 
    
     
    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 10:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 12
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength

    if (month == 1 and day > 27) or ( month == 2 and day < 23):
        winterStartDiff = (dateVal-winterStart).days     
        winterEndDiff   = (dateVal-winterStart).days
        winterLength     = 12
        if winterStartDiff*winterEndDiff <= 0:
            winterRatio    = winterStartDiff/winterLength
            
                          
    if (day > 21 and month == 3) or (month == 4 and day < 14):
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 9
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if (month == 5 and day > 7 and day < 13): 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 1
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if (month == 5 and day > 12 and day < 30):  
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 5
        witsunCount      = 2
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (day > 7 and month == 7) or month == 8 : 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 40 
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if (day > 13 and month == 10) or (day < 9 and month == 11): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 12
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if day > 13 and month == 12: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 14
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal
    
            
def getSachsen2014(dateVal,month,day):

    chrisStart    = datetime(2013, 12, 21)
    chrisEnd      = datetime(2014, 1, 3)  
    winterStart   = datetime(2014, 2, 17) 
    winterEnd     = datetime(2014, 3, 1)
    easterStart   = datetime(2014, 4, 18)
    easterEnd     = datetime(2014, 4, 26)
    witsun1Start  = datetime(2014, 5, 30)
    witsun1End    = datetime(2014, 5, 30)
    witsun2Start  = datetime(2014, 5, 30)
    witsun2End    = datetime(2014, 5, 30)
    summerStart   = datetime(2014, 7, 21)
    summerEnd     = datetime(2014, 8, 29)
    autumnStart   = datetime(2014, 10, 20)
    autumnEnd     = datetime(2014, 10, 31)
    nextChrisStart  = datetime(2014, 12, 22)
    nextChrisEnd    = datetime(2015, 1, 3)     

    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 11:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 14
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength
            
    if (month == 2 and day > 9) or ( month == 3 and day < 9):
        winterStartDiff = (dateVal-winterStart).days     
        winterEndDiff   = (dateVal-winterStart).days
        winterLength     = 13
        if winterStartDiff*winterEndDiff <= 0:
            winterRatio    = winterStartDiff/winterLength
                          
    if  (month == 4 and day > 10)  or (month == 5 and day < 4) :
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 9
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if month == 5 and day > 27 : 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 1
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if month == 5 and day > 27 : 
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 1
        witsunCount      = 1
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (month == 7 and day > 13) or month == 8 or (month == 9 and day < 6): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 40
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if ( day > 12 and month == 10) or ( month == 11 and day < 8): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 12
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if month == 12 and day > 14: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 13
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal    
    
def getSachsen2015(dateVal,month,day):

    chrisStart    = datetime(2014, 12, 22)
    chrisEnd      = datetime(2015, 1, 3)  
    winterStart   = datetime(2015, 2, 9) 
    winterEnd     = datetime(2015, 2, 21)
    easterStart   = datetime(2015, 4, 2)
    easterEnd     = datetime(2015, 4, 11)
    witsun1Start  = datetime(2015, 5, 15)
    witsun1End    = datetime(2015, 5, 15)
    witsun2Start  = datetime(2015, 5, 15)
    witsun2End    = datetime(2015, 5, 15)
    summerStart   = datetime(2015, 7, 13)
    summerEnd     = datetime(2015, 8, 21)
    autumnStart   = datetime(2015, 10, 12)
    autumnEnd     = datetime(2015, 10, 24)
    nextChrisStart  = datetime(2015, 12, 21)
    nextChrisEnd    = datetime(2016, 1, 2)     


    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 11:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 13
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength

    if month == 2:
        winterStartDiff = (dateVal-winterStart).days     
        winterEndDiff   = (dateVal-winterStart).days
        winterLength     = 13
        if winterStartDiff*winterEndDiff <= 0:
            winterRatio    = winterStartDiff/winterLength
                          
    if (month == 3 and day > 25) or (month == 4 and day < 19):
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 10
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if (month == 5 and day > 12  and day < 18): 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 1
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if (month == 5 and day > 12  and day < 18): 
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 1
        witsunCount      = 1
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (month == 7  and day > 5 )  or (month == 8 and day < 29): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 40 
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if (month == 10 and day > 4) : 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 13
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if month == 12 and day > 13: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 13
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal     
    
############## SACHSEN ANHALT SCHOOL VACATIONS ####################
def getSachsenAnhalt2013(dateVal,month,day):

    chrisStart      = datetime(2012, 12, 19)
    chrisEnd        = datetime(2013, 1, 4)  
    winterStart     = datetime(2013, 2, 1) 
    winterEnd       = datetime(2013, 2, 8)
    easterStart     = datetime(2013, 3, 25)
    easterEnd       = datetime(2013, 3, 30)
    witsun1Start    = datetime(2013, 5, 10)
    witsun1End      = datetime(2013, 5, 18)
    witsun2Start    = datetime(2013, 5, 10)
    witsun2End      = datetime(2013, 5, 18)
    summerStart     = datetime(2013, 7, 15)
    summerEnd       = datetime(2013, 8, 28)
    autumnStart     = datetime(2013, 10, 21)
    autumnEnd       = datetime(2013, 10, 25)
    nextChrisStart  = datetime(2013, 12, 21)
    nextChrisEnd    = datetime(2014, 1, 3) 
    
     
    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 12:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 17
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength

    if (month == 1 and day > 23) or ( month == 2 and day < 16):
        winterStartDiff = (dateVal-winterStart).days     
        winterEndDiff   = (dateVal-winterStart).days
        winterLength     = 8
        if winterStartDiff*winterEndDiff <= 0:
            winterRatio    = winterStartDiff/winterLength
            
                          
    if (day > 17 and month == 3) or (month == 4 and day < 6):
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 6
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if (month == 5 and day > 2 and day < 26): 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 9
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if (month == 5 and day > 2 and day < 26): 
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 9
        witsunCount      = 1
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (day > 7 and month == 7) or month == 8 or (month == 9 and day < 7) : 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 45
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if (day > 13 and month == 10) or (day < 2 and month == 11): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 5
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if day > 13 and month == 12: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 14
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal
    
            
def getSachsenAnhalt2014(dateVal,month,day):

    chrisStart    = datetime(2013, 12, 21)
    chrisEnd      = datetime(2014, 1, 3)  
    winterStart   = datetime(2014, 2, 1) 
    winterEnd     = datetime(2014, 2, 12)
    easterStart   = datetime(2014, 4, 14)
    easterEnd     = datetime(2014, 4, 17)
    witsun1Start  = datetime(2014, 5, 30)
    witsun1End    = datetime(2014, 6, 7)
    witsun2Start  = datetime(2014, 5, 30)
    witsun2End    = datetime(2014, 6, 7)
    summerStart   = datetime(2014, 7, 21)
    summerEnd     = datetime(2014, 9, 3)
    autumnStart   = datetime(2014, 10, 27)
    autumnEnd     = datetime(2014, 10, 30)
    nextChrisStart  = datetime(2014, 12, 22)
    nextChrisEnd    = datetime(2015, 1, 5)     

    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 11:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 14
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength
            
    if (month == 1 and day > 23) or ( month == 2 and day < 20):
        winterStartDiff = (dateVal-winterStart).days     
        winterEndDiff   = (dateVal-winterStart).days
        winterLength     = 12
        if winterStartDiff*winterEndDiff <= 0:
            winterRatio    = winterStartDiff/winterLength
                          
    if  (month == 4 and day > 6 and day < 25) :
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 4
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if (month == 5 and day > 22) or (month == 6 and day < 15) : 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 9
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if (month == 5 and day > 22) or (month == 6 and day < 15) : 
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 9
        witsunCount      = 1
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (month == 7 and day > 13) or month == 8 or (month == 9 and day < 11): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 45
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if ( day > 19 and month == 10) or ( month == 11 and day < 6): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 4
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if month == 12 and day > 14: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 15
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal    
    
def getSachsenAnhalt2015(dateVal,month,day):

    chrisStart    = datetime(2014, 12, 22)
    chrisEnd      = datetime(2015, 1, 5)  
    winterStart   = datetime(2015, 2, 2) 
    winterEnd     = datetime(2015, 2, 14)
    easterStart   = datetime(2015, 4, 2)
    easterEnd     = datetime(2015, 4, 2)
    witsun1Start  = datetime(2015, 5, 15)
    witsun1End    = datetime(2015, 5, 23)
    witsun2Start  = datetime(2015, 5, 15)
    witsun2End    = datetime(2015, 5, 23)
    summerStart   = datetime(2015, 7, 13)
    summerEnd     = datetime(2015, 8, 26)
    autumnStart   = datetime(2015, 10, 17)
    autumnEnd     = datetime(2015, 10, 24)
    nextChrisStart  = datetime(2015, 12, 21)
    nextChrisEnd    = datetime(2016, 1, 5)     


    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 13:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 15
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength

    if (month == 1 and day > 25) or (month == 2 and day < 22):
        winterStartDiff = (dateVal-winterStart).days     
        winterEndDiff   = (dateVal-winterStart).days
        winterLength     = 13
        if winterStartDiff*winterEndDiff <= 0:
            winterRatio    = winterStartDiff/winterLength
                          
    if (month == 4 and day < 6):
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 1
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if (month == 5 and day > 7  and day < 31): 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 9
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if (month == 5 and day > 7  and day < 31): 
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 9
        witsunCount      = 1
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (month == 7  and day > 5 )  or month == 8: 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 45
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if (month == 10 and day > 9) : 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 8
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if month == 12 and day > 13: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 18
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return  retVal 
    
    
############## SCHLESWIG SCHOOL VACATIONS ####################
def getScheleswig2013(dateVal,month,day):

    chrisStart      = datetime(2012, 12, 24)
    chrisEnd        = datetime(2013, 1, 5)  
    winterStart     = None 
    winterEnd       = None
    easterStart     = datetime(2013, 3, 25)
    easterEnd       = datetime(2013, 4, 9)
    witsun1Start    = datetime(2013, 5, 10)
    witsun1End      = datetime(2013, 5, 10)
    witsun2Start    = datetime(2013, 5, 10)
    witsun2End      = datetime(2013, 5, 10)
    summerStart     = datetime(2013, 6, 24)
    summerEnd       = datetime(2013, 8, 3)
    autumnStart     = datetime(2013, 10, 4)
    autumnEnd       = datetime(2013, 10, 18)
    nextChrisStart  = datetime(2013, 12, 23)
    nextChrisEnd    = datetime(2014, 1, 6) 
    
     
    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 13:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 13
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength
              
    if (day > 17 and month == 3) or (month == 4 and day < 17):
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 12
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if (month == 5 and day > 7 and day < 13): 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 1
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if (month == 5 and day > 7 and day < 13):  
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 1
        witsunCount      = 1
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (day > 16 and month == 6) or month == 7 or (month == 8 and  day < 11): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 41 
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if (day > 25 and month == 9) or (day < 16 and month == 10): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 15
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if day > 15 and month == 12: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 15
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal
    
            
def getScheleswig2014(dateVal,month,day):

    chrisStart    = datetime(2013, 12, 23)
    chrisEnd      = datetime(2014, 1, 6)  
    winterStart   = None 
    winterEnd     = None
    easterStart   = datetime(2014, 4, 16)
    easterEnd     = datetime(2014, 5, 2)
    witsun1Start  = datetime(2014, 5, 30)
    witsun1End    = datetime(2014, 5, 30)
    witsun2Start  = datetime(2014, 5, 30)
    witsun2End    = datetime(2014, 5, 30)
    summerStart   = datetime(2014, 7, 14)
    summerEnd     = datetime(2014, 8, 23)
    autumnStart   = datetime(2014, 10, 13)
    autumnEnd     = datetime(2014, 10, 25)
    nextChrisStart  = datetime(2014, 12, 22)
    nextChrisEnd    = datetime(2015, 1, 6)     

    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 14:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 15
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength
              
    if  (month == 4 and day > 8)  or ( month == 5 and day < 10) :
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 17
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if month == 5 and day > 27 : 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 1
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if month == 5 and day > 27 :
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 1
        witsunCount      = 1
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (month == 7 and day > 6) or month == 8 : 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 41
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if ( day > 5 and month == 10): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 13
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if month == 12 and day > 14: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 16
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal    
    
def getScheleswig2015(dateVal,month,day):

    chrisStart    = datetime(2014, 12, 22)
    chrisEnd      = datetime(2015, 1, 6)  
    winterStart   = None 
    winterEnd     = None
    easterStart   = datetime(2015, 4, 1)
    easterEnd     = datetime(2015, 4, 17)
    witsun1Start  = datetime(2015, 5, 15)
    witsun1End    = datetime(2015, 5, 15)
    witsun2Start  = datetime(2015, 5, 15)
    witsun2End    = datetime(2015, 5, 15)
    summerStart   = datetime(2015, 7, 20)
    summerEnd     = datetime(2015, 8, 29)
    autumnStart   = datetime(2015, 10, 19)
    autumnEnd     = datetime(2015, 10, 31)
    nextChrisStart  = datetime(2015, 12, 21)
    nextChrisEnd    = datetime(2016, 1, 6)     



    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 14:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 16
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength
              
    if (month == 3 and day > 25) or (month == 4 and day < 25):
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 17
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if (month == 5 and day > 12 and day < 18): 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 1
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if (month == 5 and day > 12 and day < 18):  
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 1
        witsunCount      = 1
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (month == 7  and day > 12 ) or month == 8 or (month == 9 and day < 5): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 41 
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if (month == 10 and day > 11)or (month == 11 and day < 8): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 13
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if month == 12 and day > 13: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 17
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal
        
    
############## THURINGEN SCHOOL VACATIONS ####################
def getThuringen2013(dateVal,month,day):

    chrisStart      = datetime(2012, 12, 24)
    chrisEnd        = datetime(2013, 1, 5)  
    winterStart     = datetime(2013, 2, 18) 
    winterEnd       = datetime(2013, 2, 23)
    easterStart     = datetime(2013, 3, 25)
    easterEnd       = datetime(2013, 4, 6)
    witsun1Start    = datetime(2013, 5, 10)
    witsun1End      = datetime(2013, 5, 10)
    witsun2Start    = datetime(2013, 5, 10)
    witsun2End      = datetime(2013, 5, 10)
    summerStart     = datetime(2013, 7, 15)
    summerEnd       = datetime(2013, 8, 23)
    autumnStart     = datetime(2013, 10, 21)
    autumnEnd       = datetime(2013, 11, 2)
    nextChrisStart  = datetime(2013, 12, 23)
    nextChrisEnd    = datetime(2014, 1, 4) 
    
     
    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 13:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 13
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength

    if month == 2 and  day > 10:
        winterStartDiff = (dateVal-winterStart).days     
        winterEndDiff   = (dateVal-winterStart).days
        winterLength     = 6
        if winterStartDiff*winterEndDiff <= 0:
            winterRatio    = winterStartDiff/winterLength
            
                          
    if (day > 17 and month == 3) or (month == 4 and day < 14):
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 13
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if (month == 5 and day > 7  and day < 13): 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 1
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if (month == 5 and day > 7  and day < 13):  
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 1
        witsunCount      = 1
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (day > 7 and month == 7) or month == 8 : 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 40 
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if (day > 13 and month == 10) or (day < 10 and month == 11): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 13
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if day > 15 and month == 12: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 13
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal
    
            
def getThuringen2014(dateVal,month,day):

    chrisStart    = datetime(2013, 12, 23)
    chrisEnd      = datetime(2014, 1, 4)  
    winterStart   = datetime(2014, 2, 17) 
    winterEnd     = datetime(2014, 2, 22)
    easterStart   = datetime(2014, 4, 19)
    easterEnd     = datetime(2014, 5, 2)
    witsun1Start  = datetime(2014, 5, 30)
    witsun1End    = datetime(2014, 5, 30)
    witsun2Start  = datetime(2014, 5, 30)
    witsun2End    = datetime(2014, 5, 30)
    summerStart   = datetime(2014, 7, 21)
    summerEnd     = datetime(2014, 8, 29)
    autumnStart   = datetime(2014, 10, 6)
    autumnEnd     = datetime(2014, 10, 18)
    nextChrisStart  = datetime(2014, 12, 22)
    nextChrisEnd    = datetime(2015, 1, 3)     

    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 12:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 13
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength
            
    if (month == 2 and day > 9) or ( month == 3 and day < 3):
        winterStartDiff = (dateVal-winterStart).days     
        winterEndDiff   = (dateVal-winterStart).days
        winterLength     = 6
        if winterStartDiff*winterEndDiff <= 0:
            winterRatio    = winterStartDiff/winterLength
                          
    if  (month == 4 and day > 11)  or (month == 5 and day < 10) :
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 14
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if month == 6 and day > 27 : 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 1
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if month == 6 and day > 27 :  
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 1
        witsunCount      = 1
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (month == 7 and day > 13) or month == 8 or (month == 9 and day < 6): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 40
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if ( day < 26 and month == 10): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 13
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if month == 12 and day > 14: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 13
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal    
    
def getThuringen2015(dateVal,month,day):

    chrisStart    = datetime(2014, 12, 22)
    chrisEnd      = datetime(2015, 1, 3)  
    winterStart   = datetime(2015, 2, 2) 
    winterEnd     = datetime(2015, 2, 7)
    easterStart   = datetime(2015, 3, 30)
    easterEnd     = datetime(2015, 4, 11)
    witsun1Start  = datetime(2015, 5, 15)
    witsun1End    = datetime(2015, 5, 15)
    witsun2Start  = datetime(2015, 5, 15)
    witsun2End    = datetime(2015, 5, 15)
    summerStart   = datetime(2015, 7, 13)
    summerEnd     = datetime(2015, 8, 21)
    autumnStart   = datetime(2015, 10, 5)
    autumnEnd     = datetime(2015, 10, 17)
    nextChrisStart  = datetime(2015, 12, 23)
    nextChrisEnd    = datetime(2016, 1, 2)     


    chrisStartDiff     = -999
    chrisEndDiff       = -999
    chrisRatio         = -999  
    chrisLength        = -999
    winterStartDiff    = -999
    winterEndDiff      = -999
    winterRatio        = -999
    winterLength       = -999
    easterStartDiff    = -999
    easterEndDiff      = -999
    easterRatio        = -999
    easterLength       = -999
    witsunCount        = -999 
    witsun1StartDiff   = -999
    witsun1EndDiff     = -999
    witsun1Ratio       = -999
    witsun1Length      = -999 
    witsun2StartDiff   = -999
    witsun2EndDiff     = -999
    witsun2Ratio       = -999
    witsun2Length      = -999     
    summerStartDiff    = -999
    summerEndDiff      = -999
    summerRatio        = -999
    summerLength       = -999     
    autumnStartDiff    = -999
    autumnEndDiff      = -999
    autumnRatio        = -999
    autumnLength       = -999 
    nextChrisStartDiff = -999
    nextChrisEndDiff   = -999
    nextChrisRatio     = -999
    nextChrisLength    = -999     
        
    if month == 1 and day < 11:
        chrisStartDiff = (dateVal-chrisStart).days     
        chrisEndDiff   = (dateVal-chrisStart).days
        chrisLength     = 13
        if chrisStartDiff*chrisEndDiff <= 0:
            chrisRatio    = chrisStartDiff/chrisLength

    if (month == 1 and day > 25) or ( month ==2 and day < 15):
        winterStartDiff = (dateVal-winterStart).days     
        winterEndDiff   = (dateVal-winterStart).days
        winterLength     = 6
        if winterStartDiff*winterEndDiff <= 0:
            winterRatio    = winterStartDiff/winterLength
                          
    if (month == 3 and day > 22) or (month == 4 and day < 19):
        easterStartDiff = (dateVal-easterStart).days     
        easterEndDiff   = (dateVal-easterEnd).days
        easterLength    = 13
        if easterStartDiff*easterEndDiff <= 0:
            easterRatio    = easterStartDiff/easterLength         
        
    if (month == 5 and day > 12 and day < 18): 
        witsun1StartDiff = (dateVal-witsun1Start).days     
        witsun1EndDiff   = (dateVal-witsun1End).days
        witsun1Length   = 1
        witsunCount     = 1
        if witsun1StartDiff*witsun1EndDiff <= 0:
            witsun1Ratio    = witsun1StartDiff/witsun1Length 
            
    if (month == 5 and day > 12 and day < 18):  
        witsun2StartDiff = (dateVal-witsun2Start).days     
        witsun2EndDiff   = (dateVal-witsun2End).days
        witsun2Length    = 1
        witsunCount      = 1
        if witsun2StartDiff*witsun2EndDiff <= 0:
            witsun2Ratio    = witsun2StartDiff/witsun2Length             
            
    if (month == 7  and day > 5 ) or  (month == 8 and day < 29): 
        summerStartDiff = (dateVal-summerStart).days     
        summerEndDiff   = (dateVal-summerEnd).days
        summerLength    = 40 
        if summerStartDiff*summerEndDiff <= 0:
            summerRatio    = summerStartDiff/summerLength             
            
    if (month == 9 and day > 27) or (month == 10 and day < 25): 
        autumnStartDiff = (dateVal-autumnStart).days     
        autumnEndDiff   = (dateVal-autumnEnd).days
        autumnLength    = 13
        if autumnStartDiff*autumnEndDiff <= 0:
            autumnRatio    = autumnStartDiff/autumnLength        
                 
    if month == 12 and day > 15: 
        nextChrisStartDiff = (dateVal-nextChrisStart).days     
        nextChrisEndDiff   = (dateVal-nextChrisEnd).days
        nextChrisLength = 11
        if nextChrisStartDiff*nextChrisEndDiff <= 0:
            nextChrisRatio    = nextChrisStartDiff/nextChrisLength  
                              
 
    retVal = np.zeros((1,33),dtype=float)
 
    retVal[0,0] =  chrisStartDiff
    retVal[0,1] =  chrisEndDiff 
    retVal[0,2] =  chrisRatio
    retVal[0,3] =  chrisLength
    retVal[0,4] =  winterStartDiff 
    retVal[0,5] =  winterEndDiff 
    retVal[0,6] =  winterRatio
    retVal[0,7] =  winterLength 
    retVal[0,8] =  easterStartDiff 
    retVal[0,9] =  easterEndDiff
    retVal[0,10] = easterRatio 
    retVal[0,11] = easterLength
    retVal[0,12] = witsunCount
    retVal[0,13] = witsun1StartDiff 
    retVal[0,14] = witsun1EndDiff
    retVal[0,15] = witsun1Ratio
    retVal[0,16] = witsun1Length  
    retVal[0,17] = witsun2StartDiff 
    retVal[0,18] = witsun2EndDiff
    retVal[0,19] = witsun2Ratio 
    retVal[0,20] = witsun2Length 
    retVal[0,21] = summerStartDiff
    retVal[0,22] = summerEndDiff 
    retVal[0,23] = summerRatio    
    retVal[0,24] = summerLength
    retVal[0,25] = autumnStartDiff 
    retVal[0,26] = autumnEndDiff
    retVal[0,27] = autumnRatio
    retVal[0,28] = autumnLength 
    retVal[0,29] = nextChrisStartDiff 
    retVal[0,30] = nextChrisEndDiff
    retVal[0,31] = nextChrisRatio 
    retVal[0,32] = nextChrisLength 
    
    
    return retVal
        
    
    
                                                                                                                                                                                                                                                                                                                                              
