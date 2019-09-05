#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 15:30:26 2018

@author: carolinadepasquale
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import itertools
import statistics as stat
import scipy.stats as stats
#import scipy.stats.mstats as mstats
import copy
import random
from math import ceil
import seaborn as sns
#from sklearn import datasets
from sklearn import preprocessing
from scipy import signal
import scipy.fftpack as fftpack
import statsmodels.tsa.stattools as stattools
from statsmodels.graphics import tsaplots
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import os
import re
#%% FUNCTIONS START
# The windowing functions already work in seconds!!
#%% Get medians for every 'bin', not windowed
def medianGrouping(data,stepsize): #stepsize should be numerical
    steps = np.arange(0, max(data['frameTime'] + stepsize), stepsize)
    labels = np.array(list(map(str, steps[0:len(steps)-1])))
    
    data['Categories'] = pd.cut(data['frameTime'], bins=steps, labels=labels)
    medianData= data.groupby(pd.cut(data['frameTime'], bins=steps, labels=labels)).median()
    return(medianData)
#%% grouping with Standard deviation 
def binGrouping(data,stepsize): #stepsize should be numerical
    steps = np.arange(0, max(data['frameTime'] + stepsize), stepsize)
    labels = np.array(list(map(str, steps[0:len(steps)-1])))
    
    data['Categories'] = pd.cut(data['frameTime'], bins=steps, labels=labels)
    stdData= data.groupby(pd.cut(data['frameTime'], bins=steps, labels=labels)).std()
    return(stdData)

#%% find threshold function
# most of the spurious data is on the lower end, but it's more spread in the higher tail. 
def findThresholds(data,column,quantile):
    quantilelist = []
    for x in data:
        quantilelist.append(data[x][str(column)].quantile(quantile))
        
    if quantile > .5:
        threshold = np.mean(quantilelist) + np.std(quantilelist)
    elif quantile < .5:
        threshold = np.mean(quantilelist) - np.std(quantilelist)
    else:
        print('it broke!')
    return threshold
#%% remove outliers function
def removeOutliersOld(data, columns, loudTresh, F0tresh, F0top, F0bottom):
    loudnessCheck = data.pcm_loudness_sma < loudTresh
    F0Check = data.F0final_sma < F0tresh
    F0topOutliers = data.F0final_sma > F0top
    F0bottomOutliers  = data.F0final_sma < F0bottom
    for y in columns:
        data[y][loudnessCheck] = np.nan
        data[y][F0Check] = np.nan
        data[y][F0topOutliers] = np.nan
        data[y][F0bottomOutliers] = np.nan
    return data
#%%
def removeOutliers(dataRaw, columns, F0tresh, topQuantileF0, bottomQuantileF0, topLoud, bottomLoud, column1 = 'F0final_sma', column2 = 'pcm_loudness_sma'):
    data = copy.deepcopy(dataRaw)
    for x in data:
        F0Check = data[x].F0final_sma < F0tresh
        for y in columns:
            data[x][y][F0Check] = np.nan
        print('first',x)
    F0top = findThresholds(data,column1,topQuantileF0)
    F0bottom = findThresholds(data,column1,bottomQuantileF0)
    LoudTop = findThresholds(data,column2,topLoud)
    LoudBottom = findThresholds(data,column2,bottomLoud)
    for x in data:
        F0topOutliers = data[x].F0final_sma > F0top
        F0bottomOutliers  = data[x].F0final_sma < F0bottom
        loudTopOut = data[x].pcm_loudness_sma > LoudTop
        loudBottomOut = data[x].pcm_loudness_sma < LoudBottom
        for y in columns:
            data[x][y][F0topOutliers] = np.nan
            data[x][y][F0bottomOutliers] = np.nan
            data[x][y][loudTopOut] = np.nan
            data[x][y][loudBottomOut] = np.nan
        print('second', x)
    return data

#%% convert audio to seconds:
def convertTime(time, frequency = 100):
    convertedTime = time*frequency
    return convertedTime
#%% Chiara Windowed corr final
def windowsWithSteps(step, max_time):
    # Rounds max time
    max_time = step * round(float(max_time) / step)
    total_windows = ceil(float(max_time) / step)
    window_starts = np.arange(0, max_time, int(float(max_time) / total_windows))
#
    return window_starts

def correlate_windows(data1, data2, window_size, step, max_time,overlap_percentage):
    window_starts = windowsWithSteps(step, max_time)
    total_windows = len(window_starts)
    correlations = pd.DataFrame(np.nan,
                                index=np.arange(0, total_windows),
                                columns=['window_start', 'window_end', 'correlation_coeff', 'p_value', 'percent_overlap'])
    for n_window in np.arange(0, total_windows):
        window_start_time = window_starts[n_window]
        window_end_time = window_start_time + window_size
        temp1 = data1[data1['window_start'].between(window_start_time, window_end_time) | data1['window_end'].between(window_start_time, window_end_time)]
        temp2 = data2[data2['window_start'].between(window_start_time, window_end_time) | data2['window_end'].between(window_start_time, window_end_time)]
        percent_overlap = 1 - (pd.isnull(temp1['median_F0']) | pd.isnull(temp2['median_F0'])).sum() / len(temp1['median_F0'])
        if percent_overlap >= overlap_percentage:
            corr = stats.spearmanr(temp1['median_F0'], temp2['median_F0'], nan_policy="omit")
        else:
            corr = [np.nan, np.nan]
        correlations.loc[n_window] = [window_start_time, window_end_time, corr[0], corr[1], percent_overlap]       
    return correlations
#%%Corr windows general

def correlate_windows_general(data1, data2, featureColumn1, featureColumn2, window_size, step, max_time,overlap_percentage):
    window_starts = windowsWithSteps(step, max_time)
    total_windows = len(window_starts)
    correlations = pd.DataFrame(np.nan,
                                index=np.arange(0, total_windows),
                                columns=['window_start', 'window_end', 'correlation_coeff', 'p_value', 'percent_overlap'])

    for n_window in np.arange(0, total_windows):
        window_start_time = window_starts[n_window]
        window_end_time = window_start_time + window_size
        temp1 = data1[data1['window_start'].between(window_start_time, window_end_time) | data1['window_end'].between(window_start_time, window_end_time)]
        temp2 = data2[data2['window_start'].between(window_start_time, window_end_time) | data2['window_end'].between(window_start_time, window_end_time)]
        temp1,temp2 = checkSizes(temp1,temp2)
        percent_overlap = 1 - (pd.isnull(temp1[featureColumn1]) | pd.isnull(temp2[featureColumn2])).sum() / len(temp1[featureColumn1])
        if percent_overlap >= overlap_percentage:
            try:
                corr = stats.spearmanr(temp1[featureColumn1], temp2[featureColumn2], nan_policy="omit")
            except:
                corr = [np.nan, np.nan]
        else:
            corr = [np.nan, np.nan]

        correlations.loc[n_window] = [window_start_time, window_end_time, corr[0], corr[1], percent_overlap]
        
    return correlations   
        
#%% Windows with steps slope
def windowsWithStepsSlope(data, timeColumn, featureColumn, window_size, step, max_time): #defaults to median, requires NO NANS
    # Rounds max time
    max_time = step * round(float(max_time) / step)
    total_windows = ceil(float(max_time) / step)
    windowedData = np.zeros(total_windows)
    windowedData.fill(np.nan)
    windows = np.zeros((total_windows, 2))
    max_time_dataset = max(data[timeColumn])

    for n_window in np.arange(0, total_windows):
        window_start_time = n_window * step
        window_end_time = window_start_time + window_size
        if window_start_time <= max_time_dataset:
            temp = data[(data[timeColumn] >= window_start_time) & (data[timeColumn] < window_end_time)]
            if len(temp) > 0:
                slope, intercept, r_value, p_value, std_err = stats.linregress(temp[timeColumn],temp[featureColumn])
                windowedData[n_window] = slope
        windows[n_window] = [window_start_time, window_end_time]

    windowedData = pd.DataFrame({"window_start": windows[:, 0], "window_end": windows[:, 1], 'slope': windowedData})

    return windowedData
#%%Chiara's windowing
def windowsWithStepsChiara(data, window_size, step, max_time, stattype=np.nanmedian): #defaults to median
    # Rounds max time
    max_time = step * round(float(max_time) / step)
    total_windows = ceil(float(max_time) / step)
    windowedData = np.zeros(total_windows)
    windowedData.fill(np.nan)
    windows = np.zeros((total_windows, 2))
    max_time_dataset = max(data['frameTime'])
    for n_window in np.arange(0, total_windows):
        window_start_time = n_window * step
        window_end_time = window_start_time + window_size
        if window_start_time <= max_time_dataset:
            temp = data[(data['frameTime'] >= window_start_time) & (data['frameTime'] < window_end_time)]
            if len(temp) > 0:
                windowedData[n_window] = stattype(temp['F0final_sma'])
        windows[n_window] = [window_start_time, window_end_time]
    windowedData = pd.DataFrame({"window_start": windows[:, 0], "window_end": windows[:, 1], "median_F0": windowedData})
    return windowedData

#%% windowed cross correlation 
def correlate_windows_cross(data1, data2, featureColumn1, featureColumn2, window_size, step, max_time,overlap_percentage):
    window_starts = windowsWithSteps(step, max_time)
    total_windows = len(window_starts)
    correlations = pd.DataFrame(np.nan,
                                index=np.arange(0, total_windows),
                                columns=['window_start', 'window_end', 'lag', 'max_coeff', 'percent_overlap'])

    for n_window in np.arange(0, total_windows):
        window_start_time = window_starts[n_window]
        window_end_time = window_start_time + window_size
        #temp1 = data1[data1['frameTime'].between(window_start_time, window_end_time)]
        temp1 = data1[data1['window_start'].between(window_start_time, window_end_time) | data1['window_end'].between(window_start_time, window_end_time)]
        #temp2 = data2[data2['frameTime'].between(window_start_time, window_end_time)]
        temp2 = data2[data2['window_start'].between(window_start_time, window_end_time) | data2['window_end'].between(window_start_time, window_end_time)]
        percent_overlap = 1 - ((temp1[featureColumn1]==0) | (temp2[featureColumn2]==0)).sum() / len(temp1[featureColumn1])
        #percent_overlap = 1 - (pd.isnull(temp1['median_F0']) | pd.isnull(temp2['median_F0'])).sum() / len(temp1['median_F0'])
        if percent_overlap >= overlap_percentage:
            try:
                corr = signal.correlate(temp1[featureColumn1], temp2[featureColumn2])
                lag = corr.argmax() - (len(temp1) - 1)
                maxcoeff = corr.max()
            except:
                lag = np.nan
                maxcoeff = np.nan
        else:
            lag = np.nan
            maxcoeff = np.nan

        correlations.loc[n_window] = [window_start_time, window_end_time, lag, maxcoeff, percent_overlap]
        
    return correlations  

#%%window general
def windowsWithStepsGeneral(data, timeColumn, featureColumn, window_size, step, max_time, stattype=np.nanmedian): #defaults to median
    # Rounds max time
    max_time = step * round(float(max_time) / step)
    total_windows = ceil(float(max_time) / step)
    windowedData = np.zeros(total_windows)
    windowedData.fill(np.nan)
    windows = np.zeros((total_windows, 2))
    max_time_dataset = max(data[timeColumn])

    for n_window in np.arange(0, total_windows):
        window_start_time = n_window * step
        window_end_time = window_start_time + window_size
        if window_start_time <= max_time_dataset:
            temp = data[(data[timeColumn] >= window_start_time) & (data[timeColumn] < window_end_time)]
            if len(temp) > 0:
                windowedData[n_window] = stattype(temp[featureColumn])
        windows[n_window] = [window_start_time, window_end_time]

    windowedData = pd.DataFrame({"window_start": windows[:, 0], "window_end": windows[:, 1], featureColumn: windowedData})

    return windowedData
#%% Make stuff the same size 
def checkSizes(data1, data2):
    shortData = data1 if len(data1) < len(data2) else data2
    longData = data1 if len(data1) > len(data2) else data2
    nanDf = pd.DataFrame({shortData.columns[0]: np.nan, 
                          "window_start": longData['window_start'][len(shortData):len(longData)],
                          "window_end": longData['window_end'][len(shortData):len(longData)]})
    shortData = shortData.append(nanDf, ignore_index=True)
    if len(data1) < len(data2):
        data1 = shortData
    elif len(data1) > len(data2):
        data2 = shortData
    else:
        pass
    return(data1,data2)
#%%make stuff same size
def changeSizes(data1, data2):
    shortData = data1 if len(data1) < len(data2) else data2
    longData = data1 if len(data1) > len(data2) else data2
    nanDf = pd.DataFrame({'frameTime': longData['frameTime'][len(shortData):len(longData)], 
                          "F0final_sma": np.nan,'voicingFinalUnclipped_sma': np.nan, 'pcm_loudness_sma': np.nan})
    shortData = shortData.append(nanDf, ignore_index=True)
    if len(data1) < len(data2):
        data1 = shortData
    elif len(data1) > len(data2):
        data2 = shortData
    else:
        pass
    return(data1,data2)
    
#%% Make two array the same size, append 0 instead of nan
def changeSizesArray(data1, data2):
    shortData = data1 if len(data1) < len(data2) else data2
    longData = data1 if len(data1) > len(data2) else data2
    filler = pd.Series(np.zeros(len(longData)-len(shortData)))
    shortData = shortData.append(filler, ignore_index=True)
    if len(data1) < len(data2):
        data1 = shortData
    elif len(data1) > len(data2):
        data2 = shortData
    else:
        pass
    return(data1,data2)
#%% get turn mean function
def getTurnMeanDF(df):
    turnlen = np.zeros(len(df))
    for turn in range(len(df)):
        turnlen[turn] = len(df[turn])
    return np.mean(turnlen)
#%% get turn mean dictionary
def getTurnMean(dictionary):
    turnlen = np.zeros(len(dictionary))
    for index, turn in zip(range(len(dictionary)),dictionary):
        turnlen[index] = len(dictionary[turn])
    return np.mean(turnlen)
#%% get turn length function
def getTurnLenDF(df):
    turnlen = np.zeros(len(df))
    for turn in range(len(df)):
        turnlen[turn] = len(df[turn])
    return turnlen
#%%
def getTurnLen(dictionary):
    turnlen = np.zeros(len(dictionary))
    for index, turn in zip(range(len(dictionary)),dictionary):
        turnlen[index] = len(dictionary[turn])
    return turnlen 
#%%Chiara's script turns 
def find_turns(data):
    speakers = list(data.columns.values)
    speaker = speakers[0]
    other_speaker = speakers[1]

    turnStarts = []
    turnEnds = []
    for index, row in data.iterrows():
        if index == 0:
            if ~np.isnan(row[speaker]):
                turnStart = True
                lastStart = index
                turnStarts.append(speaker)
                turnEnds.append(index)
            else:
                turnStart = False
        else:
            if np.isnan(row[speaker]):
                if turnStart is True and ~np.isnan(row[other_speaker]):
                    turnStart = False
                    turnStarts.append(other_speaker)
                    turnEnds.append(index)
            else:
                if turnStart is False:
                    turnStart = True
                    lastStart = index
                    turnStarts.append(speaker)
                    turnEnds.append(index)

    if turnStart is False:
        turnStarts.append(other_speaker)
        turnEnds.append(index)

    turnChanges = pd.DataFrame({'speaker': turnStarts,
                                'starts_speaking': turnEnds},
                               columns=["speaker", "starts_speaking"])

    return turnChanges

#REQUIRES data to be in this format:
#pat = patientData.F0final_sma
#doc = doctorData.F0final_sma
#dataa = pd.DataFrame({'pat': pat, 'doc': doc})
#%%
def find_all_turns(data1,data2,turn_changes_data,column): #turn_changes_data should be provided as a list, e.g. turnChanges['starts_speaking']
    turns1 = {}
    turns2 = {}
    start = 0
    counter = 0
    for x in turn_changes_data:
        end = x-1
        temp1 = data1[data1['frameIndex'].between(start, end)]
        temp2 = data2[data2['frameIndex'].between(start, end)]
        if temp1[column].notnull().any():
            turns1['turn{0}'.format(counter)] = temp1
        elif temp2[column].notnull().any():
            turns2['turn{0}'.format(counter)] = temp2
        start = x
        counter+=1
    return(turns1,turns2)
#%% Descriptive statistics 
def describe_df(data, cols):
    IQR = pd.DataFrame(np.nan,index = ['IQR'],columns = data[cols].columns)
    for column in data[cols]:
        IQR[column] = stats.iqr(data[column], nan_policy='omit')
    data_desc_stats = data[cols].describe()
    data_desc_stats = data_desc_stats.append(IQR)
    return(data_desc_stats)
    
#%% Normalise data
def normalise(data):
    normData = copy.deepcopy(data)
    normData = (data - data.mean()) / (data.max() - data.min())
    return(normData)
    
#%% Standardise data
def standardise(data):
    stdData = copy.deepcopy(data)
    stdData = (data - data.mean()) / data.std()
    return(stdData)

#%% Create dataframe of the same column for plotting. The column that one wants to isolate should be passed as string for the column argument, 'args' will
#   take multiple rows. If all rows should be selected use ':'
def make_df(data,column,*args):
    dataframe = {}
    for x in data:
        try:
            dataframe[x] = data[x][column][list(args)]
        except:
            dataframe[x] = data[x][column]
    dataframe = pd.DataFrame(dataframe)
    return(dataframe)
#%% Convert simple dictionary into a dataframe, and create a session column that matches the name in the other data, plus add index column
def makeDF_renameColumns(data, dataCol):
    df = pd.DataFrame(list(data.items()), columns = ['session', dataCol] )
    df[dataCol] = [np.asscalar(x) for x in df[dataCol]]
    df['sessionName'] = np.zeros(len(df))
    df['index'] = np.zeros(len(df))
    for idx, name in enumerate(df['session']):
        x = re.search(r'seduta+\d+_', name)
        xN = int(x[0][6:-1])
        if 'seduta{0}'.format(xN) in name:
            df['sessionName'].loc[idx] = 'session{0}'.format(xN)
            df['index'].loc[idx] = xN
    return df
#%%Rename with Regex general
def rename(data, newname = 'session{0}', oldname = 'seduta{0}', regex = r'seduta+\d+_'):
    if type(data) == str:
        nameList = [None]
        x = re.search(regex, data)
        try:
            xN = int(x[0][6:-1])
        except:
            xN = int(x[0])
        nameList = newname.format(xN)
    else:
        nameList = [None]*len(data)
        for idx, name in enumerate(data):
            x = re.search(regex, name)
            xN = int(x[0][6:-1])
            if oldname.format(xN) in name:
                nameList[idx] = newname.format(xN)
    return nameList
#%% create new dic with renamed keys
def newDicCopy(dictionary, **regexargs):
    newDict = {}
    for key in dictionary.keys():
        newKey = rename(key)
        newDict[newKey] = dictionary[key]
    return newDict
#%% Outputting function simple
def outputCSV(data1, data2, filename1, filename2, sep):
    outputFilenamePat = filename1 #example: 'examplePat.csv'
    outputFilenameDoc = filename2

    outputDataPat = data1 #patientDataMediansLarge[['frameTime','F0final_sma']] #example
    outputDataDoc = data2 #doctorDataMediansLarge[['frameTime','F0final_sma']] #example

    outputDataPat.to_csv(outputFilenamePat, sep= sep)
    outputDataDoc.to_csv(outputFilenameDoc, sep= sep)
#%% Check if a directory exists, if not, make it
def check_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
#%%  outputting function with folder creation
def outFunction(data1, data2, folderPath, outFolder):
    check_make_dir(outFolder+folderPath)
    for x,y in zip(data1, data2):
        outputCSV(data1[x],data2[y],outFolder+folderPath+'/{0}.csv'.format(x),outFolder+folderPath+'/{0}.csv'.format(y), ',')

#%% FUNCTIONS END