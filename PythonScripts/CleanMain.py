# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat
import scipy.stats as stats
import copy
import random
from math import ceil
import seaborn as sns
from sklearn import preprocessing
from scipy import signal
import scipy.fftpack as fftpack
import statsmodels.tsa.stattools as stattools
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics import tsaplots
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import auc
import os
import python_speech_features
import scipy.io
import re
#%% Script import
dir_path = os.path.dirname(os.path.realpath('/Volumes/GoogleDrive/Il mio Drive/Ph.D./PsychologicalSyncStudy/AnalysisScripts/PythonScripts/CleanMain.py'))
os.chdir(dir_path)
import Functions
#%% DATA IMPORTS AND PREPROCESSING

#%%
max_session_n = 17
inFolder = '/Volumes/GoogleDrive/Il mio Drive/Ph.D./PsychologicalSyncStudy/OpenSmileAnalysis/'
outFolder = '/Volumes/GoogleDrive/Il mio Drive/Ph.D./PsychologicalSyncStudy/AnalysisOutput/'
#%% Import all sessions in two dictionaries AUDIO
allPatDataRaw = {}
allDocDataRaw = {}
for x in range(1,max_session_n):
    try:
        allPatDataRaw["session{0}".format(x)] = pd.read_csv(inFolder+'S{0}/S{0}prosody1.csv'.format(x), sep = ';')
        allDocDataRaw["session{0}".format(x)] = pd.read_csv(inFolder+"S{0}/S{0}prosody2.csv".format(x), sep = ';')
    except:
        x+1
        
#%% Session list
session_list = []
validSessions = []
for x in range(1,max_session_n):
    try:
        allPatDataRaw["session{0}".format(x)]
        session_list.append('session{0}'.format(x))
        validSessions.append(x)
    except:
        x+1
        
#%% data exploration plot
fig, axes = plt.subplots(nrows=5, ncols=3, figsize = (20,20), sharex=False)
axes_list = [item for sublist in axes for item in sublist] 
for x in allPatDataRaw:
    ax = axes_list.pop(0)
    sns.distplot(allPatDataRaw[x].F0final_sma.dropna(), ax=ax, label = 'pat')
    sns.distplot(allDocDataRaw[x].F0final_sma.dropna(), ax=ax, label = 'doc')
    ax.set_xticks(range(0,ceil(max(allPatDataRaw[x].F0final_sma.dropna())),20))
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    ax.set_title(x)
    ax.legend(loc='upper right')
    # Just to position axes labels:
    plt.sca(ax)
    plt.xticks(rotation=90)
    plt.tight_layout()
for ax in axes_list:
    ax.remove()
#%% data exploration plot loud
fig, axes = plt.subplots(nrows=5, ncols=3, figsize = (20,20), sharex=False)
axes_list = [item for sublist in axes for item in sublist] 
for x in allPatDataRaw:
    ax = axes_list.pop(0)
    sns.distplot(allPatDataRaw[x].pcm_loudness_sma.dropna(), ax=ax, label = 'pat')
    sns.distplot(allDocDataRaw[x].pcm_loudness_sma.dropna(), ax=ax, label = 'doc')
    ax.set_xticks(np.arange(0,ceil(max(allPatDataRaw[x].pcm_loudness_sma.dropna())),.1))
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    ax.set_title(x)
    ax.legend(loc='upper right')
    # Just to position axes labels:
    plt.sca(ax)
    plt.xticks(rotation=90)
    plt.tight_layout()
for ax in axes_list:
    ax.remove()
#%% data descriptions for thesis:
typestat = [np.mean,np.std]
summariesReport = ['mean','std','min','max','IQR']
resultsPat = {}
for x in summariesReport:
    resultsPat[x] = {}
    for y in typestat:
        resultsPat[x][y] = y([patDescriptiveStats[k][patDescriptiveStats[k].index == x].get('pcm_loudness_sma', 'NaN') for k in patDescriptiveStats])
#mean_general = mean([patDescriptiveStats[k][patDescriptiveStats[k].index == 'mean'].get('pcm_loudness_sma', 'NaN')[1] for k in patDescriptiveStats])

#%% Substitute all 0s with nans only for F0 and remove outliers
cols = ["F0final_sma","voicingFinalUnclipped_sma","pcm_loudness_sma"]

F0Check = 10
F0topQuantile = .95
F0bottomQuantile = .15
topLoud = .95
bottomLoud = .05
allPatData = Functions.removeOutliers(allPatDataRaw, cols, F0Check, F0topQuantile, F0bottomQuantile, topLoud, bottomLoud)
allDocData = Functions.removeOutliers(allDocDataRaw, cols, F0Check, F0topQuantile, F0bottomQuantile, topLoud, bottomLoud)


#%% data exploration plot
fig, axes = plt.subplots(nrows=5, ncols=3, figsize = (20,20), sharex=False)
axes_list = [item for sublist in axes for item in sublist] 
for x in allPatData:
    ax = axes_list.pop(0)
    sns.distplot(allPatData[x].F0final_sma.dropna(), ax=ax, label = 'pat')
    sns.distplot(allDocData[x].F0final_sma.dropna(), ax=ax, label = 'doc')
    ax.set_xticks(range(0,ceil(max(allPatData[x].F0final_sma.dropna())),20))
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    ax.set_title(x)
    ax.legend(loc='upper right')
    # Just to position axes labels:
    plt.sca(ax)
    plt.xticks(rotation=90)
    plt.tight_layout()
for ax in axes_list:
    ax.remove()
#%% data exploration plot loud
fig, axes = plt.subplots(nrows=5, ncols=3, figsize = (20,20), sharex=False)
axes_list = [item for sublist in axes for item in sublist] 
for x in allPatData:
    ax = axes_list.pop(0)
    sns.distplot(allPatData[x].pcm_loudness_sma.dropna(), ax=ax, label = 'pat')
    sns.distplot(allDocData[x].pcm_loudness_sma.dropna(), ax=ax, label = 'doc')
    ax.set_xticks(np.arange(0,ceil(max(allPatData[x].pcm_loudness_sma.dropna())),.1))
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    ax.set_title(x)
    ax.legend(loc='upper right')
    # Just to position axes labels:
    plt.sca(ax)
    plt.xticks(rotation=90)
    plt.tight_layout()
for ax in axes_list:
    ax.remove()
#%% Add mel scale to DFs
for x,y in zip(allPatData, allDocData):
    allPatData[x]['melScale'] = python_speech_features.base.hz2mel(allPatData[x]["F0final_sma"])
    allDocData[y]['melScale'] = python_speech_features.base.hz2mel(allDocData[y]["F0final_sma"]) 

#%% import Physio
physioFolder = '/Volumes/GoogleDrive/Il mio Drive/Ph.D./PsychologicalSyncStudy/full_export2/'
physioRaw = {}
for x in range(1,max_session_n):
    if x<10:
        try:
            physioRaw["session{0}".format(x)] = pd.read_csv(physioFolder+'seduta0{0}_2015_10Hz.csv'.format(x), sep = ';')
        except:
            x+1  
    else:
        try:
            physioRaw["session{0}".format(x)] = pd.read_csv(physioFolder+'seduta{0}_2015_10Hz.csv'.format(x), sep = ';')
        except:
            x+1  
#%% Select appropriate columns and add time
allPhysio = {}
for x in physioRaw:
    allPhysio[x] = physioRaw[x][['PPG_patient','PPG_clinician','SC_patient','SC_clinician','HF_patient','HF_clinician']]
    #create time array for the physio data
    physio_Time = np.arange(0, (len(physioRaw[x])/10), .1)
    allPhysio[x]['Time'] = physio_Time
    del physio_Time
#%% Import Anxiety and Empathy scores
empathyScores = pd.read_excel("/Volumes/GoogleDrive/Il mio Drive/Ph.D./PsychologicalSyncStudy/Scores/CC_EUS.xlsx") 
anxietyScore = pd.read_excel("/Volumes/GoogleDrive/Il mio Drive/Ph.D./PsychologicalSyncStudy/Scores/CC_STAI.xlsx")
empathyScores['normPat'] = Functions.normalise(empathyScores['pz'])
empathyScores['normDoc'] = Functions.normalise(empathyScores['ter'])
empathyScores['stdPat'] = Functions.standardise(empathyScores['pz'])
empathyScores['stdDoc'] = Functions.standardise(empathyScores['ter'])
anxietyScore['normScoring'] = Functions.normalise(anxietyScore['scoring'])
anxietyScore['stdScoring'] = Functions.standardise(anxietyScore['scoring'])

empathyScores['session'] = np.zeros(len(empathyScores))
for x in empathyScores['seduta']:
    empathyScores['session'].iloc[x-1] = Functions.rename(str(empathyScores['seduta'].iloc[x-1]), oldname = '{0}', regex = r'\d+')
#%% Select only the sessions I have
validSessions = [1,2,4,5,7,9,10,11,12,13,14,15,16]
session_list = []
for x in validSessions:
    session_list.append('session{0}'.format(x))
validEmpathySessions = empathyScores.loc[empathyScores['seduta'].isin(validSessions)]
validAnxietySessions = anxietyScore.loc[anxietyScore['SEDUTA'].isin(validSessions)]

validEmpathySessions.index = validEmpathySessions['seduta']
validAnxietySessions.index = validAnxietySessions['SEDUTA']
#redundant but future code works on this
validEmpathySessions['sessionName'] = session_list
validAnxietySessions['sessionName'] = session_list
validAnxietySessions['session'] = session_list
#%% import MFCC and vowel space
patMatFolder = '/Volumes/GoogleDrive/Il mio Drive/Ph.D./PsychologicalSyncStudy/Audio_gender_sep/OutPat/'
#docMatFolder = '/Volumes/GoogleDrive/Il mio Drive/Ph.D./PsychologicalSyncStudy/Audio_gender_sep/OutDoc/'
patMFCCvowels = {}
#docMFCCvowels = {}
for file in os.listdir(patMatFolder):
    try:
        patMFCCvowels[file] = scipy.io.loadmat(patMatFolder+file)
        print(file)
    except:
        print(f'This is not a mat file: {file}')
#for file in os.listdir(docMatFolder):
#    docMFCCvowels[file] = scipy.io.loadmat(docMatFolder+file)
##%% Putting them in a dictionary of dataframes with column headers and everything.
#allPatMFCC = {}
allPatVowel = {'formants' : {}, 'ratio' : {}}
#allDocMFCC = {}
#allDocVowel = {'formants' : {}, 'ratio' : {}}
#MFCCcols = ['MFCC1','MFCC2','MFCC3','MFCC4','MFCC5','MFCC6','MFCC7','MFCC8','MFCC9','MFCC10','MFCC11','MFCC12','MFCC13']
formantCols = ['F1','F2','F3','F4','F5']
#for x,y in zip(docMFCCvowels,patMFCCvowels):
#    if 'MFCC' in x and 'MFCC' in y:
#        allDocMFCC[x] = pd.DataFrame(docMFCCvowels[x]['MFCC'].transpose(), columns = MFCCcols)
#        allPatMFCC[y] = pd.DataFrame(patMFCCvowels[y]['MFCC'].transpose(), columns = MFCCcols)
#    else:
#        allDocVowel['formants'][x] = pd.DataFrame(docMFCCvowels[x]['formants'], columns = formantCols)
#        allDocVowel['ratio'][x] = docMFCCvowels[x]['vowelSpace']
#        allPatVowel['formants'][y] = pd.DataFrame(patMFCCvowels[y]['formants'], columns = formantCols)
#        allPatVowel['ratio'][y] = patMFCCvowels[y]['vowelSpace']
        
for x in patMFCCvowels:
    allPatVowel['formants'][x] = pd.DataFrame(patMFCCvowels[x]['formants'], columns = formantCols)
    allPatVowel['ratio'][x] = patMFCCvowels[x]['vowelSpace']
    print(x)
##%% Rename Sessions so as to comply with the rest of the code 
#patMFCCs = Functions.newDicCopy(allPatMFCC)
#docMFCCs = Functions.newDicCopy(allDocMFCC)
#
#
patFormants = Functions.newDicCopy(allPatVowel['formants'])
patRatios = Functions.newDicCopy(allPatVowel['ratio'])
#docFormants = Functions.newDicCopy(allDocVowel['formants'])
#
#del patMFCCvowels, docMFCCvowels, allPatVowel, allDocVowel, allPatMFCC, allDocMFCC

#%% Descriptive stats AUDIO
cols = ["F0final_sma","voicingFinalUnclipped_sma","pcm_loudness_sma"]#, 'melScale']
patDescriptiveStats = {}
docDescriptiveStats = {}
for x,y in zip(allPatData, allDocData):
    patDescriptiveStats[x] = Functions.describe_df(allPatData[x], cols)
    docDescriptiveStats[y] = Functions.describe_df(allDocData[y], cols)

#%% Descriptive stats PHYSIO
physioCols = ['PPG_patient','PPG_clinician','SC_patient','SC_clinician','HF_patient','HF_clinician']
physioDescriptiveStats = {}
for x in allPhysio:
    physioDescriptiveStats[x] = Functions.describe_df(allPhysio[x], physioCols)
    
#%% Remove Nans for correlations
allPatDataNoNans = {}
allDocDataNoNans = {}
for x,y in zip(allPatData, allDocData):
    allPatDataNoNans[x] = copy.deepcopy(allPatData[x]).replace({np.nan:0})
    allDocDataNoNans[y] = copy.deepcopy(allDocData[y]).replace({np.nan:0})

#%% Data Binning
allBinnedPat = {}
allBinnedDoc = {}
for x,y in zip(allPatData, allDocData):
    allBinnedPat['medians{0}'.format(x)] = Functions.medianGrouping(allPatData[x],1)
    allBinnedDoc['medians{0}'.format(y)] = Functions.medianGrouping(allDocData[y],1)

#%% Data windowing
allWindowedPat = {}
allWindowedDoc = {}
frame_size = 10
step = 5
for x in range(1,max_session_n):
    try:
        max_pat = max(allPatData["session{0}".format(x)]['frameTime'])
        max_doc = max(allDocData["session{0}".format(x)]['frameTime'])
        max_time = max(max_pat,max_doc)
        allWindowedPat["session{0}".format(x)] = Functions.windowsWithStepsChiara(allPatData["session{0}".format(x)], frame_size, step, max_time)
        allWindowedDoc["session{0}".format(x)] = Functions.windowsWithStepsChiara(allDocData["session{0}".format(x)], frame_size, step, max_time)
    except:
        x+1
#%% Window melscale
windowedPatMel = {}
windowedDocMel = {}
frame_size = 10
step = 5
for x in range(1,max_session_n):
    session = "session{0}".format(x)
    try:
        max_pat = max(allPatData[session]['frameTime'])
        max_doc = max(allDocData[session]['frameTime'])
        max_time = max(max_pat,max_doc)
        windowedPatMel[session] = Functions.windowsWithStepsGeneral(allPatData[session], 'frameTime', 'melScale', frame_size, step, max_time)
        windowedDocMel[session] = Functions.windowsWithStepsGeneral(allDocData[session], 'frameTime', 'melScale', frame_size, step, max_time)
    except:
        x+1

#%% Window loud
windowedPatLoud = {}
windowedDocLoud = {}
frame_size = 10
step = 5
for x in allPatData:
        max_pat = max(allPatData[x]['frameTime'])
        max_doc = max(allDocData[x]['frameTime'])
        max_time = max(max_pat,max_doc)
        windowedPatLoud[x] = Functions.windowsWithStepsGeneral(allPatData[x], 'frameTime', 'pcm_loudness_sma', frame_size, step, max_time)
        windowedDocLoud[x] = Functions.windowsWithStepsGeneral(allDocData[x], 'frameTime', 'pcm_loudness_sma', frame_size, step, max_time)

#%% windowed slope
windowedPatSlope = {}
windowedDocSlope = {}
frame_size = 10
step = 5
for x in allPatDataNoNans:
        max_pat = max(allPatDataNoNans[x]['frameTime'])
        max_doc = max(allDocDataNoNans[x]['frameTime'])
        max_time = max(max_pat,max_doc)
        windowedPatSlope[x] = Functions.windowsWithStepsSlope(allPatDataNoNans[x], 'frameTime', 'melScale', frame_size, step, max_time)
        windowedDocSlope[x] = Functions.windowsWithStepsSlope(allDocDataNoNans[x], 'frameTime', 'melScale', frame_size, step, max_time)


#%% Window Physio
windowedPhysio = {}
frame_size = 10
step = 5
for x in allPhysio:
        max_time = max(allPhysio[x]['Time'])
        windowedPhysio[x] = {}
        for item in physioCols:
            windowedPhysio[x][item] = Functions.windowsWithStepsGeneral(allPhysio[x], 'Time', item, frame_size, step, max_time)
            
#%% Batch windowed correlations
correlations = {}
frame_size = 40
step = 10
for x in range(1,max_session_n):
    try:
        max_pat = max(allPatData["session{0}".format(x)]['frameTime'])
        max_doc = max(allDocData["session{0}".format(x)]['frameTime'])
        max_time = max(max_pat,max_doc)
        correlations['session{0}'.format(x)] = Functions.correlate_windows(allWindowedPat["session{0}".format(x)], allWindowedDoc["session{0}".format(x)], frame_size, step, max_time, .2)
    except:
        x+1

#%% Loudness correlations
correlationsLoud = {}
frame_size = 60
step = 10
for x in windowedPatLoud:
        max_pat = max(allPatData[x]['frameTime'])
        max_doc = max(allDocData[x]['frameTime'])
        max_time = max(max_pat,max_doc)
        correlationsLoud[x] = Functions.correlate_windows_general(windowedPatLoud[x], windowedDocLoud[x], 'pcm_loudness_sma', 'pcm_loudness_sma', frame_size, step, max_time, .3)

#%% SLOPE correlations  
correlationsSlope = {}
frame_size = 60
step = 10
for x in windowedPatSlope:
        max_pat = max(allPatDataNoNans[x]['frameTime'])
        max_doc = max(allDocDataNoNans[x]['frameTime'])
        max_time = max(max_pat,max_doc)
        correlationsSlope[x] = Functions.correlate_windows_general(windowedPatSlope[x], windowedDocSlope[x], 'slope', 'slope', frame_size, step, max_time, .3)
  
#%% correlate physio pat
frame_size = 60
step = 10
physioAudioCorrsPat = {}
physioStuff = ['PPG','SC','HF']
for item in physioStuff:
    physioAudioCorrsPat[item] = {}
    for x in session_list:
        max_audio = max(allPatData[x]['frameTime'])
        max_physio = max(allPhysio[x]['Time'])
        physioAudioCorrsPat[item][x] = Functions.correlate_windows_general(allWindowedPat[x], windowedPhysio[x][item+'_{0}'.format('patient')], 'median_F0', item+'_{0}'.format('patient'), frame_size, step, max_time, .2)

#%% correlate physio doc
frame_size = 60
step = 10
physioAudioCorrsDoc = {}
physioStuff = ['PPG','SC','HF']
for item in physioStuff:
    physioAudioCorrsDoc[item] = {}
    for x in session_list:
        max_audio = max(allDocData[x]['frameTime'])
        max_physio = max(allPhysio[x]['Time'])
        physioAudioCorrsDoc[item][x] = Functions.correlate_windows_general(allWindowedDoc[x], windowedPhysio[x][item+'_{0}'.format('clinician')], 'median_F0', item+'_{0}'.format('clinician'), frame_size, step, max_time, .2)

#%% Shuffled correlations
shuffledCorrs = {}
frame_size = 60
step = 10
for x in allPatData:
        max_pat = max(allPatData[x]['frameTime'])
        y = random.choice(list(allDocData))
        max_doc = max(allDocData[y]['frameTime'])
        max_time = max(max_pat,max_doc)
        #random.choice(list(allDocData))
        shuffledCorrs[x] = Functions.correlate_windows_general(allWindowedPat[x], allWindowedDoc[y],'median_F0','median_F0', frame_size, step, max_time, .2)
#%% Cross correlations prep
featureColumn = 'median_F0'    
allWindowedPatNoNans = copy.deepcopy(allWindowedPat)
allWindowedDocNoNans = copy.deepcopy(allWindowedDoc)
for x,y in zip(allWindowedPatNoNans, allWindowedDocNoNans):
    allWindowedDocNoNans[x][featureColumn] = allWindowedDocNoNans[x][featureColumn].replace({np.nan:0})
    allWindowedDocNoNans[y][featureColumn] = allWindowedDocNoNans[y][featureColumn].replace({np.nan:0})
#%% cross correlations
crossCorrelations = {}
for x in allPatData:
        max_pat = max(allPatData[x]['frameTime'])
        max_doc = max(allDocData[x]['frameTime'])
        max_time = max(max_pat,max_doc)
        crossCorrelations[x] = Functions.correlate_windows_cross(allWindowedPatNoNans[x], allWindowedDocNoNans[x], featureColumn,featureColumn,frame_size, step, max_time, .2)

#%% Area under the curve???
area_under_curve_Pat = {}
area_under_curve_Doc = {}
for x in allPatDataNoNans:
    area_under_curve_Pat[x] = auc(allPatDataNoNans[x]['frameTime'], allPatDataNoNans[x]['F0final_sma'])
    area_under_curve_Doc[x] = auc(allDocDataNoNans[x]['frameTime'], allDocDataNoNans[x]['F0final_sma'])
    
norm_area_under_curveP = {}
norm_area_under_curveD = {}
for x in area_under_curve_Pat:
    pat = [allPatDataNoNans[x]['frameTime'], allPatDataNoNans[x]['F0final_sma']]
    doc = [allDocDataNoNans[x]['frameTime'], allDocDataNoNans[x]['F0final_sma']]
    norm_area_under_curveP[x]= (np.trapz(pat)/(np.trapz(pat)+np.trapz(doc)))*100
    norm_area_under_curveD[x]= (np.trapz(doc)/(np.trapz(pat)+np.trapz(doc)))*100      

#%%Speaking time
speaking_time_pat = {}
speaking_time_doc = {}

for x in allPatData:
    pat = allPatData[x].size - np.count_nonzero(np.isnan(allPatData[x]['voicingFinalUnclipped_sma']))
    doc = allDocData[x].size - np.count_nonzero(np.isnan(allDocData[x]['voicingFinalUnclipped_sma']))
    speaking_time_pat[x] = (pat/(pat+doc))*100
    speaking_time_doc[x] = (doc/(pat+doc))*100

pat = pd.DataFrame(list(speaking_time_pat.items()), columns = ['session', 'speaking_time_pat'])
doc = pd.DataFrame(list(speaking_time_doc.items()), columns = ['session', 'speaking_time_doc'])
speaking_time = pd.merge(pat,doc, on = 'session')

del speaking_time_pat, speaking_time_doc

#%% Granger Causality stuff %%SONO QUI

cols = ["F0final_sma","pcm_loudness_sma"]
#Test for stationarity
stationarityTestPat = {}
stationarityTestDoc = {}
for col in cols:
    stationarityTestPat[col] = {}
    stationarityTestDoc[col] = {}
    for x in allPatData:
        stationarityTestPat[col][x] = stattools.adfuller(allPatDataNoNans[x][col])
        stationarityTestDoc[col][x] = stattools.adfuller(allDocDataNoNans[x][col])
        print(x)

#decompose series to check for seasonality
seasonalitytestPat = {}
seasonalitytestDoc = {}
for col in cols:
    stationarityTestPat[col] = {}
    stationarityTestDoc[col] = {}
    for x in allPatData:
        stationarityTestPat[col][x] = seasonal_decompose(allPatDataNoNans[x][col], model='additive', freq = 1)
        stationarityTestDoc[col][x] = seasonal_decompose(allDocDataNoNans[x][col], model='additive', freq = 1)
        print(x)
#%%create dataframes with all the sessions for descriptive stats. This helps with plotting and modelling
patF0meanIQRstd = Functions.make_df(patDescriptiveStats, 'F0final_sma', 'mean','IQR','std').transpose()
docF0meanIQRstd = Functions.make_df(docDescriptiveStats, 'F0final_sma', 'mean','IQR','std').transpose()
patLoudmeanIQRstd = Functions.make_df(patDescriptiveStats, 'pcm_loudness_sma', 'mean','IQR','std').transpose()
docLoudmeanIQRstd = Functions.make_df(docDescriptiveStats, 'pcm_loudness_sma', 'mean','IQR','std').transpose()
patMelmeanIQRstd = Functions.make_df(patDescriptiveStats, 'melScale', 'mean','IQR','std').transpose()
docMelmeanIQRstd = Functions.make_df(docDescriptiveStats, 'melScale', 'mean','IQR','std').transpose()
patSCmeanIQRstd = Functions.make_df(physioDescriptiveStats, 'SC_patient', 'mean','IQR','std').transpose()
patHFmeanIQRstd = Functions.make_df(physioDescriptiveStats, 'HF_patient', 'mean','IQR','std').transpose()
patPPGmeanIQRstd = Functions.make_df(physioDescriptiveStats, 'PPG_patient', 'mean','IQR','std').transpose()
docSCmeanIQRstd = Functions.make_df(physioDescriptiveStats, 'SC_clinician', 'mean','IQR','std').transpose()
docHFmeanIQRstd = Functions.make_df(physioDescriptiveStats, 'HF_clinician', 'mean','IQR','std').transpose()
docPPGmeanIQRstd = Functions.make_df(physioDescriptiveStats, 'PPG_clinician', 'mean','IQR','std').transpose()
patF0meanIQRstd['sessionName'] = session_list
docF0meanIQRstd['sessionName'] = session_list
patLoudmeanIQRstd['sessionName'] = session_list
docLoudmeanIQRstd['sessionName'] = session_list 
patMelmeanIQRstd['sessionName'] = session_list 
docMelmeanIQRstd['sessionName'] = session_list 

#%%PLOTS START 
empathyLabels = validEmpathySessions; empathyLabels.index = empathyLabels['sessionName']; empathyLabels = empathyLabels.transpose()
anxietyLabels = validAnxietySessions; anxietyLabels.index = anxietyLabels['sessionName']; anxietyLabels = anxietyLabels.transpose()
#%% Plot windowed data for every session
pat = []
doc = []
for x, y in zip(allWindowedPat,allWindowedDoc):
    pat.append(allWindowedPat[x]['median_F0'])
    doc.append(allWindowedDoc[y]['median_F0'])
    
fig, axes = plt.subplots(nrows=5, ncols=3, figsize = (20,20), sharex=True)
fig.suptitle('Windowed F0 values for patient and doctor', verticalalignment = 'bottom')
# axes.flat returns the set of axes as a flat (1D) array instead
# of the two-dimensional version we used earlier
for ax, title, x, y in zip(axes.flat, session_list, pat, doc):
    ax.plot(x, 'c', label = 'Patient')
    ax.plot(y, 'm', label = 'Doctor')
    ax.set_title(title + '\n Empathy pat:%.2f' %empathyLabels[title]['normPat'] + '/Empathy doc:%.2f' %empathyLabels[title]['normDoc'] 
    + '/Anxiety:%.2f' %anxietyLabels[title]['normScoring'])
    ax.set_ylabel('F0 in Hz')
    ax.legend(loc='upper right')
    ax.grid(True)
    plt.tight_layout()
#plt.savefig('PatDocWindowed.png')

#%% PLot MelScale every session
patMel = []
docMel = []
for x, y in zip(windowedPatMel,windowedDocMel):
    patMel.append(windowedPatMel[x]['melScale'])
    docMel.append(windowedDocMel[y]['melScale'])
    
fig, axes = plt.subplots(nrows=5, ncols=3, figsize = (20,20), sharex=True)
fig.suptitle('Windowed F0 Mel scale values for patient and doctor', verticalalignment = 'bottom')
# axes.flat returns the set of axes as a flat (1D) array instead
# of the two-dimensional version we used earlier
for ax, title, x, y in zip(axes.flat, session_list, patMel, docMel):
    ax.plot(x, 'c', label = 'Patient')
    ax.plot(y, 'm', label = 'Doctor')
    ax.set_title(title + '\n Empathy pat:%.2f' %empathyLabels[title]['normPat'] + '/Empathy doc:%.2f' %empathyLabels[title]['normDoc'] 
    + '/Anxiety:%.2f' %anxietyLabels[title]['normScoring'])
    ax.set_ylabel('Mel scale F0')
    ax.legend(loc='upper right')
    ax.grid(True)
    plt.tight_layout()

#%% PLot loudness every session
patLoud = []
docLoud = []
for x in windowedPatLoud:
    patLoud.append(windowedPatLoud[x]['pcm_loudness_sma'])
    docLoud.append(windowedDocLoud[x]['pcm_loudness_sma'])
    
fig, axes = plt.subplots(nrows=5, ncols=3, figsize = (20,20), sharex=True)
fig.suptitle('Windowed loudness values for patient and doctor', verticalalignment = 'bottom')
# axes.flat returns the set of axes as a flat (1D) array instead
# of the two-dimensional version we used earlier
for ax, title, x, y in zip(axes.flat, session_list, patLoud, docLoud):
    ax.plot(x, 'c', label = 'Patient')
    ax.plot(y, 'm', label = 'Doctor')
    ax.set_title(title)# + '\n Empathy pat:%.2f' %empathyLabels[title]['normPat'] + '/Empathy doc:%.2f' %empathyLabels[title]['normDoc'] 
    #+ '/Anxiety:%.2f' %anxietyLabels[title]['normScoring'])
    ax.set_ylabel('Mel scale F0')
    ax.legend(loc='upper right')
    ax.grid(True)
    plt.tight_layout()
#%% plot physio stuff
physioStuff = ['PPG','SC','HF']
for item in physioStuff:
    pat = []
    doc = []
    for x in session_list:
        pat.append(windowedPhysio[x][item+'_{0}'.format('patient')])
        doc.append(windowedPhysio[x][item+'_{0}'.format('clinician')])

    fig, axes = plt.subplots(nrows=5, ncols=3, figsize = (20,20), sharex=True)
    fig.suptitle('Windowed {0} values for patient and doctor'.format(item), verticalalignment = 'bottom')
    for ax, p, d, title in zip(axes.flat, pat, doc, session_list):
        ax.plot(p[item+'_{0}'.format('patient')], 'c', label = 'Patient')
        ax.plot(d[item+'_{0}'.format('clinician')], 'm', label = 'Doctor')
        ax.set_title(title)# + '\n Empathy pat:%.2f' %empathyLabels[title]['normPat'] + '/Empathy doc:%.2f' %empathyLabels[title]['normDoc'] 
        #+ '/Anxiety:%.2f' %anxietyLabels[title]['normScoring'])
        ax.set_ylabel(item)
        ax.legend(loc='upper right')
        ax.grid(True)
        plt.tight_layout()
#%% Plot descriptive stats with anxiety and empathy 
#### WORK IN PROGRESS#####
plt.plot(patF0meanIQRstd['sessionName'],Functions.normalise(patF0meanIQRstd['std']), label = 'patf0') #can be done for 'std' 'IQR' and 'mean'
plt.plot(docF0meanIQRstd['sessionName'],Functions.normalise(docF0meanIQRstd['std']),label = 'docf0')
plt.plot(patLoudmeanIQRstd['sessionName'],Functions.normalise(patLoudmeanIQRstd['std']),label = 'patloud')
plt.plot(docLoudmeanIQRstd['sessionName'],Functions.normalise(docLoudmeanIQRstd['std']),label = 'docloud')
plt.plot(validEmpathySessions['sessions'],validEmpathySessions['normPat'], 'bo', label = 'EmpathyPat')
plt.plot(validEmpathySessions['sessions'],validEmpathySessions['normDoc'], 'ro', label = 'EmpathyDoc')
plt.legend(loc='upper right')
plt.title('Standard deviation of F0 and loudness with empathy scores')

#%% plot descriptive stats of audio and physio
plt.plot(patF0meanIQRstd['sessionName'],Functions.normalise(patF0meanIQRstd['std']), label = 'patf0') #can be done for 'std' 'IQR' and 'mean'
plt.plot(docF0meanIQRstd['sessionName'],Functions.normalise(docF0meanIQRstd['std']),label = 'docf0')
plt.plot(patLoudmeanIQRstd['sessionName'],Functions.normalise(patLoudmeanIQRstd['std']),label = 'patloud')
plt.plot(docLoudmeanIQRstd['sessionName'],Functions.normalise(docLoudmeanIQRstd['std']),label = 'docloud')
#%% Plot vowel ratio with empathy and anxiety #not functioning properly yet
plt.figure()
plt.plot(patVowelRatios['ratio'],'co', label = 'patient vowel ratio')
plt.plot(validEmpathySessions['normPat'],'bo',label = 'patient empaty score')
plt.plot(docVowelRatios['ratio'],'o', color = 'orange',label = 'doctor vowel ratio')
plt.plot(validEmpathySessions['normDoc'],'ro',label = 'doctor empathy score')
plt.xticks(validEmpathySessions['seduta'], validEmpathySessions['sessionName'])
plt.title('Perceived empathy and vowel space ratio')
plt.legend(loc='lower right')

plt.figure()
plt.plot(patVowelRatios['ratio'],'co', label = 'patient vowel ratio')
plt.plot(validAnxietySessions['normScoring'],'bo',label = 'patient anxiety score')
plt.plot(docVowelRatios['ratio'],'o', color = 'orange',label = 'doctor vowel ratio')
plt.xticks(patVowelRatios['index'], patVowelRatios['sessionName'])
plt.title('Anxiety score and vowel space ratio')
plt.legend(loc='lower right')

#%% Box and whiskers plot of correlations distribution
correlation_coefficients = Functions.make_df(correlations,'correlation_coeff', ':')

fig = plt.figure(1, figsize=(9, 6))
bp = correlation_coefficients.boxplot(grid=False)
plt.title('Boxplot of correlation coefficients')
plt.plot(validAnxietySessions['sessionName'], validAnxietySessions['normScoring'], 'ro', label = 'Anxiety')
plt.plot(validEmpathySessions['sessionName'],validEmpathySessions['stdPat'], 'mo', label = 'Patient')
plt.plot(validEmpathySessions['sessionName'],validEmpathySessions['stdDoc'], 'co', label = 'Doctor')
#fig.savefig(outFolder+'CorrelationCoefficients')

#%% Box and whiskers plot of lags distribution
lags_distrib = Functions.make_df(crossCorrelations,'lag', ':')

fig = plt.figure(2, figsize=(9, 6))
bp = lags_distrib.boxplot(grid=False)
plt.title('Boxplot of lags')
#fig.savefig(outFolder+'CorrelationCoefficients')

#%% Box and whiskers plot of SHUFFLED correlations distribution
shuffled_coefficients = Functions.make_df(shuffledCorrs,'correlation_coeff', ':')

fig = plt.figure(1, figsize=(9, 6))
bp = shuffled_coefficients.boxplot(grid=False)
plt.title('Boxplot of shuffled correlation coefficients')
#fig.savefig(outFolder+'CorrelationCoefficients')

#%% Box and whiskers plot of SLOPE correlations distribution
correlation_coefficientsSlope = Functions.make_df(correlationsSlope,'correlation_coeff', ':')

fig = plt.figure(1, figsize=(9, 6))
bp = correlation_coefficientsSlope.boxplot(grid=False)
plt.title('Boxplot of Slope correlation coefficients')
plt.plot(validAnxietySessions['sessionName'], validAnxietySessions['normScoring'], 'ro', label = 'Anxiety')
plt.plot(validEmpathySessions['sessionName'],validEmpathySessions['stdPat'], 'mo', label = 'Patient')
plt.plot(validEmpathySessions['sessionName'],validEmpathySessions['stdDoc'], 'co', label = 'Doctor')

#%% Box and whiskers plot of Loudness correlations distribution
correlation_coefficientsLoud = Functions.make_df(correlationsLoud,'correlation_coeff', ':')

fig = plt.figure(1, figsize=(9, 6))
bp = correlation_coefficientsLoud.boxplot(grid=False)
plt.title('Boxplot of correlation coefficients')
plt.plot(validAnxietySessions['sessionName'], validAnxietySessions['normScoring'], 'ro', label = 'Anxiety')
plt.plot(validEmpathySessions['sessionName'],validEmpathySessions['stdPat'], 'mo', label = 'Patient')
plt.plot(validEmpathySessions['sessionName'],validEmpathySessions['stdDoc'], 'co', label = 'Doctor')
#%% plot correlations as time series
corrs = []
for x in correlations:
    corrs.append(correlations[x]['correlation_coeff'])
    
fig, axes = plt.subplots(nrows=5, ncols=3, figsize = (20,20), sharex=True)
fig.suptitle('Correlation coefficients as time series')
# axes.flat returns the set of axes as a flat (1D) array instead
# of the two-dimensional version we used earlier
for ax, title, x in zip(axes.flat, session_list, corrs):
    ax.plot(x)
    ax.set_title(title + '\n Empathy pat:%.2f' %empathyLabels[title]['normPat'] + '/Empathy doc:%.2f' %empathyLabels[title]['normDoc'] 
    + '/Anxiety:%.2f' %anxietyLabels[title]['normScoring'])
    ax.set_ylabel('Corr coefficient')
    ax.grid(True)
    plt.tight_layout()
#%% plot correlations SLOPE as time series
corrsSlope = []
for x in correlationsSlope:
    corrsSlope.append(correlationsSlope[x]['correlation_coeff'])
    
fig, axes = plt.subplots(nrows=5, ncols=3, figsize = (20,20), sharex=True)
fig.suptitle('Correlation coefficients of slope as time series')
# axes.flat returns the set of axes as a flat (1D) array instead
# of the two-dimensional version we used earlier
for ax, title, x in zip(axes.flat, session_list, corrsSlope):
    ax.plot(x)
    ax.set_title(title + '\n Empathy pat:%.2f' %empathyLabels[title]['normPat'] + '/Empathy doc:%.2f' %empathyLabels[title]['normDoc'] 
    + '/Anxiety:%.2f' %anxietyLabels[title]['normScoring'])
    ax.set_ylabel('Corr coefficient')
    ax.grid(True)
    plt.tight_layout()    

#%% plot correlations Loudness as time series
corrsLoud = []
for x in correlationsLoud:
    corrsLoud.append(correlationsLoud[x]['correlation_coeff'])
    
fig, axes = plt.subplots(nrows=5, ncols=3, figsize = (20,20), sharex=True)
fig.suptitle('Correlation coefficients of loudness as time series')
# axes.flat returns the set of axes as a flat (1D) array instead
# of the two-dimensional version we used earlier
for ax, title, x in zip(axes.flat, session_list, corrsLoud):
    ax.plot(x)
    ax.set_title(title + '\n Empathy pat:%.2f' %empathyLabels[title]['normPat'] + '/Empathy doc:%.2f' %empathyLabels[title]['normDoc'] 
    + '/Anxiety:%.2f' %anxietyLabels[title]['normScoring'])
    ax.set_ylabel('Corr coefficient')
    ax.grid(True)
    plt.tight_layout() 
#%% plot correlation coefficients with lags
lags = []
for x in crossCorrelations:
    lags.append(crossCorrelations[x]['lag'])

fig, axes = plt.subplots(nrows=5, ncols=3, figsize = (20,20), sharex=True)
fig.suptitle('Correlation coefficients and lags (negative = doctor leading)')
    
for ax, title, x, y in zip(axes.flat, session_list, corrs, lags):
    ax.plot(x)
    ax.set_title(title + '\n Empathy pat:%.2f' %empathyLabels[title]['normPat'] + '/Empathy doc:%.2f' %empathyLabels[title]['normDoc'] 
    + '/Anxiety:%.2f' %anxietyLabels[title]['normScoring'])
    ax.set_ylabel(r"corr coeff", color="c")
    ax2 = ax.twinx()
    ax2.plot(y, color="m")
    ax2.set_ylabel(r"lag", color="m")
    ax.grid(True)
    plt.tight_layout()
    
#%% shuffled correlations as time series
shuffled_corrs = []
for x in shuffledCorrs:
    shuffled_corrs.append(shuffledCorrs[x]['correlation_coeff'])
    
fig, axes = plt.subplots(nrows=4, ncols=2, figsize = (20,20), sharex=True)
fig.suptitle('Shuffled correlation coefficients as time series')
# axes.flat returns the set of axes as a flat (1D) array instead
# of the two-dimensional version we used earlier
for ax, title, x in zip(axes.flat, session_list, shuffled_corrs):
    ax.plot(x)
    ax.set_title(title)
    ax.set_ylabel('Corr coefficient')
    ax.grid(True)
    plt.tight_layout()
    
#%% Plot correlations audiophysio 
physioStuff = ['PPG','SC','HF']
for item in physioStuff:
    patCorr = []
    docCorr = []
    for x in session_list:
        patCorr.append(physioAudioCorrsPat[item][x]['correlation_coeff'])
        docCorr.append(physioAudioCorrsDoc[item][x]['correlation_coeff'])

    fig1, axes = plt.subplots(nrows=4, ncols=2, figsize = (20,20), sharex=True)
    fig1.suptitle('Correlation between {0} and audio for patient\'s data'.format(item))
    for ax, p, title in zip(axes.flat, patCorr, session_list):
        ax.plot(p, 'b')
        ax.set_title(title)# + '\n Empathy pat:%.2f' %empathyLabels[title]['normPat'] + '/Empathy doc:%.2f' %empathyLabels[title]['normDoc'] 
        #+ '/Anxiety:%.2f' %anxietyLabels[title]['normScoring'])
        ax.set_ylabel('correlation coefficient')
        ax.grid(True)
        plt.tight_layout()
    fig2, axes = plt.subplots(nrows=4, ncols=2, figsize = (20,20), sharex=True)
    fig2.suptitle('Correlation between {0} and audio for doctor\'s data'.format(item))
    for ax,d, title in zip(axes.flat, docCorr, session_list):
        ax.plot(d, 'r')
        ax.set_title(title)# + '\n Empathy pat:%.2f' %empathyLabels[title]['normPat'] + '/Empathy doc:%.2f' %empathyLabels[title]['normDoc'] 
        #+ '/Anxiety:%.2f' %anxietyLabels[title]['normScoring'])
        ax.set_ylabel('correlation coefficient')
        ax.grid(True)
        plt.tight_layout()   
#%% Plot correlations coeff percentages
xdata = list(percentage_of_correlationsDF.columns.values)[1:]
ydata = percentage_of_correlationsDF.dropna().drop('session', axis=1)
xpos = np.arange(len(xdata))
labels = percentage_of_correlationsDF['session'].dropna()
fig1, ax = plt.subplots()
index = np.arange(len(xdata))
bar_width = 0.1

rects1 = plt.scatter(index - bar_width*4, ydata.iloc[0], 10, label = labels.iloc[0])
rects2 = plt.scatter(index - bar_width*3, ydata.iloc[1], 10, label = labels.iloc[1])
rects3 = plt.scatter(index - bar_width*2, ydata.iloc[2], 10, label = labels.iloc[2])
rects4 = plt.scatter(index - bar_width, ydata.iloc[3], 10, label = labels.iloc[3])
rects5 = plt.scatter(index + bar_width, ydata.iloc[4], 10, label = labels.iloc[4])
rects6 = plt.scatter(index + bar_width*2, ydata.iloc[5], 10, label = labels.iloc[5])
rects7 = plt.scatter(index + bar_width*3, ydata.iloc[6], 10, label = labels.iloc[6])
rects8 = plt.scatter(index + bar_width*4, ydata.iloc[7], 10, label = labels.iloc[7])
 
plt.xlabel('Correlation Thresholds')
plt.ylabel('Percentage')
plt.title('Percentage of correlations within certain thresholds')
plt.xticks(index, xdata)
ax.set_xticks(index + 0.5,minor=True)
plt.legend(loc='upper right')
ax.xaxis.grid(which='minor')
ax.yaxis.grid(linestyle = ':')
 
plt.tight_layout()
plt.show()

#%% Outputs

Functions.outFunction(patDescriptiveStats,docDescriptiveStats,'DescriptiveStatsRaw', outFolder)
Functions.outFunction(patDescriptiveStatsSTD,docDescriptiveStatsSTD,'DescriptiveStatsRawSTD', outFolder)
