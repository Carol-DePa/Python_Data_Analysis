#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:10:45 2019

@author: carolinadepasquale
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import random
from math import ceil, isnan
import os
import python_speech_features
import scipy.stats as stats
#%% Script import
dir_path = os.path.dirname(os.path.realpath('/Volumes/GoogleDrive/Il mio Drive/Ph.D./PsychologicalSyncStudy/AnalysisScripts/PythonScripts/CleanMain.py'))
os.chdir(dir_path)
import Functions
#%%
#Data import
doctorData = pd.read_csv("/Users/carolinadepasquale/Desktop/TestDance/DanceSteps/prosody1.csv", sep = ';')
patientData = pd.read_csv("/Users/carolinadepasquale/Desktop/TestDance/DanceSteps/prosody2.csv", sep = ';') 
cols = ["F0final_sma","voicingFinalUnclipped_sma","pcm_loudness_sma"]
#patientData[cols] = patientData[cols].replace({0:np.nan})
#doctorData[cols] = doctorData[cols].replace({0:np.nan})
#%%
sns.distplot(patientData.F0final_sma.dropna())
sns.distplot(doctorData.F0final_sma.dropna())

sns.distplot(patientData.pcm_loudness_sma.dropna())
sns.distplot(doctorData.pcm_loudness_sma.dropna())

#%% test clean
loudnessCheckPat = patientData.pcm_loudness_sma < 0.2
F0CheckPat = patientData.F0final_sma < 10
loudnessCheckDoc = doctorData.pcm_loudness_sma < 0.2
F0CheckDoc = doctorData.F0final_sma < 10

for x in cols:
    patientData[x][loudnessCheckPat] = np.nan
    patientData[x][F0CheckPat] = np.nan
    doctorData[x][loudnessCheckDoc] = np.nan
    doctorData[x][F0CheckDoc] = np.nan
#%%
F0topOutliersPat = patientData.F0final_sma > patientData.F0final_sma.quantile(.75)
F0bottomOutliersPat  = patientData.F0final_sma < patientData.F0final_sma.quantile(.15)
F0topOutliersDoc = doctorData.F0final_sma > doctorData.F0final_sma.quantile(.75)
F0bottomOutliersDoc  = doctorData.F0final_sma < doctorData.F0final_sma.quantile(.15)

for x in cols:   
    patientData[x][F0topOutliersPat] = np.nan
    doctorData[x][F0topOutliersDoc] = np.nan
for x in cols:  
    patientData[x][F0bottomOutliersPat] = np.nan
    doctorData[x][F0bottomOutliersDoc] = np.nan

#%%
#Add melscale
patientData['melScale'] = python_speech_features.base.hz2mel(patientData["F0final_sma"])
doctorData['melScale'] = python_speech_features.base.hz2mel(doctorData["F0final_sma"])
cols = ["F0final_sma","pcm_loudness_sma", 'melScale']

#%%
data1 = patientData
data2 = doctorData
time_point = 5
time_points = [5,15,25,35,45]
minute = 6000 # one minute
window_size = 1*minute
window_start_time = time_point*minute
actual_start = data1[window_start_time:]['F0final_sma'].first_valid_index()
window_end_time = actual_start + window_size

def check_windows(data1,data2,actual_start,window_end_time, window_size):
    if data2[actual_start:]['F0final_sma'].first_valid_index() < window_end_time:
        window1 = data1[actual_start:window_end_time]
        window2 = data2[actual_start:window_end_time]
    else:
        actual_start = data2[window_start_time:].first_valid_index()
        window_end_time = actual_start + window_size
        window1 = data1[actual_start:window_end_time]
        window2 = data2[actual_start:window_end_time]
    return window1,window2

def get_summary_windows(data1,data2,time_points,window_duration):
    minute = 6000
    window_size = window_duration*minute
    data1_summary = {}
    data2_summary = {}
    for x in time_points:
        window_start_time = x*minute
        actual_start = data1[window_start_time:]['F0final_sma'].first_valid_index()
        window_end_time = actual_start + window_size
        window1,window2 = check_windows(data1,data2,actual_start,window_end_time, window_size)
        data1_summary[x] = Functions.describe_df(window1, cols)
        data2_summary[x] = Functions.describe_df(window2, cols)
    return data1_summary, data2_summary

def get_raw_windows(data1,data2,time_points,window_duration):
    minute = 6000
    window_size = window_duration*minute
    data1_raw = {}
    data2_raw = {}
    for x in time_points:
        window_start_time = x*minute
        actual_start = data1[window_start_time:]['F0final_sma'].first_valid_index()
        window_end_time = actual_start + window_size
        data1_raw[x],data2_raw[x] = check_windows(data1,data2,actual_start,window_end_time, window_size)
        #data1_raw[x] = window1
        #data2_raw[x] = window2
    return data1_raw, data2_raw

patientSummary,doctorSummary = get_summary_windows(data1,data2,time_points,1)
patientWindows,doctorWindows = get_raw_windows(data1,data2,time_points,1)

#%%
doctorSummaryDF = {}; patientSummaryDF = {}

for x in cols:
    doctorSummaryDF[x] = Functions.make_df(doctorSummary, x, 'mean','IQR','std').transpose() 
    patientSummaryDF[x] = Functions.make_df(patientSummary, x, 'mean','IQR','std').transpose()
#%% smooth raw windows, by stattyoe and feature (cols)
smoothData = {'median': np.nanmedian,'std': np.nanstd}
frame_size = 5
step = 2
patientWindowsSmooth = {}
doctorWindowsSmooth = {}
for y in patientWindows:
    max_pat = max(patientWindows[y]['frameTime'])
    max_doc = max(doctorWindows[y]['frameTime'])
    max_time = max(max_pat,max_doc)
    patientWindowsSmooth[y] = {}
    doctorWindowsSmooth[y] = {}
    for x in smoothData:
        patientWindowsSmooth[y][x] = {}
        doctorWindowsSmooth[y][x] = {}
        for col in cols:
            patientWindowsSmooth[y][x][col] = Functions.windowsWithStepsGeneral(patientWindows[y], 'frameTime', col, frame_size, step, max_time, stattype= smoothData[x])
            doctorWindowsSmooth[y][x][col] = Functions.windowsWithStepsGeneral(doctorWindows[y], 'frameTime', col, frame_size, step, max_time, stattype= smoothData[x])
            
#%%
statsTypes = ['mean','IQR','std']
for stattype in statsTypes:
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize = (10,10))
    fig.suptitle(stattype)   
    for ax, x in zip(axes.flat, cols):
        ax.plot(doctorSummaryDF[x].index.values, doctorSummaryDF[x][stattype], 'go', label = 'Doctor')
        ax.plot(patientSummaryDF[x].index.values, patientSummaryDF[x][stattype], 'mo', label = 'Patient')
        ax.set_title(x)
        #ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)    
        ax.grid(True)
    plt.savefig('/Volumes/GoogleDrive/Il mio Drive/Ph.D./TestDance/OutputGraphs/Summary_of{0}s.png'.format(stattype))
        #plt.tight_layout()      
#%%
#plot the selected minutes as time series
for feature in cols:
    for smooth in smoothData:
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize = (20,20))
        fig.suptitle("Time series smooth data of {0}' {1}".format(feature,smooth))   
        for ax, key in zip(axes.flat, patientWindows):
            ax.plot(patientWindowsSmooth[key][smooth][feature]['window_start'], patientWindowsSmooth[key][smooth][feature][feature], 'c', label = 'Patient')
            ax.plot(doctorWindowsSmooth[key][smooth][feature]['window_start'], doctorWindowsSmooth[key][smooth][feature][feature], 'm', label = 'Doctor')
            #ax.plot(y, 'm', label = 'Doctor')
            ax.set_title(key)
            plt.tight_layout()
    fig.legend(loc='lower right')
    
#%% plot smooth time series
for feature in cols:
    for smooth in smoothData:
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize = (20,20))
        fig.suptitle("Time series smooth data of {0}' {1}".format(feature,smooth))   
        for ax, key in zip(axes.flat, patientWindows):
            ax.plot(patientWindowsSmooth[key][smooth][feature]['window_start'], patientWindowsSmooth[key][smooth][feature][feature], 'c', label = 'Patient')
            ax.plot(doctorWindowsSmooth[key][smooth][feature]['window_start'], doctorWindowsSmooth[key][smooth][feature][feature], 'm', label = 'Doctor')
            #ax.plot(y, 'm', label = 'Doctor')
            ax.set_title(key)
            plt.tight_layout()
            fig.legend(loc='lower right')
        plt.savefig("/Volumes/GoogleDrive/Il mio Drive/Ph.D./TestDance/OutputGraphs/Smooth_ts{1}_of_{0}.png".format(feature,smooth))
#%%
fig, axes = plt.subplots(nrows=3, ncols=1, figsize = (10,10), sharex=True)
#fig.suptitle(stattype)   
for ax, x in zip(axes.flat, cols):
    for key, value in patientWindows.items():
        #sns.boxplot(data=patientWindows[time][x], ax = ax)
        ax.boxplot(value[x])
        #ax.set_xticklabels(key)
    #sns.swarmplot(data=correlation_coefficients[x], size = 2, color = 'white', ax = ax)
        ax.set_title(x)
    #ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)    
        ax.grid(True)
        
        
for key, value in patientWindows.items():
    for col in value:
        if col in cols:
            print(patientWindows[key][col])