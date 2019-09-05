#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 14:35:31 2019

@author: carolinadepasquale
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
import statsmodels.tsa.stattools 
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
    print(x) 
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
    print(x)    

#%% Substitute all 0s with nans only for F0 and remove outliers
cols = ["F0final_sma","voicingFinalUnclipped_sma","pcm_loudness_sma"]

F0Check = 10
F0topQuantile = .95
F0bottomQuantile = .15
topLoud = .95
bottomLoud = .05
allPatData = Functions.removeOutliers(allPatDataRaw, cols, F0Check, F0topQuantile, F0bottomQuantile, topLoud, bottomLoud)
allDocData = Functions.removeOutliers(allDocDataRaw, cols, F0Check, F0topQuantile, F0bottomQuantile, topLoud, bottomLoud)

#%% Add mel scale to DFs
for x,y in zip(allPatData, allDocData):
    allPatData[x]['melScale'] = python_speech_features.base.hz2mel(allPatData[x]["F0final_sma"])
    allDocData[y]['melScale'] = python_speech_features.base.hz2mel(allDocData[y]["F0final_sma"]) 
    print(x)
#%% Remove Nans, you need this for the slope anyway
allPatDataNoNans = {}
allDocDataNoNans = {}
for x,y in zip(allPatData, allDocData):
    allPatDataNoNans[x] = copy.deepcopy(allPatData[x]).replace({np.nan:0})
    allDocDataNoNans[y] = copy.deepcopy(allDocData[y]).replace({np.nan:0})
    print(x)
#%% Data windowing
smoothData = {'median': np.nanmedian,'std': np.nanstd}
frame_size = 10
step = 5
patientSmooth = {}
doctorSmooth = {}
cols = ["F0final_sma","pcm_loudness_sma", 'melScale']

for col in cols:
    patientSmooth[col] = {}
    doctorSmooth[col] = {}
    for x in smoothData:
        patientSmooth[col][x] = {}
        doctorSmooth[col][x] = {}
        for y in allPatData:
            max_pat = max(allPatData[y]['frameTime'])
            max_doc = max(allDocData[y]['frameTime'])
            max_time = max(max_pat,max_doc)
            patientSmooth[col][x][y] = Functions.windowsWithStepsGeneral(allPatData[y], 'frameTime', col, frame_size, step, max_time, stattype= smoothData[x])
            doctorSmooth[col][x][y] = Functions.windowsWithStepsGeneral(allDocData[y], 'frameTime', col, frame_size, step, max_time, stattype= smoothData[x])
            print(col,x,y)  
#%% windowed slope
for col in cols:
    patientSmooth[col]['slope'] = {}
    doctorSmooth[col]['slope'] = {}
    for x in allPatDataNoNans:
        max_pat = max(allPatDataNoNans[x]['frameTime'])
        max_doc = max(allDocDataNoNans[x]['frameTime'])
        max_time = max(max_pat,max_doc)
        patientSmooth[col]['slope'][x] = Functions.windowsWithStepsSlope(allPatDataNoNans[x], 'frameTime', col, frame_size, step, max_time)
        doctorSmooth[col]['slope'][x] = Functions.windowsWithStepsSlope(allDocDataNoNans[x], 'frameTime', col, frame_size, step, max_time)
        print(col,x)
#%% Correlations iterative
frame_size = 60
step = 5
correlations = {}
for feature in patientSmooth:
    correlations[feature] = {}
    for smooth in patientSmooth[feature]:
        correlations[feature][smooth] = {}
        for x in patientSmooth[feature][smooth]:
            print(feature,smooth,x)
            max_pat = max(allPatData[x]['frameTime'])
            max_doc = max(allDocData[x]['frameTime'])
            max_time = max(max_pat,max_doc)
            #I need try/except because slope is saved in "slope" as a column, instead of the feature name, so it'd fail
            try:
                correlations[feature][smooth][x] = Functions.correlate_windows_general(patientSmooth[feature][smooth][x], 
                        doctorSmooth[feature][smooth][x], feature, feature, frame_size, step, max_time, .3)
            except:
                correlations[feature][smooth][x] = Functions.correlate_windows_general(patientSmooth[feature][smooth][x], 
                        doctorSmooth[feature][smooth][x], smooth, smooth, frame_size, step, max_time, .3)                
#%% Shuffled corrs
shuffled_correlations = {}

for feature in patientSmooth:
    shuffled_correlations[feature] = {}
    for smooth in patientSmooth[feature]:
        shuffled_correlations[feature][smooth] = {}
        for x in patientSmooth[feature][smooth]:
            print(feature,smooth,x)
            max_pat = max(allPatData[x]['frameTime'])
            max_doc = max(allDocData[x]['frameTime'])
            max_time = max(max_pat,max_doc)
            y = random.choice(list(allDocData))
            #I need try/except because slope is saved in "slope" as a column, instead of the feature name, so it'd fail
            try:
                shuffled_correlations[feature][smooth][x] = Functions.correlate_windows_general(patientSmooth[feature][smooth][x], 
                        doctorSmooth[feature][smooth][y], feature, feature, frame_size, step, max_time, .3)
            except:
                shuffled_correlations[feature][smooth][x] = Functions.correlate_windows_general(patientSmooth[feature][smooth][x], 
                        doctorSmooth[feature][smooth][y], smooth, smooth, frame_size, step, max_time, .3)  

#%% Find significant coeffs: p<.05 #THIS IS OPTIMISED
significantCorrs = {}
for feature in correlations:
    significantCorrs[feature] = {}
    for smooth in correlations[feature]:
        significantCorrs[feature][smooth] = {}
        for x in correlations[feature][smooth]:
            significantCorrs[feature][smooth][x] = correlations[feature][smooth][x][correlations[feature][smooth][x]['p_value'] < .05]
            print(feature,smooth,x)
#%%
shuffled_significantCorrs = {}
for feature in shuffled_correlations:
    shuffled_significantCorrs[feature] = {}
    for smooth in shuffled_correlations[feature]:
        shuffled_significantCorrs[feature][smooth] = {}
        for x in shuffled_correlations[feature][smooth]:
            shuffled_significantCorrs[feature][smooth][x] = shuffled_correlations[feature][smooth][x][shuffled_correlations[feature][smooth][x]['p_value'] < .05]
            print(feature,smooth,x)
#%% get proportions of significant correlations over total correlations, and positive significant over total significant
proportions_sig_corrs = {'general' : {}, 'positive': {}, 'negative' : {}, 'ratio_score' : {}}

for kind in proportions_sig_corrs:
    for feature in significantCorrs:
        proportions_sig_corrs[kind][feature] = {}
        for smooth in significantCorrs[feature]:
            proportions_sig_corrs[kind][feature][smooth] = {}
            for x in significantCorrs[feature][smooth]:
                print(kind,feature,smooth,x)
                if kind == 'general':
                    proportions_sig_corrs[kind][feature][smooth][x] = len(significantCorrs[feature][smooth][x])/len(correlations[feature][smooth][x])
                elif kind == 'positive':
                    proportions_sig_corrs[kind][feature][smooth][x] = sum(significantCorrs[feature][smooth][x]['correlation_coeff']>0)/len(correlations[feature][smooth][x])
                elif kind == 'negative':
                    proportions_sig_corrs[kind][feature][smooth][x] = sum(significantCorrs[feature][smooth][x]['correlation_coeff']<0)/len(correlations[feature][smooth][x])
                elif kind == 'ratio_score':
                    proportions_sig_corrs[kind][feature][smooth][x] = sum(significantCorrs[feature][smooth][x]['correlation_coeff']>0)/sum(significantCorrs[feature][smooth][x]['correlation_coeff']<0)
                else:
                    print('Mistake Somewhere')
                        
#%%
proportions_sig_corrs_Shuffled = {'general' : {}, 'positive': {}, 'negative' : {}, 'ratio_score' : {}}

for kind in proportions_sig_corrs_Shuffled:
    for feature in shuffled_significantCorrs:
        proportions_sig_corrs_Shuffled[kind][feature] = {}
        for smooth in shuffled_significantCorrs[feature]:
            proportions_sig_corrs_Shuffled[kind][feature][smooth] = {}
            for x in shuffled_significantCorrs[feature][smooth]:
                print(kind,feature,smooth,x)
                if kind == 'general':
                    proportions_sig_corrs_Shuffled[kind][feature][smooth][x] = len(shuffled_significantCorrs[feature][smooth][x])/len(shuffled_correlations[feature][smooth][x])
                elif kind == 'positive':
                    proportions_sig_corrs_Shuffled[kind][feature][smooth][x] = sum(shuffled_significantCorrs[feature][smooth][x]['correlation_coeff']>0)/len(shuffled_correlations[feature][smooth][x])
                elif kind == 'negative':
                    proportions_sig_corrs_Shuffled[kind][feature][smooth][x] = sum(shuffled_significantCorrs[feature][smooth][x]['correlation_coeff']<0)/len(shuffled_correlations[feature][smooth][x])
                elif kind == 'ratio_score':
                    proportions_sig_corrs_Shuffled[kind][feature][smooth][x] = sum(shuffled_significantCorrs[feature][smooth][x]['correlation_coeff']>0)/sum(shuffled_significantCorrs[feature][smooth][x]['correlation_coeff']<0)
                else:
                    print('Mistake Somewhere')
  
#%% Create dataframe
significant_corrs_prop_df = pd.DataFrame(columns = ['group','feature','windowing','session','coeff_proportion'])
for group in proportions_sig_corrs:
    for feature in proportions_sig_corrs[group]:
        for windowing in proportions_sig_corrs[group][feature]:
            for session in proportions_sig_corrs[group][feature][windowing]:
                significant_corrs_prop_df = significant_corrs_prop_df.append({'group': group, 'feature': feature,'windowing' : windowing, 'session': session, 'coeff_proportion': proportions_sig_corrs[group][feature][windowing][session]}, ignore_index=True)
                print(group,feature,windowing,session)

#%%
shuffled_significant_corrs_prop_df = pd.DataFrame(columns = ['group','feature','windowing','session','coeff_proportion'])
for group in proportions_sig_corrs_Shuffled:
    for feature in proportions_sig_corrs_Shuffled[group]:
        for windowing in proportions_sig_corrs_Shuffled[group][feature]:
            for session in proportions_sig_corrs_Shuffled[group][feature][windowing]:
                shuffled_significant_corrs_prop_df = shuffled_significant_corrs_prop_df.append({'group': group, 'feature': feature,'windowing' : windowing, 'session': session, 'coeff_proportion': proportions_sig_corrs_Shuffled[group][feature][windowing][session]}, ignore_index=True)
                print(group,feature,windowing,session)

#%%
significant_corrs_prop_df_clean = copy.deepcopy(significant_corrs_prop_df).replace([np.inf, -np.inf], -9999999999999)
shuffled_significant_corrs_prop_df_clean = copy.deepcopy(shuffled_significant_corrs_prop_df).replace([np.inf, -np.inf], np.nan)
#%% significant corrs proportions with empathy and anxiety
significant_corrs_empathy = pd.merge(significant_corrs_prop_df_clean, validEmpathySessions,on='session')
significant_corrs_empathy_anxiety = pd.merge(significant_corrs_empathy, validAnxietySessions[['session','scoring', 'normScoring']],on='session')

#%% fake vs real df
fake_real = [significant_corrs_prop_df_clean[significant_corrs_prop_df_clean.group =='general'], shuffled_significant_corrs_prop_df_clean[shuffled_significant_corrs_prop_df_clean.group =='general']]
fake_real_df = pd.concat(fake_real, keys=['fake', 'real'])
fake_real_df['type'] = np.nan
fake_real_df['type'].loc['real'] = 'real'
fake_real_df['type'].loc['fake'] = 'fake'

#%%Corrs Results 
tests = [stats.spearmanr, stats.kendalltau]
targets = {'empathyPat' : validEmpathySessions['pz'], 'empathyTer' : validEmpathySessions['ter'], 'anxiety': validAnxietySessions['scoring']}
#data = {'general' : significant_corrs_prop_df, 'positive': positive_significant_corrs_prop_df, 'negative' : negative_significant_corrs_prop_df}
data = significant_corrs_prop_df_clean
results = {}
for x in data.group.unique():
    results[x] = {}
    for score in targets:
        results[x][score] = {}
        for feature in data.feature.unique():
            results[x][score][feature] = {}
            for windowing in data.windowing.unique():
                results[x][score][feature][windowing] = {}
                for t in tests:
                    print(x,score,feature,windowing,t)
                    results[x][score][feature][windowing][t.__name__] = t(data[(data['group'] == x)&(data['windowing'] == windowing)&(data['feature'] == feature)]['coeff_proportion'],targets[score])

#%% flatten that dict!! THIS DOESN'T SEEM TO WORK
#from https://towardsdatascience.com/how-to-flatten-deeply-nested-json-objects-in-non-recursive-elegant-python-55f96533103d
# =============================================================================
# from itertools import chain, starmap
# 
# def flatten_json_iterative_solution(dictionary):
#     """Flatten a nested json file"""
# 
#     def unpack(parent_key, parent_value):
#         """Unpack one level of nesting in json file"""
#         # Unpack one level only!!!
#         
#         if isinstance(parent_value, dict):
#             for key, value in parent_value.items():
#                 temp1 = parent_key + '_' + key
#                 yield temp1, value
#         elif isinstance(parent_value, list):
#             i = 0 
#             for value in parent_value:
#                 temp2 = parent_key + '_'+str(i) 
#                 i += 1
#                 yield temp2, value
#         else:
#             yield parent_key, parent_value    
# 
#             
#     # Keep iterating until the termination condition is satisfied
#     while True:
#         # Keep unpacking the json file until all values are atomic elements (not dictionary or list)
#         dictionary = dict(chain.from_iterable(starmap(unpack, dictionary.items())))
#         # Terminate condition: not any value in the json file is dictionary or list
#         if not any(isinstance(value, dict) for value in dictionary.values()) and \
#            not any(isinstance(value, list) for value in dictionary.values()):
#             break
# 
#     return dictionary
# 
# df = flatten_json_iterative_solution(results)
# =============================================================================
#%% PLot to investigate
for d in significantCorrs:
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize = (20,20), sharex=True)
    fig.suptitle('%s with p<0.05'%d)
    for ax, x in zip(axes.flat, significantCorrs[d]):
        ax.plot(significantCorrs[d][x]['window_start'], significantCorrs[d][x]['correlation_coeff'],'o', markersize=3)
        ax.set_title(x)
        ax.set_ylabel('Corr coefficient')
        ax.grid(True)

#%%
#%% Box and whiskers plot of correlations distribution
correlation_coefficients = {}
for d in significantCorrs:
    correlation_coefficients[d] = Functions.make_df(significantCorrs[d],'correlation_coeff', ':')
    
fig, axes = plt.subplots(nrows=3, ncols=2, figsize = (20,20), sharex=True)
for ax, x in zip(axes.flat,correlation_coefficients):
    sns.boxplot(data=correlation_coefficients[x], ax = ax)
    sns.swarmplot(data=correlation_coefficients[x], size = 2, color = 'white', ax = ax)
    ax.set_title('%s with p<0.05'%x)
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    # Just to position axes labels:
    plt.sca(ax)
    plt.xticks(rotation=20)
    plt.tight_layout()

#%%
fig, axes = plt.subplots(nrows=3, ncols=2, figsize = (20,20), sharex=True)
fig.suptitle('Significant coefficients by group as time series')
# axes.flat returns the set of axes as a flat (1D) array instead
# of the two-dimensional version we used earlier
for ax, group in zip(axes.flat, significantCorrs):
    for x in significantCorrs[group]:
        ax.plot(significantCorrs[group][x]['window_start'], significantCorrs[group][x]['correlation_coeff'], 'o', markersize = 2, label = x)
        ax.set_title(group)
        ax.set_ylabel('Corr coefficient')
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
        ax.grid(True)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='right')

#%% corr plots
corrScores = {'anxiety': validAnxietySessions['scoring'], 'empathy_ter': validEmpathySessions['ter'], 'empathy_pat' : validEmpathySessions['pz']}
for score in corrScores:
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize = (20,20), sharex=True)
    fig.suptitle('Significant coefficients and %s'%score)   
    for ax, x in zip(axes.flat, significant_corrs_prop_df.group.unique()):
        ax.plot(significant_corrs_prop_df[significant_corrs_prop_df['group'] == x]['coeff_proportion'], corrScores[score], 'o', markersize = 4)
        ax.set_title(x)
        ax.set_ylabel(score)
        ax.set_xlabel('coefficient proportion')
        #ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)    
        ax.grid(True)
        plt.tight_layout()
#handles, labels = ax.get_legend_handles_labels()
#fig.legend(handles, labels, loc='right')

#%% Same but for pos corrs
corrScores = {'anxiety': validAnxietySessions['scoring'], 'empathy_ter': validEmpathySessions['ter'], 'empathy_pat' : validEmpathySessions['pz']}
for score in corrScores:
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize = (20,20), sharex=True)
    fig.suptitle('Positive significant coefficients and %s'%score)   
    for ax, x in zip(axes.flat, positive_significant_corrs_prop_df.group.unique()):
        ax.plot(positive_significant_corrs_prop_df[positive_significant_corrs_prop_df['group'] == x]['coeff_proportion'], corrScores[score], 'o', markersize = 4)
        ax.set_title(x)
        ax.set_ylabel(score)
        ax.set_xlabel('coefficient proportion')
        #ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)    
        ax.grid(True)
        plt.tight_layout()      