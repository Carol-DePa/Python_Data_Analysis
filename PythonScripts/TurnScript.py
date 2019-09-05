#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:28:41 2018

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
from statsmodels.tsa.stattools import acf
from statsmodels.graphics import tsaplots
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
import os
import python_speech_features
import scipy.io
import re
#%% Script import
dir_path = os.path.dirname(os.path.realpath('/Volumes/GoogleDrive/Il mio Drive/Ph.D./PsychologicalSyncStudy/AnalysisScripts/PythonScripts/CleanMain.py'))
os.chdir(dir_path)
import Functions
#%%
# USE CLEAN MAIN FOR IMPORTS of F0, MFCC, Formants
max_session_n = 17
inFolder = '/Volumes/GoogleDrive/Il mio Drive/Ph.D./PsychologicalSyncStudy/PraatPause/Full/'
outFolder = '/Volumes/GoogleDrive/Il mio Drive/Ph.D./PsychologicalSyncStudy/PraatPause/Outs/'
#%% All Turns Changes
turnChanges = {}
for x, y in zip(allPatData, allDocData):
    pat = allPatData[x].F0final_sma
    doc = allDocData[y].F0final_sma
    data = pd.DataFrame({'pat': pat, 'doc': doc})
    turnChanges[x] = Functions.find_turns(data)
    print(x)
    del pat,doc,data
#pat = patientData.F0final_sma
#doc = doctorData.F0final_sma
#dataa = pd.DataFrame({'pat': pat, 'doc': doc})
#%% Find all turns
allPatTurns = {}
allDocTurns = {}
for x in turnChanges:
    allPatTurns[x], allDocTurns[x] = Functions.find_all_turns(allPatData[x],allDocData[x],turnChanges[x]['starts_speaking'], 'F0final_sma')
    print(x)

#%% plot turns SONO QUI
fig, axes = fig, axes = plt.subplots(nrows=5, ncols=3, figsize = (20,20), sharex=False)
axes_list = [item for sublist in axes for item in sublist] 
fig.suptitle('Turn lengths by session', verticalalignment = 'bottom')
for session in allDocTurns:
    ax = axes_list.pop(0)
    ax.plot([len(allDocTurns[session][x]) for x in allDocTurns[session]], linestyle='--', linewidth=.5, marker='o', markersize=1, label = 'Doc')
    ax.plot([len(allPatTurns[session][x]) for x in allPatTurns[session]], linestyle='--', linewidth=.5, marker='o', markersize=1, label = 'Pat')
    ax.set_title(session)
    ax.set_ylabel('Turn length (100Hz)')
    ax.set_xlabel('Turn number')
    plt.tight_layout()

handles, labels = ax.get_legend_handles_labels()
ax = axes_list.pop(0)
ax.text(0.55, 0.6,labels) #getting there
#lines = ax.plot(range(2), range(2),& nbsp;range(2), range(2))
#fig_legend.legend(lines, labels, loc='center', frameon=False)
#ax.plot(handles,labels)
# Put a legend to the right of the current axis
#ax.legend(handles,labels, loc='center left', bbox_to_anchor=(1, 0.5))
#ax.legend(handles,labels,loc='right', bbox_to_anchor=(1.45, .5),fancybox=True, shadow=False,)
for ax in axes_list:
    ax.remove()


#%% get turns mean, std, and lengths
turnPatMean = {}
turnDocMean = {}
turnPatLen = {}
turnDocLen = {}
turnPatSD = {}
turnDocSD = {}
turnPatCount = {}
turnDocCount = {}
for x in allPatTurns:
    turnPatMean[x] = Functions.getTurnMean(allPatTurns[x])
    turnDocMean[x] = Functions.getTurnMean(allDocTurns[x])

    turnPatLen[x] = Functions.getTurnLen(allPatTurns[x])
    turnDocLen[x] = Functions.getTurnLen(allDocTurns[x])
    
    turnPatCount[x] = len(allPatTurns[x])
    turnDocCount[x] = len(allDocTurns[x])

for x in turnPatLen:
    turnPatSD[x] = np.std(turnPatLen[x])
    turnDocSD[x] = np.std(turnDocLen[x])
    
turnPatMean = pd.DataFrame(list(turnPatMean.items()), columns=['session', 'mean']); 
turnPatCount = pd.DataFrame(list(turnPatCount.items()), columns=['session', 'count'])
turnPatSD = pd.DataFrame(list(turnPatSD.items()), columns=['session', 'SD'])

turnDocMean = pd.DataFrame(list(turnDocMean.items()), columns=['session', 'mean']); 
turnDocCount = pd.DataFrame(list(turnDocCount.items()), columns=['session', 'count'])
turnDocSD = pd.DataFrame(list(turnDocSD.items()), columns=['session', 'SD'])
#%% Silence import
combinedSilences = {}
for x in range(1,max_session_n):
    try:
        combinedSilences["session{0}".format(x)] = pd.read_csv(inFolder+'session{0}.txt'.format(x), sep = '\t')
        combinedSilences["session{0}".format(x)].rename(columns={combinedSilences["session{0}".format(x)].columns[1]: "pauseLength" }, inplace = True)
    except:
        x+1
#%% Silent sections
csvFolder = 'SilenceCSVs/'
silentSections = {}
for file in os.listdir(inFolder+csvFolder):
    silentSections[file] = pd.read_csv(inFolder+csvFolder+file, sep = '\t', skiprows = 2)

del silentSections['ALLCSVs.csv']
silentSections = Functions.newDicCopy(silentSections)

#%%
onlySilence = {}
for x in silentSections:
    onlySilence[x] = silentSections[x][silentSections[x]['Label']=='xxx ']

#%%
longSilence = {}    
for x in onlySilence:
    longSilence[x] = silentSections[x][silentSections[x]['Duration']>3]
    
longSilencesDistrib = Functions.make_df(longSilence,'Duration', ':')
#%%
session_list = []
validSessions = []
for x in range(1,max_session_n):
    try:
        combinedSilences["session{0}".format(x)]
        session_list.append('session{0}'.format(x))
        validSessions.append(x)
    except:
        x+1
        
#%% Select only the sessions I have

validEmpathySessions = empathyScores.loc[empathyScores['seduta'].isin(validSessions)]
validAnxietySessions = anxietyScore.loc[anxietyScore['SEDUTA'].isin(validSessions)]

validEmpathySessions.index = validEmpathySessions['seduta']
validAnxietySessions.index = validAnxietySessions['SEDUTA']

#redundant but future code works on this
validEmpathySessions['sessionName'] = session_list
validAnxietySessions['sessionName'] = session_list

empathyLabels = validEmpathySessions; empathyLabels.index = empathyLabels['sessionName']; empathyLabels = empathyLabels.transpose()
anxietyLabels = validAnxietySessions; anxietyLabels.index = anxietyLabels['sessionName']; anxietyLabels = anxietyLabels.transpose()
#%%
silencesDistrib = Functions.make_df(combinedSilences,'pauseLength', ':')

fig = plt.figure(1, figsize=(9, 6))
bp = silencesDistrib.boxplot(grid=False)

plt.plot(silencesDistrib.columns,Functions.normalise(sum(silencesDistrib)))
plt.plot(validEmpathySessions['sessions'],validEmpathySessions['stdPat'], 'mo', label = 'Patient')
plt.plot(validEmpathySessions['sessions'],validEmpathySessions['stdDoc'], 'co', label = 'Doctor')
#%%
fig, axes = plt.subplots(nrows=5, ncols=3, figsize = (20,20), sharex=True)
for x, ax in zip(onlySilence, axes.flat):
    ax.plot(onlySilence[x]['Start'], onlySilence[x]['Duration'], 'o', markersize = 1)
    try:
        ax.set_title(x+ '\n Empathy pat:%.2f' %empathyLabels[x]['normPat'] + '/Empathy doc:%.2f' %empathyLabels[x]['normDoc'] 
        + '/Anxiety:%.2f' %anxietyLabels[x]['normScoring'])
    except:
        ax.set_title(x)
    plt.tight_layout()
#%%
fig, axes = plt.subplots(nrows=5, ncols=3, figsize = (20,20), sharex=True)
for x, ax in zip(longSilence, axes.flat):
    ax.plot(longSilence[x]['Start'], longSilence[x]['Duration'], 'o', markersize = 1)
    try:
        ax.set_title(x+ '\n Empathy pat:%.2f' %empathyLabels[x]['normPat'] + '/Empathy doc:%.2f' %empathyLabels[x]['normDoc'] 
        + '/Anxiety:%.2f' %anxietyLabels[x]['normScoring'])
    except:
        ax.set_title(x)
    plt.tight_layout()

#%% Plot silences with empathy and anxiety
fig = plt.subplots(nrows=1, ncols=3, figsize = (20,20), sharex=True)
ax1 = plt.subplot(111)
ax1.plot(sum(silencesDistrib),validEmpathySessions['pz'], 'o')


#%% Test
longSilenceCount = {}
for x in longSilence:
    longSilenceCount[x] = np.shape(longSilence[x])[0]

longSilenceCount = pd.DataFrame(longSilenceCount, index = [0]).transpose()
    