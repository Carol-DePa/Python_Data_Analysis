#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 15:07:09 2018

@author: carolinadepasquale
"""
#%% Standardise raw data in every session
cols = ["F0final_sma","voicingFinalUnclipped_sma","pcm_loudness_sma"]
allStdPat = copy.deepcopy(allPatData)
allStdDoc = copy.deepcopy(allDocData)
for x,y in zip(allPatData, allDocData):
    for column in cols:
        allStdPat[x][column] = Functions.standardise(allStdPat[x][column])
        allStdDoc[y][column] = Functions.standardise(allStdDoc[y][column])
    
#%% Descriptive stats for standardised data
patDescriptiveStatsSTD = {}
docDescriptiveStatsSTD = {}
for x,y in zip(allStdPat, allStdDoc):
    patDescriptiveStatsSTD[x] = Functions.describe_df(allStdPat[x],cols)
    docDescriptiveStatsSTD[y] = Functions.describe_df(allStdDoc[y],cols)

#%% Standardise windowed data in every session
allWinStdPat = copy.deepcopy(allWindowedPat)
allWinStdDoc = copy.deepcopy(allWindowedDoc)

for x,y in zip(allWindowedPat, allWindowedDoc):
    allWinStdPat[x]['median_F0'] = Functions.standardise(allWinStdPat[x]['median_F0'])
    allWinStdDoc[y]['median_F0'] = Functions.standardise(allWinStdDoc[y]['median_F0'])

    
#%% Interpolate all data Raw
interpolatedRawPatient = copy.deepcopy(allPatData)
interpolatedRawDoctor = copy.deepcopy(allDocData)
for x,y in zip(interpolatedRawPatient, interpolatedRawDoctor):
    interpolatedRawPatient[x]['F0final_sma'][interpolatedRawPatient[x]['F0final_sma'] > interpolatedRawPatient[x]['F0final_sma'].quantile(.70)] = np.nan
    interpolatedRawDoctor[y]['F0final_sma'][interpolatedRawDoctor[y]['F0final_sma'] > interpolatedRawDoctor[y]['F0final_sma'].quantile(.70)] = np.nan
    interpolatedRawPatient[x] = interpolatedRawPatient[x].interpolate(method = 'ffill')
    interpolatedRawDoctor[y] = interpolatedRawDoctor[y].interpolate(method = 'ffill')
    
#%% Interpolate all data STD
interpolatedSTDPatient = copy.deepcopy(allWinStdPat)
interpolatedSTDDoctor = copy.deepcopy(allWinStdDoc)
for x,y in zip(interpolatedSTDPatient, interpolatedSTDDoctor):
    interpolatedSTDPatient[x]['median_F0'][interpolatedSTDPatient[x]['median_F0'] > interpolatedSTDPatient[x]['median_F0'].quantile(.70)] = np.nan
    interpolatedSTDDoctor[y]['median_F0'][interpolatedSTDDoctor[y]['median_F0'] > interpolatedSTDDoctor[y]['median_F0'].quantile(.70)] = np.nan
    interpolatedSTDPatient[x] = interpolatedSTDPatient[x].interpolate()
    interpolatedSTDDoctor[y] = interpolatedSTDDoctor[y].interpolate()

#%% Plot stamdardised windowed for every session
stdpat = []
stddoc = []    

for x, y in zip(allWinStdPat,allWinStdDoc):
    stdpat.append(allWinStdPat[x]['median_F0'])
    stddoc.append(allWinStdDoc[y]['median_F0'])
    
fig, axes = plt.subplots(nrows=4, ncols=2, figsize = (20,20), sharex=True)
fig.suptitle('Standardised windowed F0 values for patient and doctor')
# axes.flat returns the set of axes as a flat (1D) array instead
# of the two-dimensional version we used earlier
for ax, title, x, y in zip(axes.flat, session_list, stdpat, stddoc):
    ax.plot(x, 'b', label = 'Patient')
    ax.plot(y, 'g', label = 'Doctor')
    ax.set_title(title + '\n Empathy pat:%.2f' %empathyLabels[title]['normPat'] + '/Empathy doc:%.2f' %empathyLabels[title]['normDoc'] 
    + '/Anxiety:%.2f' %anxietyLabels[title]['normScoring'])
    ax.set_ylabel('Standardised F0')
    ax.legend(loc='upper right')
    ax.grid(True)
    plt.tight_layout()
#plt.savefig('PatDocWindowedSTD.png')   
    
#%% Plot interpolated pat with doc
stdpat = []
stddoc = []
interpolstdpat = []
for x, y, z in zip(allWinStdPat,allWinStdDoc, interpolatedSTDPatient):
    stdpat.append(allWinStdPat[x]['median_F0'])
    stddoc.append(allWinStdDoc[y]['median_F0'])
    interpolstdpat.append(interpolatedSTDPatient[x]['median_F0'])

fig, axes = plt.subplots(nrows=4, ncols=2, figsize = (20,20), sharex=True)
fig.suptitle('Patient std F0 with interpolations and doctor std F0')
# axes.flat returns the set of axes as a flat (1D) array instead
# of the two-dimensional version we used earlier
for ax, title, x, y, z in zip(axes.flat, session_list, stdpat, stddoc, interpolstdpat):
    ax.plot(x, 'b', label = 'Patient')
    ax.plot(y, 'g', label = 'Doctor')
    ax.plot(z, 'r:', label = 'Interpolated Patient')
    ax.set_title(title + '\n Empathy pat:%.2f' %empathyLabels[title]['normPat'] + '/Empathy doc:%.2f' %empathyLabels[title]['normDoc'] 
    + '/Anxiety:%.2f' %anxietyLabels[title]['normScoring'])
    ax.set_ylabel('Standardised F0')
    ax.legend(loc='upper right')
    ax.grid(True)
    plt.tight_layout()
#plt.savefig('InterpolatedPatdata&realDocdata.png')
#%% plot autocorrelation plots of pat and doc raw data for every session
fig = plt.figure(figsize = (20,20))
fig.suptitle('Autocorrelation plots')
for i, x, y in zip(range(1,len(session_list*2)+1,2), allPatData, allDocData):
    tsaplots.plot_acf(allPatData[x]['melScale'].dropna(), ax = plt.subplot(4, 4, i), lags = 50, title = 'Patient {0}'.format(x))
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False) 
    tsaplots.plot_acf(allDocData[y]['melScale'].dropna(), ax = plt.subplot(4, 4, i+1), lags = 50, title = 'Doctor {0}'.format(y))
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False) 
    