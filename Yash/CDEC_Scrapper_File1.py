# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:09:39 2024

This script downloads, cleans and saves all the CDEC data. 

@author: amonkar
"""

# %%

#Load Modules/Libraries
import numpy as np
import pandas as pd 


cfs_tafd = 2.29568411*10**-5 * 86400 / 1000


# %% 
#---------These are the functions to dowload the CDEC data--------------------# 

#Get the temporal resolution
def res_to_dur_code(res):
    map = {
        'hourly':'H',
        'daily':'D',
        'monthly':'M',
        'event':'E'}

    return map[res]

#Download the raw data. 
def download_raw(station_id, sensor_num, dur_code, start_date, end_date):

    url = 'https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet' + \
          '?Stations=' + station_id + \
          '&SensorNums=' + str(sensor_num) + \
          '&dur_code=' + dur_code + \
          '&Start=' + start_date + \
          '&End=' + end_date

    df = pd.read_csv(url, header=None, skiprows=1, index_col=None, na_values='m')
    df.columns = ['STATION_ID', 'DURATION', 'SENSOR_NUMBER', 'SENSOR_TYPE',  
                  'DATE TIME', 'OBS DATE', 'VALUE' , 'DATA_FLAG' , 'UNITS' ]
    df['DATE TIME'] = pd.to_datetime(df['DATE TIME'], format='%Y%m%d %H%M')
    df['OBS DATE'] = pd.to_datetime(df['OBS DATE'], format='%Y%m%d %H%M')
    df.set_index('DATE TIME', inplace=True)

    return df['VALUE']

#Get a list of all stations 
def get_stations():
    """Fetches information on all CDEC sites.

    Returns
    -------
    df : pandas DataFrame
        a pandas DataFrame (indexed on site id) with station information.
    """
        # I haven't found a better list of stations, seems pretty janky
        # to just have them in a file, and not sure if/when it is updated.
    url = 'http://cdec.water.ca.gov/misc/all_stations.csv'
        # the csv is malformed, so some rows think there are 7 fields
    col_names = ['ID','meta_url','name','num','lat','lon','junk']
    df = pd.read_csv(url, names=col_names, header=None, usecols=range(6), quotechar="'")
    df = df.set_index('ID')

    return df


#Get the Sensors
def get_sensors(sensor_id=None):
    """
    Gets a list of sensor ids as a DataFrame indexed on sensor
    number. Can be limited by a list of numbers.

    Usage example::

        from ulmo import cdec
        # to get all available sensor info
        sensors = cdec.historical.get_sensors()
        # or to get just one sensor
        sensor = cdec.historical.get_sensors([1])

    Parameters
    ----------
    sites : iterable of integers or ``None``

    Returns
    -------
    df : pandas DataFrame
        a python dict with site codes mapped to site information
    """

    url = 'http://cdec.water.ca.gov/misc/senslist.html'
    df = pd.read_html(url,header=0)[0]
    df = df.set_index('Sensor No')

    if sensor_id is None:
        return df
    else:
        return df.loc[sensor_id]



# %%


download_raw('SBB', 8, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #BND
download_raw('FTO', 8, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #ORO fnf
download_raw('YRS', 8, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #YRS fnf
download_raw('NAT', 8, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #FOL fnf
download_raw('NML', 8, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #NML fnf
download_raw('TLG', 8, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #TLG fnf
download_raw('MRC', 8, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #MRC fnf
download_raw('SJF', 8, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #MIL fnf

#Shasta
download_raw('SHA', 76, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #SHA in
download_raw('SHA', 23, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #SHA out
download_raw('SHA', 15, res_to_dur_code('daily'), '1999-10-01', '2016-9-30')/1000 #SHA storage -- convert to TAF
download_raw('SHA', 74, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #SHA evap
download_raw('SHA', 94, res_to_dur_code('daily'), '1999-10-01', '2016-9-30')/1000 #SHA tocs -- convert to TAF
download_raw('SHA', 45, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #SHA precip


#Oroville
download_raw('ORO', 76, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #SHA in
download_raw('ORO', 23, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #SHA out
download_raw('ORO', 15, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #SHA storage -- convert to TAF
download_raw('ORO', 74, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #SHA evap
download_raw('ORO', 94, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #SHA tocs -- convert to TAF
download_raw('ORO', 45, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #SHA precip


#Folsom
download_raw('FOL', 76, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #SHA in
download_raw('FOL', 23, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #SHA out
download_raw('FOL', 15, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #SHA storage -- convert to TAF
download_raw('FOL', 74, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #SHA evap
download_raw('FOL', 94, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #SHA tocs -- convert to TAF
download_raw('FOL', 45, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #SHA precip


#Observed Delta Outflow
download_raw('DTO', 23, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') 


#Pumping 
download_raw('HRO', 70, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #Banks Pumping
download_raw('TRP', 70, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #Tracy Pumping


#Compute Delta Inflow
#Delta Inflow = Delta Outflow + Banks Pumping + Tracy Pumping


#Other reservoirs for flood control index
download_raw('FMD', 15, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #French Meadows on the American River
download_raw('UNV', 15, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #Union Valley on the American River
download_raw('HHL', 15, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #Union Valley on the American River




