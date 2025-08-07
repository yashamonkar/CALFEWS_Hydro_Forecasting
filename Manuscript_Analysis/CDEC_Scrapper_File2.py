# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:58:48 2024

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

#Storage
download_raw('ORO', 15, res_to_dur_code('daily'), '1996-10-01', '2016-9-30') #ORO storage -- convert to TAF
download_raw('SHA', 15, res_to_dur_code('daily'), '1996-10-01', '2016-9-30') #SHA storage -- convert to TAF
download_raw('FOL', 15, res_to_dur_code('daily'), '1996-10-01', '2016-9-30') #FOL storage -- convert to TAF
#YRS Storage missing 
#MHB Storage missing
download_raw('PAR', 15, res_to_dur_code('daily'), '1996-10-01', '2016-9-30') 
download_raw('NHG', 15, res_to_dur_code('daily'), '1996-10-01', '2016-9-30') 
download_raw('NML', 15, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
download_raw('DNP', 15, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
download_raw('EXC', 15, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
#PFT Storage Missing
#KWH storage missing 
#SUC storage missing 
download_raw('ISB', 15, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')



###Inflow
download_raw('ORO', 76, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
download_raw('SHA', 76, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
download_raw('FOL', 76, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
#YRS inflow is missing 
#MHB inflow missing 
download_raw('PAR', 76, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
download_raw('NHG', 76, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
download_raw('NML', 76, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
download_raw('DNP', 76, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
download_raw('EXC', 76, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
download_raw('MIL', 76, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
#PFT inflow missing
#KWH inflow missing 
#SUC inflow missing 
download_raw('ISB', 76, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')





###Outflow
download_raw('ORO', 23, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
download_raw('SHA', 23, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
download_raw('FOL', 23, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
#YRS missing 
#MHB missing 
download_raw('PAR', 23, res_to_dur_code('daily'), '1996-10-01', '2016-9-30') # Does not match
download_raw('NHG', 23, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
download_raw('NML', 23, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
download_raw('DNP', 23, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
download_raw('EXC', 23, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
download_raw('MIL', 23, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
#PFT missing
#KWH missing 
#SUC missing 
download_raw('ISB', 23, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')




###Evaporation
download_raw('ORO', 74, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')  #They have Oroville and Shasta missed up. 
download_raw('SHA', 74, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')  #They have Oroville and Shasta missed up.
download_raw('FOL', 74, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
#YRS missing 
#MHB missing 
#PAR missing
#NHG missing
download_raw('NML', 74, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
#DNP missing
#EXC missing
download_raw('MIL', 74, res_to_dur_code('daily'), '1996-10-01', '2016-9-30') #Does not match
#PFT missing
#KWH missing 
#SUC missing 
#ISB missing




###Precipitation
download_raw('ORO', 45, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
download_raw('SHA', 45, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
download_raw('FOL', 45, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
#YRS missing 
#MHB missing 
download_raw('PAR', 45, res_to_dur_code('daily'), '1996-10-01', '2016-9-30') # Does not match
download_raw('NHG', 45, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
download_raw('NML', 45, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
download_raw('DNP', 45, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
download_raw('EXC', 45, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
download_raw('MIL', 45, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')
#PFT missing
#KWH missing 
#SUC missing 
download_raw('ISB', 45, res_to_dur_code('daily'), '1996-10-01', '2016-9-30')



#Pumping 
download_raw('CCC', 70, res_to_dur_code('daily'), '1996-10-01', '2016-9-30') #Banks Pumping
download_raw('HRO', 70, res_to_dur_code('daily'), '1996-10-01', '2016-9-30') #Banks Pumping
download_raw('TRP', 70, res_to_dur_code('daily'), '1999-10-01', '2016-9-30') #Tracy Pumping


