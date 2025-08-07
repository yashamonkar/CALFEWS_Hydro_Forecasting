# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:40:11 2024

@author: amonkar
"""

import numpy as np
import pandas as pd 



# %% Get the stations 
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

stations = get_stations()



# %% Get the Sensors

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


sensors = get_sensors(None)


# %%

def _res_to_dur_code(res):
    map = {
        'hourly':'H',
        'daily':'D',
        'monthly':'M',
        'event':'E'}

    return map[res]



# %% Download the Raw Data. 

#dur_code = _res_to_dur_code('daily')
#station_id = 'FTO'
#sensor_num = 8
#start_date = '1999-10-01'
#end_date = '2009-10-01'



def _download_raw(station_id, sensor_num, dur_code, start_date, end_date):

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






# %%

#start = '1999-10-01'
#end = '2009-10-01'
#station_ids = ['SHA', 'ORO']
#sensor_id = 6
#resolutions = 'daily'

def get_data(station_ids=None, sensor_ids=None, resolutions=None, start, end):
    """
    Downloads data for a set of CDEC station and sensor ids. If either is not
    provided, all available data will be downloaded. Be really careful with
    choosing hourly resolution as the data sets are big, and CDEC's servers
    are slow as molasses in winter.


    Usage example::

        from ulmo import cdec
        dat = cdec.historical.get_data(['PRA'],resolutions=['daily'])

    Parameters
    ----------
    station_ids : iterable of strings or ``None``

    sensor_ids : iterable of integers or ``None``
        check out  or use the ``get_sensors()`` function to see a list of
        available sensor numbers

    resolutions : iterable of strings or ``None``
        Possible values are 'event', 'hourly', 'daily', and 'monthly' but not
        all of these time resolutions are available at every station.


    Returns
    -------
    dict : a python dict
        a python dict with site codes as keys. Values will be nested dicts
        containing all of the sensor/resolution combinations.
    """
    
    #Get the resolution code 
    dur_code = _res_to_dur_code(resolutions)

    d = {}

    for station_id in station_ids:
        d[station_id] = _download_raw(station_id, sensor_id, dur_code, start, end)

    return d





# %% #Get the sensors associated with a stations

def get_station_sensors(station_ids=None, sensor_ids=None, resolutions=None):
    """
    Gets available sensors for the given stations, sensor ids and time
    resolutions by providing the CDEC url.

    Usage example::
        ----------
        available_sensors = get_station_sensors(['ORO'])


    Parameters
    ----------
    station_id : a single string 


    Returns
    -------
    string : a url where the station meta data and sensors can be found. 
    """

    url = 'https://cdec.water.ca.gov/dynamicapp/staMeta?station_id=%s' % (station_id)


    return url