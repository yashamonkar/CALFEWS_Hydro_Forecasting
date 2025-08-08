# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:35:20 2025

@author: amonkar

Function to extract the CALFEWS output data.
"""

import h5py
import pandas as pd
import numpy as np

def get_results_sensitivity_number_outside_model(results_file, sensitivity_number):
    values = {}
    numdays_index = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    with h5py.File(results_file, 'r') as f:
        ### time series of model output
        data = f['s' + sensitivity_number]
        ### get column names for data
        c = 0
        names = []
        read_data = True
        while read_data:
            try:
                colnames = data.attrs['columns' + str(c)]
                for k in colnames:
                    names.append(k)
                c += 1
            except:
                read_data = False
        names = list(map(lambda x: str(x).split("'")[1], names))
        df_data = pd.DataFrame(data[:], columns=names)
        start_date = pd.to_datetime(data.attrs['start_date'])
        start_year = start_date.year
        start_month = start_date.month
        start_day = start_date.day
    datetime_index = []
    monthcount = start_month
    yearcount = start_year
    daycount = start_day
    leapcount = np.remainder(start_year, 4)
    for t in range(0, df_data.shape[0]):
        datetime_index.append(f"{yearcount}-{monthcount}-{daycount}")
        daycount += 1
        if leapcount == 0 and monthcount == 2 and ((yearcount % 100) > 0 or (yearcount % 400) == 0):
            numdays_month = numdays_index[monthcount - 1] + 1
        else:
            numdays_month = numdays_index[monthcount - 1]
        if daycount > numdays_month:
            daycount = 1
            monthcount += 1
            if monthcount == 13:
                monthcount = 1
                yearcount += 1
                leapcount += 1
            if leapcount == 4:
                leapcount = 0
    dt = pd.to_datetime(datetime_index) 
    df_data.index = dt
    return df_data

