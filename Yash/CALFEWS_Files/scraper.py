import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point, Polygon, LineString
#import geopandas as gpd
from matplotlib.colors import ListedColormap
import matplotlib.pylab as pl
import seaborn as sns
import sys
import calendar
from datetime import datetime, timedelta
import requests
import csv
import json
from copy import copy

class Scraper():

  def __init__(self, basins, data_labels, timestep = 'd', start_date = '1900-01-01', date_string_format = "%Y-%m-%d"):
    
    self.data_timeseries = []
    self.station_use = {}
    for keys in basins:
      self.station_use[keys] = {}
      for suffix in data_labels:
        if len(suffix) > 0:
          self.data_timeseries.append(keys + '_' + suffix)
        else:
          self.data_timeseries.append(keys)        
        self.station_use[keys][suffix] = [keys]

    self.date_index = []
    self.timestep = timestep
    self.string_start = start_date
    self.datetime_start = datetime.strptime(self.string_start,"%Y-%m-%d")
    self.datetime_end = datetime.now()
    self.string_end = self.datetime_end.strftime("%Y-%m-%d")

    if self.timestep == 'd':
      counter = 0
      current_datetime = self.datetime_start + timedelta(counter)
      while current_datetime < self.datetime_end:
        self.date_index.append(current_datetime)
        counter += 1
        current_datetime = self.datetime_start + timedelta(counter)        

    elif self.timestep == 'm':
      current_year = self.datetime_start.year
      current_month = self.datetime_start.month
      current_datetime = datetime(current_year, current_month, 1)
      while current_datetime < self.datetime_end:
        self.date_index.append(current_datetime)
        current_month += 1
        if current_month == 13:
          current_month = 1
          current_year += 1
        current_datetime = datetime(current_year, current_month, 1)
        
    elif self.timestep == 'y':
      current_year = self.datetime_start.year
      current_datetime = datetime(current_year, 1, 1)
      while current_datetime < self.datetime_end:
        self.date_index.append(current_datetime)
        current_year += 1
        current_datetime = datetime(current_year, 1, 1)
  def add_extra_stations(self, stations, labels):
    for keys, suffix in zip(stations, labels):
      if keys not in self.station_use:
        self.station_use[keys] = {}
      if len(suffix) == 0:
        self.data_timeseries.append(keys)
      else:
        self.data_timeseries.append(keys + '_' + suffix)
      self.station_use[keys][suffix] = [keys,]
   
  def initialize_dataframe(self):
    #Create DataFrame for writing input data to .csv file
    self.real_time_data = pd.DataFrame(index = self.date_index, columns = self.data_timeseries)
    print(self.real_time_data)
    for x in self.data_timeseries:
      self.real_time_data[x] = np.zeros(len(self.date_index))
   
  def add_station_map(self, basins, station_names, station_types):
    for x, y in zip(basins, station_names):
      for z in station_types:
        if len(z) > 0:
          if isinstance(y, list):
            self.station_use[x][z] = y
          else:
            self.station_use[x][z] = [y,]
        else:
          if isinstance(y, list):
            self.station_use[x]['none'] = y
          else:
            self.station_use[x]['none'] = [y,]
  
  def find_ratios(self, basins, value_type):
    self.station_coefs = {}
    for i in basins:
      monthly_values = {}
      for x in range(1, 13):
        monthly_values[str(x) + '_' + value_type + '_ratio'] = []
        monthly_values[str(x) + '_fnf'] = []
      for xx in self.real_time_data.index:
        current_month = xx.month
        if self.real_time_data.loc[xx, i + '_fnf'] > 0.0 and self.real_time_data.loc[xx, i + '_' + value_type] > 0.0:
          monthly_values[str(current_month) + '_' + value_type + '_ratio'].append(self.real_time_data.loc[xx, i + '_' + value_type]/self.real_time_data.loc[xx, i + '_fnf'])
          monthly_values[str(current_month) + '_fnf'].append(np.log(self.real_time_data.loc[xx, i + '_fnf']))

      self.station_coefs[i] = {}
      for coef_type in ['coef', 'constant', 'max_val', 'min_val']:
        self.station_coefs[i][coef_type] = np.zeros(12)
      for x in range(1,13):
        if len(monthly_values[str(x) + '_fnf']) > 0:
          coef = np.polyfit(np.array(monthly_values[str(x) + '_fnf']), np.array(monthly_values[str(x) + '_' + value_type + '_ratio']), 1)
          self.station_coefs[i]['coef'][x-1] = coef[0]
          self.station_coefs[i]['constant'][x-1] = coef[1]
          self.station_coefs[i]['max_val'][x-1] = max(np.array(monthly_values[str(x) + '_' + value_type + '_ratio']))
          self.station_coefs[i]['min_val'][x-1] = min(np.array(monthly_values[str(x) + '_' + value_type + '_ratio']))
        else:
          self.station_coefs[i]['coef'][x-1] = 0.0
          self.station_coefs[i]['constant'][x-1] = 0.0
          self.station_coefs[i]['max_val'][x-1] = 0.0
          self.station_coefs[i]['min_val'][x-1] = 0.0
          
  def adjust_fnf_monthly(self, monthly_real_time_data, stations_basin):
    for i in stations_basin:
      monthly_int_df = pd.DataFrame(index = monthly_real_time_data.index, columns = ['sum'])
      monthly_int_df['sum'] = np.zeros(len(monthly_real_time_data.index))
      for xx in self.real_time_data.index:
        current_month = xx.month
        current_year = xx.year
        monthly_index = datetime(current_year, current_month, 1)
        monthly_int_df.loc[monthly_index, 'sum'] += self.real_time_data.loc[xx, i + '_fnf']
      for xx in self.real_time_data.index:
        current_month = xx.month
        current_year = xx.year
        monthly_index = datetime(current_year, current_month, 1)
        if monthly_int_df.loc[monthly_index, 'sum'] > 0.0:
          self.real_time_data.loc[xx, i + '_fnf'] = self.real_time_data.loc[xx, i + '_fnf'] * monthly_real_time_data.loc[monthly_index, i + '_fnf'] / monthly_int_df.loc[monthly_index,'sum']        
        else:
          self.real_time_data.loc[xx, i + '_fnf'] = monthly_real_time_data.loc[monthly_index, i + '_fnf'] / 30.0    
        
  def fill_missing(self, basin, station_coefs, monthly_data):
    for i in basin:
      for xx in self.real_time_data.index:
        current_month = xx.month
        current_year = xx.year
        monthly_index = datetime(current_year, current_month, 1)
        if self.real_time_data.loc[xx, i + '_fnf'] == 0.0 and monthly_data.loc[monthly_index, i + '_inf'] > 0.0:
          self.real_time_data.loc[xx, i + '_fnf'] = monthly_data.loc[monthly_index, i + '_fnf'] * self.real_time_data.loc[xx, i + '_inf']/monthly_data.loc[monthly_index, i + '_inf']

      #i = CALFEWS input data basin label (for writing)
      #sens = CDEC station sensor data identifier (for reading)
      prev_index = copy(self.real_time_data.index[0])
      for xx in range(0, len(self.real_time_data.index)):
        ###Fill in empty daily full-natural flows from monthly full-natural flow values (using % of monthly inflow occuring on a given date)
        current_month = self.real_time_data.index[xx].month
        current_year = self.real_time_data.index[xx].year
        monthly_index = datetime(current_year, current_month, 1)

        if xx > 0 and xx < len(self.real_time_data.index) - 1:
          for lab in ['fnf', 'inf']:
            if self.real_time_data.loc[self.real_time_data.index[xx], i + '_' + lab] > self.real_time_data.loc[self.real_time_data.index[xx-1], i + '_' + lab] * 200.0 and self.real_time_data.loc[self.real_time_data.index[xx], i + '_' + lab] > self.real_time_data.loc[self.real_time_data.index[xx+1], i + '_' + lab] * 200.0:
              self.real_time_data.loc[self.real_time_data.index[xx], i + '_' + lab] = 0.0  

        fnf_day = copy(self.real_time_data.loc[self.real_time_data.index[xx], i + '_fnf'])
        inf_day = copy(self.real_time_data.loc[self.real_time_data.index[xx], i + '_inf'])
        fnf_month = copy(monthly_data.loc[monthly_index, i + '_fnf'])
        inf_month = copy(monthly_data.loc[monthly_index, i + '_inf'])
        
        #if missing daily fnf data and have monthly fnf & inf data, take daily fnf data from monthly fnf w/ daily fraction of inf data
        if fnf_day == 0.0 and inf_month > 0.0 and fnf_month > 0.0:
          self.real_time_data.loc[self.real_time_data.index[xx], i + '_fnf'] =  inf_day * fnf_month / inf_month
        #if missing daily fnf data and only have monthly fnf data, use average daily fnf from monthly fnf value
        elif fnf_day == 0.0 and fnf_month > 0.0:
          self.real_time_data.loc[self.real_time_data.index[xx], i + '_fnf'] = fnf_month/30.0
        #if no monthly data, use previous day's data
        elif fnf_day == 0.0: 
          self.real_time_data.loc[self.real_time_data.index[xx], i + '_fnf'] = self.real_time_data.loc[prev_index, i + '_fnf']
          
        #if missing daily inf and have monthly fnf and daily fnf, use inf/fnf ratios and daily fnf fraction, 
        if inf_day == 0.0 and fnf_month > 0.0:
          monthly_inf = station_coefs[i]['coef'][current_month - 1] * np.log(fnf_month) + station_coefs[i]['constant'][current_month - 1]
          monthly_inf = max(min(monthly_inf, station_coefs[i]['max_val'][current_month - 1]), station_coefs[i]['min_val'][current_month - 1]) * fnf_month
          if fnf_day > 0.0:
            self.real_time_data.loc[self.real_time_data.index[xx], i + '_inf'] = monthly_inf * fnf_day * 1.9835 / fnf_month
          elif self.real_time_data.loc[self.real_time_data.index[xx], i + '_fnf'] == 0.0:
            self.real_time_data.loc[self.real_time_data.index[xx], i + '_inf'] = monthly_inf/30.0
        elif self.real_time_data.loc[self.real_time_data.index[xx], i + '_inf'] == 0.0:
          self.real_time_data.loc[self.real_time_data.index[xx], i + '_inf'] = self.real_time_data.loc[prev_index, i + '_inf']

        prev_index = copy(self.real_time_data.index[xx])
        
  def make_cumulative_fci(self ,station_name, variable_name, daily_decay, daily_multiplier, min_val, max_val, extra_stations = 'none', extra_sensor = 'none', extra_label = 'none', url_parts = 'none'):
    variable_column = station_name + '_' + variable_name
    fci_vals = np.zeros(len(self.real_time_data[variable_column]))
    if extra_stations == 'none':
      skip_line = 1
    else:
      new_values = pd.DataFrame(index = self.real_time_data.index)
      new_values['total_val'] = np.zeros(len(self.real_time_data.index))
      for extra_station_name in extra_stations:
        get_name = url_parts[0] + extra_station_name + url_parts[1] + str(extra_sensor) + url_parts[2] + self.string_start + url_parts[3] + self.string_end
        extra_values = self.activate_link(get_name)
        prev_value = 0.0
        for row in extra_values:
          use_row = True
          #is the line a data field?
          try:
            sens_match = int(row[2])
          except:
            use_row = False
          if use_row:
            val_day, index_date, prev_value = self.read_cdec_row(row, extra_label, prev_value)
            if index_date >= self.datetime_start:      
              new_values.loc[index_date, 'total_val'] += val_day

      
    variable_column = station_name + '_' + variable_name
    fci_vals = np.zeros(len(self.real_time_data[variable_column]))
    for x in range(0, len(self.real_time_data[variable_column])):
      if extra_stations == 'none':
        daily_val = self.real_time_data.loc[self.real_time_data.index[x], variable_column]
      else:
        daily_val = new_values.loc[self.real_time_data.index[x], 'total_val']
      
      if self.real_time_data.index[x].month == 10 and self.real_time_data.index[x].day == 1:
        fci_vals[x] = max(min(daily_val * daily_multiplier, max_val), min_val)
      elif self.real_time_data.index[x].month >= 10 or self.real_time_data.index[x].month <= 2:
        if x > 0:
          fci_vals[x] = max(min(fci_vals[x-1] * daily_decay + daily_val * daily_multiplier, max_val), min_val)
        else:
          fci_vals[x] = max(min(daily_val * daily_multiplier, max_val), min_val)
      else:
        if x > 0:
          fci_vals[x] = fci_vals[x-1]  * daily_decay 
        else:
          fci_vals[x] = max(min(daily_val * daily_multiplier, max_val), min_val)
          
    self.real_time_data[station_name + '_fci'] = fci_vals
    
  def make_upstream_storage_fci(self, url_parts, station_list, total_auxiliary_capacity, max_val_list, fci_res_station):
    for hsl, tac, mvl in zip(station_list, total_auxiliary_capacity, max_val_list):
      get_name = url_parts[0] + hsl + url_parts[1] + str(15) + url_parts[2] + self.string_start + url_parts[3] + self.string_end
      storage_values = self.activate_link(get_name)
      prev_value = 0.0
      for row in storage_values:
        use_row = True
        #is the line a data field?
        try:
          sens_match = int(row[2])
        except:
          use_row = False
        #read data field line (one line per day at each type/basin)
        if use_row:
          val_day, index_date, prev_value = self.read_cdec_row(row, 'storage', prev_value)
          if index_date >= self.datetime_start:          
            if fci_res_station == 'PFT':
              self.real_time_data.loc[index_date, fci_res_station + '_fci'] += min(max(tac - val_day, 0.0), mvl)
            else:
              self.real_time_data.loc[index_date, fci_res_station + '_fci'] += min(max(tac - val_day, 0.0)/1000.0, mvl)

  def adjust_upstream_monthly(self, url_parts, station_list, total_auxiliary_capacity, max_val_list, fci_res_station):
  
    for hsl, tac, mvl in zip(station_list, total_auxiliary_capacity, max_val_list):
      get_name = url_parts[0] + hsl + url_parts[1] + str(15) + url_parts[2] + self.string_start + url_parts[3] + self.string_end
      storage_values = self.activate_link(get_name)
      prev_value = 0.0
      for row in storage_values:
        use_row = True
        #is the line a data field?
        try:
          sens_match = int(row[2])
        except:
          use_row = False
        #read data field line (one line per day at each type/basin)
        if use_row:
          val_day, index_date, prev_value = self.read_cdec_row(row, 'storage', prev_value)
          if index_date >= self.datetime_start:
            current_month = int(index_date.month)
            while int(index_date.month) == current_month and index_date < self.datetime_end:
              if fci_res_station == 'PFT':
                self.real_time_data.loc[index_date, fci_res_station + '_fci'] += max(tac - val_day, 0.0)
              else:
                self.real_time_data.loc[index_date, fci_res_station + '_fci'] += max(tac - val_day, 0.0)/ 1000.0
              self.real_time_data.loc[index_date, fci_res_station + '_fci'] = min(self.real_time_data.loc[index_date, fci_res_station + '_fci'], mvl)
              index_date += timedelta(1)       

  
  def assemble_fci(self, url_parts_d, url_parts_m):
    #SHASTA
    self.make_cumulative_fci('SHA', 'inf', 0.95, 1.983459, 0.0, 530000.0)

    #OROVILLE
    self.make_cumulative_fci('ORO', 'precip', 0.97, 1.0, 3.5, 11.0)

    ###FOLSOM
    folsom_headwaters_station_names = ['FMD', 'UNV', 'HHL']
    total_aux_cap = [125601.0, 277000.0, 208400.0]
    max_space = [45.0, 75.0, 80.0]
    self.make_upstream_storage_fci(url_parts_d, folsom_headwaters_station_names, total_aux_cap, max_space, 'FOL')

    #PARDEE
    pardee_headwaters_station_names = ['SLS',]
    total_aux_cap = [141900.0,]
    max_space = [70.0,]
    self.make_upstream_storage_fci(url_parts_d, pardee_headwaters_station_names, total_aux_cap, max_space, 'PAR')
    pardee_headwaters_station_names = ['LWB',]
    total_aux_cap = [52000.0,]
    max_space = [70.0,]    
    self.adjust_upstream_monthly(url_parts_m, pardee_headwaters_station_names, total_aux_cap, max_space, 'PAR')

    #NEW HOGAN
    self.make_cumulative_fci('NHG', 'precip', 0.96, 33.0/121.0, 0.0, 999999.0, extra_stations = ['NHG', 'RRF', 'SHR', 'PRY'], extra_sensor = [45,], extra_label = 'precip', url_parts = url_parts_d)

    #MILLERTON
    millerton_headwaters_station_names = ['MPL',]
    total_aux_cap = [123000.0,]
    max_space = [85.0,]    
    self.adjust_upstream_monthly(url_parts_m, millerton_headwaters_station_names, total_aux_cap, max_space, 'MIL')

    #PINE FLAT
    millerton_headwaters_station_names = ['CTG','WSN']
    total_aux_cap = [123300.0,128300.0]
    max_space = [20000.0, 20000.0]    
    self.adjust_upstream_monthly(url_parts_m, millerton_headwaters_station_names, total_aux_cap, max_space, 'PFT')
   
      
  def activate_link(self, get_name):
    response = requests.get(get_name) # i must be converted to a string
    #headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36',}
    #response = requests.post(get_name, headers=headers)
    response.raise_for_status()

    streamflow_files = csv.reader(response.text.strip().split('\n'))

    return streamflow_files
    
  def read_cdec_row(self, row, lab, prev_value):
  
    index_num = row[2]
    date_str = row[4]
    #record date (sometimes the files skip days ?I think? so you cant just do a count)
    current_year = int(date_str[0:4])
    current_month = int(date_str[4:6])
    current_day = int(date_str[6:8])
    if self.timestep == 'd':
      index_date = datetime(current_year, current_month, current_day)              
    elif self.timestep == 'm':
      index_date = datetime(current_year, current_month, 1)              
    else:
      index_date = datetime(current_year, 1, 1)              

    no_val = 0           
    if lab == 'snow':
      try:
        val_day = max(float(row[6]), prev_value)
      except:
        no_val = 1
    elif lab == 'fnf' and self.timestep == 'm':
      try:
        val_day = float(row[6])
      except:
        no_val = 1
    elif lab == 'fnf':
      try:
        val_day = float(row[6]) * 1.9835
      except:
        no_val = 1
    else:
      try:
        val_day = float(row[6])
      except:
        no_val = 1

    if no_val == 1:
      if self.timestep == 'm' and lab == 'fnf':
        val_day = 0.0
      else:
        val_day = max(prev_value, 0.0)
    if lab == 'storage' and val_day <= 0.0:
      val_day = max(prev_value, 0.0)
    elif lab == 'temp':
      if val_day < 32.0 or val_day > 100.0:
        val_day = max(prev_value, 0.0)
    elif lab == 'fnf' or lab == 'inf':
      if val_day == -9999.0 or val_day == -999.0:
        val_day = 0.0
    elif lab == 'otf' or lab == 'fnf' or lab == 'inf' or lab == 'flow':
      if val_day <= 0.0:
        val_day = max(prev_value, 0.0)

    if lab == 'storage' or lab == 'temp' or lab == 'flow' or lab == 'elevation':
      prev_value = val_day * 1.0
    #reset snowpack maximum value at the end of each water-year
    elif lab == 'snow':
      if current_day == 30 and current_month == 9:
        prev_value = 0.0
      else:
        prev_value = max(val_day * 1.0, prev_value)   
    elif lab == 'otf' or lab == 'pump' or len(lab) == 0:
      prev_value = val_day * 1.0
    else:
      prev_value = 0.0

    return val_day, index_date, prev_value

  def adjust_values(self, station_basin, label_basin, adjust_stations, adjust_sensors, url_parts):
    for lab, sens in zip(adjust_stations, adjust_sensors):
      get_name = url_parts[0] + lab + url_parts[1] + str(sens) + url_parts[2] + self.string_start + url_parts[3] + self.string_end # i must be converted to      
      streamflow_files = self.activate_link(get_name)              
      prev_value = 0.0
      for row in streamflow_files:
        use_row = True
        #is the line a data field?
        try:
          sens_match = int(row[2])
        except:
          use_row = False
        #read data field line (one line per day at each type/basin)
        if use_row:
          val_day, index_date, prev_value = self.read_cdec_row(row, label_basin, prev_value)
          if index_date >= self.datetime_start:
            self.real_time_data.loc[index_date, station_basin + '_' + label_basin] += val_day
      
  def make_monthly_daily(self, stn, sfx):
    for ixdt in self.real_time_data.index:
      if self.real_time_data.loc[ixdt, stn + '_' + sfx] == 0.0:
        self.real_time_data.loc[ixdt, stn + '_' + sfx] = self.real_time_data.loc[ixdt - timedelta(1), stn + '_' + sfx]  
        
  def link_api(self, stations_basin, sensor_basin, labels_basin, url_parts, url_parts_h, hourly_dict, use_delta_cats = False):
    #API link w/ CDEC website - basin level data
    if use_delta_cats:
      start_date_string = self.string_end_dayflow
      leap_count = 3
    else:
      start_date_string = self.string_start
    
    for i in stations_basin:
      #i = CALFEWS input data basin label (for writing)
      #lab = CALFEWS input data type label (for writing)
      #station_api = CDEC station label(s) associated with each basin (can be different for different data types)
      #sens = CDEC station sensor data identifier (for reading)
      for sens, lab in zip(sensor_basin, labels_basin):
        if len(lab) > 0:
          station_api = self.station_use[i][lab]
        else:
          station_api = self.station_use[i]['none']
        for yy in station_api:
          #flow refers to when the inf/otf data is from a river gauge and not a reservoir
          #none means there is not data for that type/basin
          if yy == 'none':
            streamflow_files = []
          else:
            if lab in hourly_dict[i]:
              if lab == 'temp':
                get_name = url_parts_h[0] + yy + url_parts_h[1] + str(sens) + url_parts_h[2] + start_date_string + url_parts_h[3] + self.string_end # i must be converted to a string
              else:
                get_name = url_parts_h[0] + yy + url_parts_h[1] + str(20) + url_parts_h[2] + start_date_string + url_parts_h[3] + self.string_end # i must be converted to a string
              total_count = np.zeros(len(self.real_time_data[i + '_' + lab]))      
              print(get_name)
              print(total_count)              
            elif yy == 'flow':
              get_name = url_parts[0] + i + url_parts[1] + str(41) + url_parts[2] + start_date_string + url_parts[3] + self.string_end # i must be converted to a string
            #if no fnf, use inf station (for NHG - Calaveras River)
            elif yy == 'nofnf':
              get_name = url_parts[0] + i + url_parts[1] + str(76) + url_parts[2] + start_date_string + url_parts[3] + self.string_end # i must be converted to a string
            elif yy == 'usefnf':
              get_name = url_parts[0] + i + url_parts[1] + str(8) + url_parts[2] + start_date_string + url_parts[3] + self.string_end # i must be converted to a string
            else:
              get_name = url_parts[0] + yy + url_parts[1] + str(sens) + url_parts[2] + start_date_string + url_parts[3] + self.string_end # i must be converted to      
            print('Start Link: ' + i + ' ' + lab + ' (CDEC Station: ' + yy + ' ' + str(sens) + ')') 
            streamflow_files = self.activate_link(get_name)              

          #prev_value = what to do when there is no data point on a given day
          prev_value = 0.0
          early_toggle = 0
          start_backfill = datetime(1900, 1, 1)   
          for row in streamflow_files:
            use_row = True
            #is the line a data field?
            try:
              sens_match = int(row[2])
            except:
              use_row = False
            #read data field line (one line per day at each type/basin)
            if use_row:
              val_day, index_date, prev_value = self.read_cdec_row(row, lab, prev_value)
              
              if use_delta_cats:
                self.record_delta_values(val_day, index_date, yy, i + '_' + lab)
              else:
                if lab == 'storage' and prev_value == 0.0 and early_toggle == 0:
                  start_backfill = index_date
                  early_toggle = 1
                if early_toggle == 1 and val_day > 0.0:
                  while start_backfill < index_date:
                    self.real_time_data.loc[start_backfill, i + '_' + lab] += val_day
                    if self.timestep == 'd':
                      start_backfill += timedelta(1)
                    elif self.timestep == 'm':
                      if index_date.month == 12:                        
                        start_backfill = datetime(index_date.year + 1, 1, 1)
                      else:
                        start_backfill = datetime(index_date.year, index_date.month + 1, 1)
                    else:
                      start_backfill = datetime(index_date.year + 1, 1, 1)
                      
                  early_toggle = 0

                #gains are recorded as the difference between the gauge value and upstream releases
                if len(lab) > 0:
                  self.real_time_data.loc[index_date, i + '_' + lab] += val_day
                else:
                  self.real_time_data.loc[index_date, i] += val_day
                if lab in hourly_dict[i]:
                  day_num = index_date - self.datetime_start
                  total_count[day_num.days] += 1
                elif lab == 'gains':
                  self.real_time_data.loc[index_date, i + '_' + lab] -= self.real_time_data.loc[index_date, i + '_otf']
                  
                if lab == 'fnf' or lab == 'inf':
                  if self.real_time_data.loc[index_date, i + '_' + lab] < 0.0 and self.timestep == 'd':
                    total_value = 0.0
                    lookback_length = 0
                    past_original_date = False
                    while total_value > self.real_time_data.loc[index_date, i + '_' + lab]:
                      if self.real_time_data.loc[index_date - timedelta(lookback_length), i + '_' + lab] == 0.0:
                        break
                      lookback_length += 1
                      try:
                        total_value -= self.real_time_data.loc[index_date - timedelta(lookback_length), i + '_' + lab]
                      except:
                        past_original_date = True
                      if past_original_date:
                        break
                    for x in range(0, lookback_length + 1):
                      if index_date - timedelta(x) >= self.datetime_start:
                        self.real_time_data.loc[index_date - timedelta(x), i + '_' + lab] = max(val_day - total_value, 0.0) /(lookback_length + 1)
                elif index_date > self.real_time_data.index[0] and lab == station_api[-1]:
                  if lab == 'storage' and self.real_time_data.loc[index_date, i + '_' + lab] < 0.15 * self.real_time_data.loc[index_date - timedelta(1), i + '_' + lab]:
                    self.real_time_data.loc[index_date, i + '_' + lab] = self.real_time_data.loc[index_date - timedelta(1), i + '_' + lab] * 1.0

                #fill empty storage/snowpack values with the previous day, empty flow values w/ zeros
          if lab in hourly_dict[i]:
            for ixdt in self.real_time_data.index:
              day_num = ixdt - self.datetime_start
              if total_count[day_num.days] == 0 and day_num.days > 0:
                self.real_time_data.loc[ixdt, i + '_' + lab] =  self.real_time_data.loc[ixdt - timedelta(1), i + '_' + lab]
              else:
                if total_count[day_num.days] > 0:              
                  self.real_time_data.loc[ixdt, i + '_' + lab] = self.real_time_data.loc[ixdt, i + '_' + lab]/total_count[day_num.days]
                  if lab == 'gains':                   
                    self.real_time_data.loc[ixdt, i + '_' + lab] -=  self.real_time_data.loc[ixdt, i + '_otf']
                  
  def link_dayflow_api(self):
    #GCD (gross chanel depletions) and CCC (Contra Costa Canal) are not in the CDEC data used to 'fill' more recent days, 
    #so we just use seasonal averages that are taken from the existing DAYFLOW series
    self.delta_averages = pd.DataFrame(index = np.arange(366), columns = ['GCD', 'CCC', 'count'])
    self.delta_averages['GCD'] = np.zeros(366)
    self.delta_averages['CCC'] = np.zeros(366)
    self.delta_averages['count'] = np.zeros(366)
    self.string_end_dayflow = '2020-10-01'
    #API link w/ DAYFLOW website - delta data (note - this is only updated once per year so it needs to be 'filled' w/ CDEC data for more recent days - CDEC data isn't quite as accurate but is a good proxy)
    print('Start Link w/ DAYFLOW') 
    response = requests.get('https://data.cnra.ca.gov/dataset/06ee2016-b138-47d7-9e85-f46fae674536/resource/21c377fe-53b8-4bd6-9e1f-2025221be095/download/dayflow-results-1997-2023.csv') # i must be converted to a string
    response.raise_for_status()
    streamflow_files = csv.reader(response.text.strip().split('\n'))
    counter = 0
    leap_count = 0
    for row in streamflow_files:
      #first line in the CDEC file is a header, the rest are real data
      if counter == 0:
        counter += 1
      else:
        #DAYFLOW dates can be directly read into datetime form
        current_date = row[2]
        index_date =  datetime.strptime(current_date,"%m/%d/%Y")
        #for gains to Rio Vista/Vernalis, take flow minus total flow at all upstream gauge points
        #total flow at upstream gauges are the 'gains' plus the reservoir releases
        #RIO VISTA
        tot_sac_release = 0.0
        for trib_name in ['SHA', 'ORO', 'YRS', 'FOL']:
          tot_sac_release += self.real_time_data.loc[index_date, trib_name + '_otf'] + self.real_time_data.loc[index_date, trib_name + '_gains']
        rv_gains = float(row[10]) - float(row[9]) - tot_sac_release
        #VERNALIS
        tot_sj_release = 0.0
        for trib_name in ['NML', 'DNP', 'EXC']:
          tot_sj_release += self.real_time_data.loc[index_date, trib_name + '_otf'] + self.real_time_data.loc[index_date, trib_name + '_gains']
        ver_gains = float(row[8]) - tot_sj_release
        #delta depletions are Gross Channel Depletions minus total precipitation (i.e., positive 'delta depletion' values are net losses of flow between RIO VISTA/VERNALIS and delta outflows
        delt_dep = float(row[17]) - float(row[16])
        #EASTSIDE gains are the total eastside gains minus flow at vernalis (i.e, flow in Calaveras, Consumnes, Mokelumne)
        eastside_gains = float(row[9]) - float(row[8])
        #Barker/Contra Costa Withdrawals
        contra_pump = float(row[11])
        barker_pump = float(row[14])
        #Total delta inflow (SAC + SJ + EASTSIDE FLOWS)
        delta_inf = float(row[10])
        #Total flow at vernalis (different than vernalis 'gains' recorded above)
        vernalis_inf = float(row[8])
        
        self.real_time_data.loc[index_date, 'MHB_gains'] = float(row[5]) - self.real_time_data.loc[index_date, 'MHB_otf']
        self.real_time_data.loc[index_date, 'PAR_gains'] = float(row[6]) - self.real_time_data.loc[index_date, 'PAR_otf']
        self.real_time_data.loc[index_date, 'NHG_gains'] = float(row[7]) - self.real_time_data.loc[index_date, 'NHG_otf']
        
        self.real_time_data.loc[index_date, 'SAC_gains'] = rv_gains
        self.real_time_data.loc[index_date, 'SJ_gains'] = ver_gains
        self.real_time_data.loc[index_date, 'EAST_gains'] = eastside_gains
        self.real_time_data.loc[index_date, 'delta_depletions'] = delt_dep
        self.real_time_data.loc[index_date, 'CCC_pump'] = contra_pump
        self.real_time_data.loc[index_date, 'BAK_pump'] = barker_pump
        self.real_time_data.loc[index_date, 'delta_inflow'] = delta_inf
        self.real_time_data.loc[index_date, 'vernalis_inflow'] = vernalis_inf

        #Day of the year to record average seasonal values for CCC & GCD measures
        day_of_year = index_date.timetuple().tm_yday  # returns 1 for January 1st
        if leap_count == 0 and day_of_year > 59:
          day_of_year -= 1
        if day_of_year == 365:
          leap_count += 1
        if leap_count == 4:
          leap_count = 0  
        self.delta_averages.loc[day_of_year-1, 'GCD'] += float(row[16])
        self.delta_averages.loc[day_of_year-1, 'CCC'] += contra_pump
        self.delta_averages.loc[day_of_year-1, 'count'] += 1.0

  def record_delta_values(self, val_day, index_date, cdec_station, label_name):
    day_of_year = index_date.timetuple().tm_yday
    if calendar.isleap(index_date.year) and day_of_year > 59:
            day_of_year -= 1

    #YBY is yolo bypass - part of sac gains
    if cdec_station == 'YBY':
      self.real_time_data.loc[index_date, label_name] += val_day
    #fpt is sac flow at freeport
    elif cdec_station == 'FPT':
      self.real_time_data.loc[index_date, label_name] += val_day
      for trib_name in ['MHB', 'PAR', 'NHG']:
        self.real_time_data.loc[index_date, 'EAST_gains'] += self.real_time_data.loc[index_date, trib_name + '_otf']
      for trib_name in ['SHA', 'ORO', 'YRS', 'FOL']:
        self.real_time_data.loc[index_date, label_name] -= self.real_time_data.loc[index_date, trib_name + '_otf'] + self.real_time_data.loc[index_date, trib_name + '_gains']
    #vns is san joaquin flow at vernalis
    elif cdec_station == 'VNS':
      self.real_time_data.loc[index_date, 'vernalis_inflow'] += val_day
      self.real_time_data.loc[index_date, label_name] += val_day
      for trib_name in ['NML', 'DNP', 'EXC']:
        self.real_time_data.loc[index_date, label_name] -=  self.real_time_data.loc[index_date, trib_name + '_otf'] + self.real_time_data.loc[index_date, trib_name + '_gains']
    #sfs is stockton precip station (flow is equal to rain - converted from inches to ft, mult by 678,200 acres of drainage)
    elif cdec_station == 'SFS':
      self.real_time_data.loc[index_date, label_name] += val_day * 678200.0/(12.0*1.98) - (self.delta_averages.loc[day_of_year-1, 'GCD']/self.delta_averages.loc[day_of_year-1, 'count'])
    #bks is barker slough
    elif cdec_station == 'BKS':
      self.real_time_data.loc[index_date, label_name] += val_day
      self.real_time_data.loc[index_date, 'CCC_pump'] += self.delta_averages.loc[day_of_year-1, 'CCC']/self.delta_averages.loc[day_of_year-1, 'count']
      self.real_time_data.loc[index_date, 'delta_inflow'] += self.real_time_data.loc[index_date, 'SAC_gains'] + self.real_time_data.loc[index_date, 'SJ_gains'] + self.real_time_data.loc[index_date, 'EAST_gains']
      
      
  def fix_missing_inputs(self):
    self.real_time_data['NHG_fnf'] = self.real_time_data['PAR_fnf'] * 1.0
    self.real_time_data['YRS_evap'] = self.real_time_data['FOL_evap'] * 7.344 / 17.89 #surface area ratio
    self.real_time_data['MHB_evap'] = self.real_time_data['NML_evap'] * 0.0#no reservoir
    self.real_time_data['PAR_evap'] = self.real_time_data['NML_evap'] * (2134.0 + 7700.0) / 12500.0#surface area ratio
    self.real_time_data['NHG_evap'] = self.real_time_data['NML_evap'] * 4410.0 / 12500.0#surface area ratio
    self.real_time_data['DNP_evap'] = self.real_time_data['NML_evap'] * 13000.0 / 12500.0#surface area ratio
    self.real_time_data['EXC_evap'] = self.real_time_data['NML_evap'] * 7147.0 / 12500.0#surface area ratio
    self.real_time_data['EXC_evap'] = self.real_time_data['NML_evap'] * 7147.0 / 12500.0#surface area ratio
    self.real_time_data['PFT_evap'] = self.real_time_data['PFT_evap'] * 0.75 * 5970.0 / (12.0 * 1.983459)#pan-evap inches to cfs (using pft surface area)
    self.real_time_data['KWH_evap'] = self.real_time_data['KWH_evap'] * 0.75 * 1945.0 / (12.0 * 1.983459)#pan-evap inches to cfs (using kwh surface area)
    self.real_time_data['SUC_evap'] = self.real_time_data['KWH_evap'] * 0.75 * 2450.0 / (12.0 * 1.983459)#pan-evap inches to cfs (using kwh surface area)
    self.real_time_data['ISB_evap'] = self.real_time_data['ISB_evap'] * 0.75 * 1100.0 / (12.0 * 1.983459)#pan-evap inches to cfs (using kwh surface area)
    
    
  def make_figure_comparison(self, stations_basin, labels_basin, old_filename, new_filename):
    original_values = pd.read_csv(old_filename)
    original_values['datetime'] = pd.to_datetime(original_values['datetime'])
    original_values = original_values.set_index('datetime')
    realtime_values = pd.read_csv(new_filename)
    realtime_values['datetime'] = pd.to_datetime(realtime_values['Unnamed: 0'])
    realtime_values = realtime_values.set_index('datetime')
    print(original_values)
    print(realtime_values)
    for station_labels in original_values:
      print(station_labels)
      if station_labels != 'realization' and station_labels in realtime_values:
        fig, ax = plt.subplots(3)
        differences = np.zeros(len(original_values[station_labels]))
        cumulative_differences = np.zeros(len(original_values[station_labels]))
        running_cumu = 0.0
        for xx in range(0, len(original_values[station_labels])):
          differences[xx] = original_values[station_labels][xx] - realtime_values[station_labels][xx]
          cumulative_differences[xx] = running_cumu + differences[xx]
          running_cumu += differences[xx]
        ax[0].plot(original_values.index, original_values[station_labels], 'blue', linewidth = 1.5)
        ax[0].plot(realtime_values.index, realtime_values[station_labels], 'red', linewidth = 1.0)
        ax[1].plot(original_values.index, np.zeros(len(original_values[station_labels])), 'black', linewidth = 1.0, linestyle = '--')
        ax[1].plot(original_values.index, differences, 'black', linewidth = 1.0)
        ax[2].plot(original_values.index, cumulative_differences, 'black')
        ax[0].set_ylabel(station_labels)
        ax[1].set_ylabel('difference')
        ax[2].set_ylabel('cumulative difference')
        #for x in range(0,2):
          #ax[x].set_xticks([])
        twinaxis = ax[1].twiny()
        sns.kdeplot(y = differences, shade = True, ax = twinaxis)
        twinaxis.set_xlabel('')
        twinaxis.set_ylabel('')
        
        for ax_use in range(0, 3):
          ax[ax_use].set_xlim([realtime_values.index[0], realtime_values.index[-1]])
        plt.tight_layout()
        plt.savefig('comp_figures/' + station_labels + '.png', dpi = 300)
        plt.close()
      elif station_labels != 'realization':
        print('no label: ' + station_labels)