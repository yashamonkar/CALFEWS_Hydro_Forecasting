# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:17:55 2024

@author: amonkar
"""

# Set the working directory
import os
working_directory = r'C:\Users\amonkar\Documents\GitHub\CALFEWS'
os.chdir(working_directory)


# import libraries
import numpy as np
import pandas as pd
import h5py
import json
from datetime import datetime
import matplotlib.pyplot as plt
from itertools import compress
from datetime import datetime
from calfews_src import *
from calfews_src.visualizer import Visualizer
import shutil
from scipy.stats import linregress





# %%
#Read the EIA dataframe -- for validation
eia = pd.read_csv('Yash/EIA/EIA_Monthy_Gen.csv', index_col=0)

#Read the WAPA dataset
wapa = pd.read_csv('Yash/WAPA/WAPA_Daily_Gen.csv', index_col=0)
wapa.index = pd.to_datetime(wapa.index)


# results hdf5 file location from CALFEWS simulations
# Define the output folder path -- Added by Yash
output_folder = "results/Historical_validation_1997-2016/"
output_file = output_folder + 'results.hdf5'
fig_folder = output_folder + 'figs/'


# %%


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
      datetime_index.append(str(yearcount) + '-' + str(monthcount) + '-' + str(daycount))
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


# now load simulation output
datDaily = get_results_sensitivity_number_outside_model(output_file, '')



def get_columns_with_ending(df, ending):
    # Filter columns that end with the specified ending
    filtered_columns = [col for col in df.columns if col.endswith(ending)]
    # Create a subset of the dataframe with the filtered columns
    subset_df = df[filtered_columns]
    return subset_df

getStorage = get_columns_with_ending(datDaily, '_S')
getRelease = get_columns_with_ending(datDaily, '_R')



# %% --- CALFEWS Generation Exploratory Plots

#Inputs
#1. Daily CALFEWS Generation in GWh (calfews_gen)
#2. Reservoir Name (res_name)

#Output:- 
#1. Daily Generation plot
#2. Monthly Generation plot
#3. Histogram of daily generation


def get_calfews_gen_plots(calfews_gen, res_name):
    
    
    #Daily Generation Plot
    plt.figure(figsize=(10, 6))
    calfews_gen.plot()
    plt.title(f'Daily Hydropower Generation at {res_name}', fontsize = 18)
    plt.xlabel('Date', fontsize = 14)
    plt.ylabel('GWh', fontsize = 14)
    plt.xticks(fontsize=12)  # Increase x-axis tick fontsize
    plt.yticks(fontsize=12)  # Increase y-axis tick fontsize
    plt.grid(True)
    plt.show()
    
    #Montly CALFEWS Generation Plot
    monthly_series = calfews_gen.resample('M').sum()
    
    # Plotting the monthly data
    plt.figure(figsize=(10, 6))
    monthly_series.plot()
    plt.title(f'Monthly Generation - {res_name}', fontsize=18)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('GWh', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)

    #ax_inset = plt.axes([0.55, 0.53, 0.33, 0.3])  # [left, bottom, width, height]
    #monthly_data = monthly_series.groupby(monthly_series.index.month)
    #df_monthly = pd.DataFrame({month: monthly_data.get_group(month).values for month in monthly_data.groups})
    #df_monthly.boxplot(ax=ax_inset, vert=False)
    #ax_inset.set_yticklabels([pd.to_datetime(f'2022-{i}-01').strftime('%b') for i in range(1, 13)])
    plt.show()
    

# %% --- EIA Validation Plot

#Inputs
#1. CALFEWS Generation in GWh (calfews_gen)
#2. EIA Generation in Gwh (eia_gen)
#3. Reservoir_Name (res_name)

#Output:- 
#1. Plot with both generations and correlation coeff 

def get_eia_validation(calfews_gen, eia_gen, res_name):
    
    #CALFEWS Generation
    monthly_series = calfews_gen.resample('MS').sum()
    
    # Convert indices to DateTime
    monthly_series.index = pd.to_datetime(monthly_series.index)
    eia_gen.index = pd.to_datetime(eia_gen.index)
    
    # Slice the data to the range between 2001 and 2016
    start_date = pd.to_datetime('2001-01-01')
    end_date = pd.to_datetime('2016-09-30')
    monthly_series = monthly_series[(monthly_series.index >= start_date) & (monthly_series.index <= end_date)]
    eia_gen = eia_gen[(eia_gen.index >= start_date) & (eia_gen.index <= end_date)]
    
    #Compute the correlation
    correlation = monthly_series.corr(eia_gen)
    
    #Generate the plot
    plt.figure(figsize=(10, 6))
    monthly_series.plot(label='CALFEWS Gen')
    eia_gen.plot(color='red', label='EIA Gen')
    plt.title(f'Monthly Hydropower Generation at {res_name} (2001-2016)\n'
              f'Pearson Correlation: {correlation:.2f}', fontsize=18)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('GWh', fontsize=14)

    # Improve the formatting of ticks
    plt.xticks(fontsize=12, rotation=45)  # Rotate x-axis ticks for better readability
    plt.yticks(fontsize=12)

    plt.grid(True)
    plt.legend(fontsize=18)  # Add a legend to distinguish the time series
    plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
    plt.show()

# %% --- WAPA Validation Plot

#Inputs
#1. WAPA Generation in GWh (calfews_gen)
#2. WAPA Generation in Gwh (eia_gen)
#3. Reservoir_Name (res_name)

#Output:- 
#1. Scatter plot with fitted regression and R-sq


def get_wapa_validation(calfews_gen, wapa_gen, res_name, weekly_plot = False, monthly_plot = False):
    
    #Load Dependencies
    from scipy.stats import linregress
    
    # Convert Series to DataFrames
    df_wapa_gen = wapa_gen.to_frame(name='wapa_gen')
    df_calfews_gen = calfews_gen.to_frame(name='calfews_gen')

    # Merge the DataFrames on their index
    common_data = pd.merge(df_wapa_gen, df_calfews_gen, left_index=True, right_index=True, how='inner')

    # Plotting the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(common_data['wapa_gen'], common_data['calfews_gen'], color='blue', label='Data Points')

    # Calculating the line of best fit
    slope, intercept, r_value, _, _ = linregress(common_data['wapa_gen'], common_data['calfews_gen'])

    # Plotting the line of best fit
    x = np.linspace(min(common_data['wapa_gen']), max(common_data['wapa_gen']), 100)
    y = slope * x + intercept
    plt.plot(x, y, color='red', label=f'Line of Best Fit: R² = {r_value**2:.2f}')

    # Adding a 45-degree line
    plt.plot([min(x), max(x)], [min(x), max(x)], color='green', linestyle='--', label='45° Line')

    # Adding labels and title
    plt.title(f'Daily Generation - {res_name}', fontsize=18)
    plt.xlabel('WAPA Generation (GWh)', fontsize=14)
    plt.ylabel('CALFEWS Generation (GWh)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()
    
    #Weekly Plot
    if weekly_plot == True:
        #Convert to weekly and get correlation
        weekly_data = common_data.resample('W').sum()
        correlation = weekly_data['calfews_gen'].corr(weekly_data['wapa_gen'])
        
        plt.figure(figsize=(10, 6))
        weekly_data['calfews_gen'].plot(label='CALFEWS Gen')
        weekly_data['wapa_gen'].plot(color='red', label='WAPA Gen')
        plt.title(f'Weekly Hydropower Generation at {res_name} (2015-2016)\n'
                  f'Pearson Correlation: {correlation:.2f}', fontsize=18)
        plt.xlabel('Week', fontsize=14)
        plt.ylabel('GWh', fontsize=14)
        plt.xticks(fontsize=12, rotation=45)  
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.legend(fontsize=18)  
        plt.tight_layout()
        plt.show()
        
        
    #Weekly Plot
    if monthly_plot == True:
        #Convert to weekly and get correlation
        monthly_data = common_data.resample('M').sum()
        correlation = monthly_data['calfews_gen'].corr(monthly_data['wapa_gen'])
        
        plt.figure(figsize=(10, 6))
        monthly_data['calfews_gen'].plot(label='CALFEWS Gen')
        monthly_data['wapa_gen'].plot(color='red', label='WAPA Gen')
        plt.title(f'Monthly Hydropower Generation at {res_name} (2015-2016)\n'
                  f'Pearson Correlation: {correlation:.2f}', fontsize=18)
        plt.xlabel('Month', fontsize=14)
        plt.ylabel('GWh', fontsize=14)
        plt.xticks(fontsize=12, rotation=45)  
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.legend(fontsize=18)  
        plt.tight_layout()
        plt.show()



# %% --- WAPA-EIA Comparision Plot

#Inputs
#1. WAPA Generation in GWh (calfews_gen)
#2. EIA Generation in Gwh (eia_gen)
#3. Reservoir_Name (res_name)

#Output:- 
#1. Plot with both generations and correlation coeff 


def get_wapa_eia_comparision(eia_gen, wapa_gen, res_name):
    
    #WAPA Generation
    monthly_series = wapa_gen.resample('MS').sum()
    
    # Convert indices to DateTime
    monthly_series.index = pd.to_datetime(monthly_series.index)
    eia_gen.index = pd.to_datetime(eia_gen.index)
    
    # Slice the data to the range between 2001 and 2016
    start_date = pd.to_datetime('2015-11-01')
    end_date = pd.to_datetime('2023-09-01')
    monthly_series = monthly_series[(monthly_series.index >= start_date) & (monthly_series.index <= end_date)]
    eia_gen = eia_gen[(eia_gen.index >= start_date) & (eia_gen.index <= end_date)]
    
    
    #Compute the correlation
    correlation = monthly_series.corr(eia_gen)
    
    #Generate the plot
    plt.figure(figsize=(10, 6))
    monthly_series.plot(label='WAPA Gen')
    eia_gen.plot(color='red', label='EIA Gen')
    plt.title(f'Monthly Hydropower Generation at {res_name} (2001-2016)\n'
              f'Pearson Correlation: {correlation:.2f}', fontsize=18)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('GWh', fontsize=14)

    # Improve the formatting of ticks
    plt.xticks(fontsize=12, rotation=45)  # Rotate x-axis ticks for better readability
    plt.yticks(fontsize=12)

    plt.grid(True)
    plt.legend(fontsize=18)  # Add a legend to distinguish the time series
    plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
    plt.show()



# %%
###Shasta Generation. 
def get_shasta_generation(release, storage, discharge_capacity = 9999999):
    
    #Curtail Releases
    release = release.apply(lambda x: min(x, discharge_capacity))
    
    #Convert Releases to Tailwater Elevation
    def get_tailwater_elev_shasta(df):
        return 1 / ((0.0000000000000197908 * ((df * 1000 / 31 / 1.9834711) - 40479.9296) ** 2) + 0.00168)
    tailwater_elevation = get_tailwater_elev_shasta(release)
    
    #Covert Storage to Forebay Elevation 
    def get_forebay_elev_shasta(df):
        return (740.270709 + 
                (0.0002185318721 * (df * 1000)) - 
                (0.0000000001006141253 * (df * 1000) ** 2) + 
                (3.224005059E-17 * (df * 1000) ** 3) - 
                (5.470777842E-24 * (df * 1000) ** 4) + 
                (3.711432277E-31 * (df * 1000) ** 5))
    forebay_elevation = get_forebay_elev_shasta(storage)
    
    #Compute the Gross Head 
    gross_head = forebay_elevation-tailwater_elevation
    
    #Compute the power generation potential (kwh/AF)
    def get_power_per_kwh(df):
        return 1.045 * ((0.83522 * df) + 30.5324)
    kwh_per_AF = get_power_per_kwh(gross_head)
    
    return kwh_per_AF*release/10**3


#Get the Shasta Curtailment Generation 
shasta_capacity = 676 #MW
shasta_discharge_capacity = 34.909 # TAc-ft/day https://web.archive.org/web/20121003060702/http://www.usbr.gov/pmts/hydraulics_lab/pubs/PAP/PAP-0845.pdf


#Compute the Generation  with Discharge Capacity
shasta_gen = get_shasta_generation(getRelease['shasta_R'], getStorage['shasta_S'], shasta_discharge_capacity)
get_calfews_gen_plots(shasta_gen, 'Shasta')
get_eia_validation(shasta_gen, eia['Shasta']/1000, 'Shasta')
get_wapa_validation(shasta_gen, wapa['Shasta']/1000, 'Shasta', weekly_plot=True, monthly_plot=True)
get_wapa_eia_comparision(eia['Shasta']/1000, wapa['Shasta']/1000, 'Shasta')






# %%
###Folsom

def get_folsom_generation(release, storage, discharge_capacity = 9999999):
    
    #Curtail Releases
    release = release.apply(lambda x: min(x, discharge_capacity))
    
    #Convert Releases to Tailwater Elevation
    def get_tailwater_elev_folsom(df):
        result = 10 ** (2.113508 - 0.035579 * np.log10((df * 1000 / 31 / 1.9834711) / 1000)
                        + 0.04750301 * np.log10((df * 1000 / 31 / 1.9834711) / 1000) ** 2)
        return result

    tailwater_elevation = get_tailwater_elev_folsom(release)
    
    #Covert Storage to Forebay Elevation 
    def get_forebay_elev_folsom(df):
        df_scaled = df * 1000
        result = (
            274.0928028 +
            (0.0007776373877 * df_scaled) -
            (0.000000002137196225 * df_scaled**2) +
            (3.44539018E-15 * df_scaled**3) -
            (2.707326223E-21 * df_scaled**4) +
            (8.144330075E-28 * df_scaled**5)
            )
        return result
    forebay_elevation = get_forebay_elev_folsom(storage)
    
    #Compute the Gross Head 
    gross_head = forebay_elevation-tailwater_elevation
    
    #Compute the power generation potential (kwh/AF)
    def get_power_per_kwh(df):
        return 0.92854 * df - 16.282
    kwh_per_AF = get_power_per_kwh(gross_head)
    
    return kwh_per_AF*release/10**3


#Get the Shasta Curtailment Generation 
folsom_capacity = 198.7 #MW
folsom_discharge_capacity = 13.682 # TAc-ft/day https://lowimpacthydro.org/wp-content/uploads/2021/05/Folsom-Nimbus-LIHI-Application-2021-March-5-2021-final.pdf

#Compute the Generation 
folsom_gen = get_folsom_generation(getRelease['folsom_R'], getStorage['folsom_S'], folsom_discharge_capacity)

#Generation and Validation Plots
get_calfews_gen_plots(folsom_gen, 'Folsom')
get_eia_validation(folsom_gen, eia['Folsom']/1000, 'Folsom')
get_wapa_validation(folsom_gen, wapa['Folsom']/1000, 'Folsom', weekly_plot=True, monthly_plot=True)
get_wapa_eia_comparision(eia['Folsom']/1000, wapa['Folsom']/1000, 'Folsom')


# %%
###New Melones

def get_new_melones_generation(release, storage, discharge_capacity = 9999999):
    
    #Curtail Releases
    release = release.apply(lambda x: min(x, discharge_capacity))
    
    def get_tailwater_elev_NM(df):
        adjusted_release = (df * 1000 / 31 / 1.9834711) - 63433.09636
        result = 498 + (1 / (0.000000000064599389 * adjusted_release ** 2 - 0.06036419))
        return result
    tailwater_elevation = get_tailwater_elev_NM(release)
    
    
    def get_forebay_elev_NM(df):
        df_scaled = df * 1000
        result = (679.2519422 +
                  (0.0005645439241 * df_scaled) -
                  (0.0000000005594273841 * df_scaled ** 2) +
                  (3.56253072E-16 * df_scaled ** 3) -
                  (1.129140824E-22 * df_scaled ** 4) +
                  (1.374728535E-29 * df_scaled ** 5))
        return result
    forebay_elevation = get_forebay_elev_NM(storage)
    
    # Compute the Gross Head 
    gross_head = forebay_elevation - tailwater_elevation
    
    # Compute the power generation potential (kwh/AF)
    def get_power_per_kwh(df):
        return 0.62455 * df + 142.3077

    kwh_per_AF = get_power_per_kwh(gross_head)
    
    return kwh_per_AF * release / 10 ** 3

#Get the Shasta Curtailment Generation 
new_melones_capacity = 300 #MW
new_melones_discharge_capacity = 16.450 #TAC-ft/day https://www.waterboards.ca.gov/waterrights/water_issues/programs/hearings/auburn_dam/exhibits/x_8.pdf


#Compute the Generation 
new_melones_gen = get_new_melones_generation(getRelease['newmelones_R'], getStorage['newmelones_S'], new_melones_discharge_capacity)



#Generation and Validation Plots
get_calfews_gen_plots(new_melones_gen, 'New Melones')
get_eia_validation(new_melones_gen, eia['New Melones']/1000, 'New Melones')
get_wapa_validation(new_melones_gen, wapa['New Melones']/1000, 'New Melones', weekly_plot=True, monthly_plot=True)
get_wapa_eia_comparision(eia['New Melones']/1000, wapa['New Melones']/1000, 'New Melones')





# %%
###San Luis


def get_san_luis_generation(release, storage):
    
    #Tailwater elevation for San Luis is fixed
    tailwater_elevation = 220
    
    #Compute forebay elevation
    data = {
    'Storage_TAF': [1, 6	, 16, 33, 59, 94, 137, 187, 243,	305, 373, 446, 523, 605, 691, 781, 875, 972, 1073, 1177,	1284	,1394,1508,1624,	1743,1865,	1990,	2039,	2117],
    'Stage': [280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 543.86, 550 ],
    'Height': [1.9677, 0.9960, 0.5688, 0.3856, 0.2856, 0.2339, 0.1996, 0.1777, 0.1610, 0.1478, 0.1378, 0.1295, 0.1223, 0.1162, 0.1109, 0.1065, 0.1026, 0.0992, 0.0961, 0.0933, 0.0907, 0.0883, 0.0860, 0.0840, 0.0820, 0.0801, 0.0789, 0.0780, 0.2597]
    }
    SanLuisStageStorage = pd.DataFrame(data)
    
    def find_stage(storage_value):
        closest_storage = SanLuisStageStorage.iloc[(SanLuisStageStorage['Storage_TAF'] - storage_value).abs().argsort()[:1]]
        stage = closest_storage['Stage'].values[0] + (storage_value - closest_storage['Storage_TAF'].values[0]) * closest_storage['Height'].values[0]
        return stage
    
    forebay_elevation = storage.apply(find_stage)
    
    # Compute the Gross Head 
    gross_head = forebay_elevation - tailwater_elevation
    
    # Compute the power generation potential (kwh/AF)
    def get_power_per_kwh(df):
        return 0.917325 * df - 11.0233

    kwh_per_AF = get_power_per_kwh(gross_head)
    
    return kwh_per_AF * release / 10 ** 3

#Compute the Generation 
sl_storage = getStorage['sanluisstate_S'] + getStorage['sanluisfederal_S']
sl_release = sl_storage.diff().fillna(0)
#sl_release = sl_release.apply(lambda x: -1*x if x < 0 else 0)
san_luis_gen = get_san_luis_generation(-sl_release, sl_storage)

#Generation and Validation Plots
get_calfews_gen_plots(san_luis_gen, 'San Luis')
get_eia_validation(san_luis_gen, eia['W R Gianelli']/1000, 'San Luis')
#San Luis Capacity is 424. 





# %%
###Keswick

def get_keswick_generation(release, discharge_capacity = 9999999):
    
    #Curtail Releases
    release = release.apply(lambda x: min(x, discharge_capacity))
    

    def get_tailwater_elev_keswick(df):
        term1 = 485.0156566
        term2 = (6.36440004 - np.log(df * 1000 / 31 / 1.9834711))**2 / 447.351268
        result = term1 * np.exp(term2)
        return result
    tailwater_elevation = get_tailwater_elev_keswick(release)
    
    
    forebay_elevation = 583.5
    
    # Compute the Gross Head 
    gross_head = forebay_elevation - tailwater_elevation
    
    # Compute the power generation potential (kwh/AF)
    def get_power_per_kwh(df):
        return 0.70399 * df + 9.4772

    kwh_per_AF = get_power_per_kwh(gross_head)
    
    return kwh_per_AF * release / 10 ** 3

#Get the Shasta Curtailment Generation 
keswick_capacity = 117 #MW
keswick_discharge_capacity = 31.73 # TAc-ft/day https://web.archive.org/web/20121003060702/http://www.usbr.gov/pmts/hydraulics_lab/pubs/PAP/PAP-0845.pdf


#Compute the Generation  with Discharge Capacity
keswick_gen = get_keswick_generation(getRelease['shasta_R'], keswick_discharge_capacity)
get_calfews_gen_plots(keswick_gen, 'Keswick')
get_eia_validation(keswick_gen, eia['Keswick']/1000, 'Keswick')
get_wapa_validation(keswick_gen, wapa['Keswick']/1000, 'Keswick', weekly_plot=True, monthly_plot=True)
get_wapa_eia_comparision(eia['Keswick']/1000, wapa['Keswick']/1000, 'Keswick')


# %%
###Nimbus

def get_nimbus_generation(release, discharge_capacity = 9999999):
    
    #Curtail Releases
    release = release.apply(lambda x: min(x, discharge_capacity))
    
    #Convert Releases to Tailwater Elevation
    def get_tailwater_elev_nimbus(df):
        intermediate_value = df * 1000 / 31 / 1.9834711
        result = 81.48069123 + 0.000553075 * intermediate_value - 402.7422903 / intermediate_value
        return result
    tailwater_elevation = get_tailwater_elev_nimbus(release)
    
    #Covert Storage to Forebay Elevation 
    forebay_elevation = 123
    
    #Compute the Gross Head 
    gross_head = forebay_elevation-tailwater_elevation
    
    #Compute the power generation potential (kwh/AF)
    def get_power_per_kwh(df):
        return 0.11191 * df + 29.8156
    kwh_per_AF = get_power_per_kwh(gross_head)
    
    return kwh_per_AF*release/10**3


#Get the Shasta Curtailment Generation 
nimbus_capacity = 15 #MW
nimbus_discharge_capacity = 10.11 # TAc-ft/day https://lowimpacthydro.org/wp-content/uploads/2021/05/Folsom-Nimbus-LIHI-Application-2021-March-5-2021-final.pdf

#Compute the Generation 
nimbus_gen = get_nimbus_generation(getRelease['folsom_R'], nimbus_discharge_capacity)

#Generation and Validation Plots
get_calfews_gen_plots(nimbus_gen, 'Nimbus')
get_eia_validation(nimbus_gen, eia['Nimbus']/1000, 'Nimbus')
get_wapa_validation(nimbus_gen, wapa['Nimbus']/1000, 'Nimbus', weekly_plot=True, monthly_plot=True)
get_wapa_eia_comparision(eia['Nimbus']/1000, wapa['Nimbus']/1000, 'Nimbus')



# %%
###O'Neill


def get_oneill_generation(release):
    
    #Tailwater elevation for San Luis is fixed
    tailwater_elevation = 172
    
    forebay_elevation = 220
    
    # Compute the Gross Head 
    gross_head = forebay_elevation - tailwater_elevation
    
    # Compute the power generation potential (kwh/AF)
    def get_power_per_kwh(df):
        return df/1.3714

    kwh_per_AF = get_power_per_kwh(gross_head)
    
    return kwh_per_AF * release / 10 ** 3

#Compute the Generation 
sl_storage = getStorage['sanluisstate_S'] + getStorage['sanluisfederal_S']
sl_release = sl_storage.diff().fillna(0)
#sl_release = sl_release.apply(lambda x: -1*x if x < 0 else 0)
oneill_gen = get_oneill_generation(-sl_release)

#Generation and Validation Plots
get_calfews_gen_plots(oneill_gen, 'ONeill')
#get_eia_validation(san_luis_gen, eia['W R Gianelli']/1000, 'San Luis')
