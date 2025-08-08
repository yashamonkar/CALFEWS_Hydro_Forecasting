# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:48:51 2024

@author: amonkar

One file to read CALFEWS Output and validate results
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

# %%

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

#%%

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

#%% Function to compute CVP Hydropower


def get_cvp_hydropower(calfews_output):
    """
    INPUTS
    1. Dictonary (CALFEWS Output)
    
    OUTPUTS
    1. Individual Plants Hydropower
    2. Total CVP Hydropower
    """
    
    #Shasta
    if 'shasta_S' in calfews_output.columns:
        shasta_capacity = 676 #MW
        shasta_discharge_capacity = 34.909 # TAc-ft/day https://web.archive.org/web/20121003060702/http://www.usbr.gov/pmts/hydraulics_lab/pubs/PAP/PAP-0845.pdf
        shasta_gen = get_shasta_generation(calfews_output['shasta_R'], calfews_output['shasta_S'], shasta_discharge_capacity)
        
    #Folsom
    if 'folsom_S' in calfews_output.columns:
        folsom_capacity = 198.7 #MW
        folsom_discharge_capacity = 13.682 # TAc-ft/day https://lowimpacthydro.org/wp-content/uploads/2021/05/Folsom-Nimbus-LIHI-Application-2021-March-5-2021-final.pdf
        folsom_gen = get_folsom_generation(calfews_output['folsom_R'], calfews_output['folsom_S'], folsom_discharge_capacity)
        #Nimbus
        nimbus_capacity = 15 #MW
        nimbus_discharge_capacity = 10.11 # TAc-ft/day https://lowimpacthydro.org/wp-content/uploads/2021/05/Folsom-Nimbus-LIHI-Application-2021-March-5-2021-final.pdf
        nimbus_gen = get_nimbus_generation(calfews_output['folsom_R'], nimbus_discharge_capacity)

    #New Melones
    if 'newmelones_S' in calfews_output.columns:
        new_melones_capacity = 300 #MW
        new_melones_discharge_capacity = 16.450 #TAC-ft/day https://www.waterboards.ca.gov/waterrights/water_issues/programs/hearings/auburn_dam/exhibits/x_8.pdf
        new_melones_gen = get_new_melones_generation(calfews_output['newmelones_R'], calfews_output['newmelones_S'], new_melones_discharge_capacity)
        
    #San Luis
    if 'sanluisstate_S' in calfews_output.columns:
        sl_storage = calfews_output['sanluisstate_S'] + calfews_output['sanluisfederal_S']
        sl_release = sl_storage.diff().fillna(0)
        san_luis_gen = get_san_luis_generation(-sl_release, sl_storage)
        #O'Neil
        oneill_gen = get_oneill_generation(-sl_release)

        
    #Keswick (Has to be updated)
    if 'shasta_S' in calfews_output.columns:
        keswick_capacity = 117 #MW
        keswick_discharge_capacity = 31.73 # TAc-ft/day https://web.archive.org/web/20121003060702/http://www.usbr.gov/pmts/hydraulics_lab/pubs/PAP/PAP-0845.pdf
        keswick_gen = get_keswick_generation(calfews_output['shasta_R'], keswick_discharge_capacity)
        
    #Return the values
    total_cvp_gen = shasta_gen+folsom_gen+nimbus_gen+new_melones_gen + san_luis_gen + oneill_gen + keswick_gen
    cvp_gen = pd.DataFrame({'Shasta':shasta_gen,
                            'Folsom': folsom_gen,
                            'Nimbus': nimbus_gen,
                            'New_Melones': new_melones_gen,
                            'San_Luis':san_luis_gen,
                            'ONeill': oneill_gen,
                            'Keswick':keswick_gen,
                            'CVP_Total':total_cvp_gen})
    
    return cvp_gen

calfews_gen = get_cvp_hydropower(datDaily)


#%% Validation with WAPA plants
#calfews_gen.columns
#wapa.columns

calfews_sub = calfews_gen['Shasta'] + calfews_gen['Folsom'] + calfews_gen['Nimbus'] + calfews_gen['New_Melones'] + calfews_gen['Keswick']
wapa_sub = wapa['Shasta'] + wapa['Folsom'] + wapa['Nimbus'] + wapa['New Melones'] + wapa['Keswick']
wapa_sub = wapa_sub/1000

# Resample both series to monthly frequency and compute mean
calfews_monthly = calfews_sub.resample('M').mean()
wapa_monthly = pd.Series(wapa_sub).resample('M').mean()

# Get common dates
common_dates = calfews_monthly.index.intersection(wapa_monthly.index)

# Filter both series to only include common dates
calfews_common = calfews_monthly[common_dates]
wapa_common = wapa_monthly[common_dates]

# Calculate correlation for common period
correlation = calfews_common.corr(wapa_common)

# Create the plot
plt.figure(figsize=(12, 6))

# Plot both series for common period only
plt.plot(calfews_common.index, calfews_common.values, label='CALFEWS Gen', color='blue')
plt.plot(wapa_common.index, wapa_common.values, label='WAPA Gen', color='red')

# Customize the plot
plt.title(f'Monthly Hydropower Generation\nPearson Correlation: {correlation:.2f}')
plt.xlabel('Month')
plt.ylabel('GWh')
plt.grid(True)
plt.legend()

# Adjust layout and display
plt.tight_layout()
plt.show()




#%% Validation with EIA plants
#calfews_gen.columns
#wapa.columns

calfews_sub = calfews_gen['Shasta']*2.5 + calfews_gen['Folsom'] + calfews_gen['New_Melones'] +  calfews_gen['Keswick'] +  calfews_gen['San_Luis']
eia_monthly = eia['Shasta']*2.5 + eia['Folsom'] + eia['New Melones'] +  eia['Keswick'] +  eia['W R Gianelli'] 
eia_monthly = eia_monthly/1000
eia_monthly.index = pd.to_datetime(eia_monthly.index.astype(str) + '-01') + pd.offsets.MonthEnd(0)

# Resample both series to weekly frequency and compute mean
calfews_monthly = calfews_sub.resample('M').sum()

# Get common dates
common_dates = calfews_monthly.index.intersection(eia_monthly.index)

# Filter both series to only include common dates
calfews_common = calfews_monthly[common_dates]
eia_common = eia_monthly[common_dates]

# Calculate correlation for common period
correlation = calfews_common.corr(eia_common)

# Plot the validation
plt.figure(figsize=(12, 6))
start_period1 = pd.to_datetime('2013-11-01')
end_period1 = pd.to_datetime('2015-12-31')
start_period2 = pd.to_datetime('2021-01-01')
end_period2 = pd.to_datetime('2022-12-31')

# Add yellow background shading with datetime objects
plt.axvspan(start_period1, end_period1, color='brown', alpha=0.25)
plt.axvspan(start_period2, end_period2, color='brown', alpha=0.25)

# Rest of your plotting code
plt.plot(calfews_common.index, calfews_common.values, label='CALFEWS', color='blue')
plt.plot(eia_common.index, eia_common.values, label='EIA (External Validation)', color='red')
plt.text(0.98, 0.1, f'Correlation: {correlation:.2f}', 
         transform=plt.gca().transAxes,  
         horizontalalignment='right',     
         verticalalignment='top',         
         fontsize=18)                     
plt.xlabel('Month', fontsize=20)
plt.ylabel('CVP Hydropower Generation \n GWh', fontsize=20)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.show()