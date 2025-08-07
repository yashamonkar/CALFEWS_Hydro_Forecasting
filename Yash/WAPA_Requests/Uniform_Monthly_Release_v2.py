# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:08:21 2024

@author: amonkar
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:14:37 2024

@author: amonkar

This analyzes the validaity of the Uniform Release Assumption. 

For each reservoir. 
1.The historic monthly storage and releases are converted to monthly generation
2. They are compared against the observed generation (EIA-923)

"""




# %%

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
import seaborn as sns


#EIA-923 Dataframe
eia = pd.read_csv('Yash/EIA/EIA_Monthy_Gen.csv', index_col=0)

#Historic Monthly Releases and Storage
historic = pd.read_csv('Yash/WAPA/Monthly_20240325.csv', index_col=0)

#Historic Water Year Types https://cdec.water.ca.gov/reportapp/javareports?name=WSIHIST
wy_types = pd.read_csv("Yash/WAPA/Water_Year_Types.csv", index_col = 0)


# %% Initial Data Wrangling
historic.index = pd.to_datetime(historic.index)
historic = historic[historic.index > '2000-01-01']
historic = historic.replace({',': ''}, regex=True).apply(pd.to_numeric, errors='coerce')


# %% --- CALFEWS Generation Exploratory Plots

#Inputs
#1. Daily CALFEWS Generation in GWh (calfews_gen)
#2. Reservoir Name (res_name)

#Output:- 
#1. Daily Generation plot
#2. Monthly Generation plot
#3. Histogram of daily generation


def get_calfews_gen_plots(calfews_gen, res_name):
    
    
    #Monthly Generation Plot
    plt.figure(figsize=(10, 6))
    calfews_gen.plot()
    plt.title(f'Monthly Hydropower Generation at {res_name} \n Using Uniform Release Assumption', fontsize = 18)
    plt.xlabel('Date', fontsize = 14)
    plt.ylabel('GWh', fontsize = 14)
    plt.xticks(fontsize=12)  # Increase x-axis tick fontsize
    plt.yticks(fontsize=12)  # Increase y-axis tick fontsize
    plt.grid(True)
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
    end_date = pd.to_datetime('2023-09-30')
    monthly_series = monthly_series[(monthly_series.index >= start_date) & (monthly_series.index <= end_date)]
    eia_gen = eia_gen[(eia_gen.index >= start_date) & (eia_gen.index <= end_date)]
    
    #Compute the correlation
    correlation = monthly_series.corr(eia_gen)
    mean_diff = (monthly_series-eia_gen).mean()
    
    #Generate the plot
    plt.figure(figsize=(10, 6))
    monthly_series.plot(label='Estimated Gen')
    eia_gen.plot(color='red', label='EIA Gen')
    plt.title(f'Monthly Hydropower Generation at {res_name} (2001-2023)\n'
              f'Mean Over-estimation: {mean_diff:.2f} GWh', fontsize=18)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('GWh', fontsize=14)

    # Improve the formatting of ticks
    plt.xticks(fontsize=12, rotation=45)  # Rotate x-axis ticks for better readability
    plt.yticks(fontsize=12)

    plt.grid(True)
    plt.legend(fontsize=18)  # Add a legend to distinguish the time series
    plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
    plt.show()


# %% --- Monthly Bias Plots

#Inputs
#1. Estimated Generation in GWh (est_gen)
#2. EIA Generation in Gwh (eia_gen)
#3. Reservoir_Name (res_name)
#4. Capacity in MW (capacity)

#Output:- 
#1. Plot with the bias for each calendar month. 


def monthly_bias_plot(est_gen, eia_gen, res_name, capacity = 0):
    # Ensure the inputs are Pandas Series
    if not isinstance(est_gen, pd.Series) or not isinstance(eia_gen, pd.Series):
        raise ValueError("Both inputs must be Pandas Series.")
    
    # Convert the index to datetime
    est_gen.index = pd.to_datetime(est_gen.index)
    eia_gen.index = pd.to_datetime(eia_gen.index)
    
    # Slice the data to the range between 2001 and 2023
    start_date = pd.to_datetime('2000-12-31')
    end_date = pd.to_datetime('2023-09-30')
    est_gen = est_gen[(est_gen.index > start_date) & (est_gen.index <= end_date)]
    eia_gen = eia_gen[(eia_gen.index > start_date) & (eia_gen.index <= end_date)]
    
    
    # Extract the month and year from the index
    est_gen_month_year = est_gen.index.strftime('%m-%Y')
    eia_gen_month_year = eia_gen.index.strftime('%m-%Y')
    
    # Calculate the bias
    bias = est_gen.values - eia_gen.values
    
    # Create a DataFrame for bias
    bias_df = pd.DataFrame({'Month-Year': est_gen_month_year, 'Bias': bias})
    bias_df['Month'] = pd.to_datetime(bias_df['Month-Year'], format='%m-%Y').dt.month
    mean_bias = bias_df['Bias'].mean()
    
    # Create a boxplot for each calendar month
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Month', y='Bias', data=bias_df)
    plt.title(f'Monthly Bias (OverEstimation) Boxplot for {res_name}\n'
              f'Mean Over-estimation: {mean_bias:.2f} GWh' , fontsize=20)
    plt.xlabel('Month', fontsize=16)
    plt.ylabel('Bias (GWh)', fontsize=16)
    plt.xticks(ticks=range(0, 12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize = 16)
    plt.yticks(fontsize=16)
    plt.grid(True, which="both", ls="--")
    plt.show()
    
    if (capacity != 0):
        
        bias_percent = pd.DataFrame({'Month-Year': est_gen_month_year, 'Bias': bias})
        bias_percent['Month'] = pd.to_datetime(bias_percent['Month-Year'], format='%m-%Y').dt.month
        bias_percent['Bias'] = 100*bias_percent['Bias']/(capacity*24*30/1000)
        mean_bias_pct = bias_percent['Bias'].mean()
        
        # Create a boxplot for each calendar month
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='Month', y='Bias', data=bias_percent)
        plt.title(f'Monthly Bias (OverEstimation) Boxplot for {res_name}\n'
                  f'Mean Over-estimation: {mean_bias:.2f} GWh ({mean_bias_pct:.2f} %)' , fontsize=20)
        plt.xlabel('Month', fontsize=16)
        plt.ylabel('Bias (%)', fontsize=16)
        plt.xticks(ticks=range(0, 12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize = 16)
        plt.yticks(fontsize=16)
        plt.grid(True, which="both", ls="--")
        plt.show()



# %% --- Water Year Bias Plots

#Inputs
#1. Estimated Generation in GWh (est_gen)
#2. EIA Generation in Gwh (eia_gen)
#3. Water Year Type
#4. Reservoir_Name (res_name)
#5. Capacity in MW (capacity)

#Output:- 
#1. Plot with the bias for each calendar month. 




def wy_bias_plot(est_gen, eia_gen, wy_type, res_name, capacity = 0):
    # Ensure the inputs are Pandas Series
    if not isinstance(est_gen, pd.Series) or not isinstance(eia_gen, pd.Series):
        raise ValueError("Both inputs must be Pandas Series.")
    
    # Convert the index to datetime
    est_gen.index = pd.to_datetime(est_gen.index)
    eia_gen.index = pd.to_datetime(eia_gen.index)
    wy_type.index = pd.to_datetime(wy_type.index)
    
    # Slice the data to the range between 2001 and 2023
    start_date = pd.to_datetime('2000-12-31')
    end_date = pd.to_datetime('2023-09-30')
    est_gen = est_gen[(est_gen.index > start_date) & (est_gen.index <= end_date)]
    eia_gen = eia_gen[(eia_gen.index > start_date) & (eia_gen.index <= end_date)]
    wy_type = wy_type[(wy_type.index > start_date) & (wy_type.index <= end_date)]
    
    
    # Extract the month and year from the index
    est_gen_month_year = est_gen.index.strftime('%m-%Y')
    eia_gen_month_year = eia_gen.index.strftime('%m-%Y')
    
    # Calculate the bias
    bias = est_gen.values - eia_gen.values
    
    # Create a DataFrame for bias
    bias_df = pd.DataFrame({'Month-Year': est_gen_month_year, 'Bias': bias, 'WY_Type': wy_type})
    bias_df['Month'] = pd.to_datetime(bias_df['Month-Year'], format='%m-%Y').dt.month
    mean_bias = bias_df['Bias'].mean()
    
    # Define the logical order for WY_Type
    wy_type_order = ['W', 'AN', 'BN', 'D', 'C']
    
    # Create a boxplot for each calendar month
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Month', y='Bias', hue='WY_Type', data=bias_df, hue_order=wy_type_order)
    plt.title(f'Monthly Bias (OverEstimation) Boxplot for {res_name}\n'
              f'Mean Over-estimation: {mean_bias:.2f} GWh' , fontsize=20)
    plt.xlabel('Month', fontsize=16)
    plt.ylabel('Bias (GWh)', fontsize=16)
    plt.xticks(ticks=range(0, 12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize = 16)
    plt.yticks(fontsize=16)
    plt.legend(title='Water Year Type', fontsize=12, title_fontsize=14)
    plt.grid(True)
    plt.show()
    
    
    # Create a boxplot for each calendar month
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='WY_Type', y='Bias', data=bias_df, order=wy_type_order)
    plt.title(f'Bias (OverEstimation) Boxplot for {res_name}\n'
              f'Mean Over-estimation: {mean_bias:.2f} GWh', fontsize=20)
    plt.xlabel('Water Year Type', fontsize=16)
    plt.ylabel('Bias (GWh)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.show()
    
    if (capacity != 0):
        # Create a DataFrame for bias
        bias_pct = pd.DataFrame({'Month-Year': est_gen_month_year, 'Bias': bias, 'WY_Type': wy_type})
        bias_pct['Month'] = pd.to_datetime(bias_df['Month-Year'], format='%m-%Y').dt.month
        bias_pct['Bias'] = 100*bias_pct['Bias']/(capacity*24*30/1000)
        mean_bias_pct = bias_pct['Bias'].mean()
        
        # Create a boxplot for each calendar month
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='Month', y='Bias', hue='WY_Type', data=bias_pct, hue_order=wy_type_order)
        plt.title(f'Monthly Bias (OverEstimation) Boxplot for {res_name}\n'
                  f'Mean Over-estimation: {mean_bias:.2f} GWh ({mean_bias_pct:.2f} %)' , fontsize=20)
        plt.xlabel('Month', fontsize=16)
        plt.ylabel('Bias (%)', fontsize=16)
        plt.xticks(ticks=range(0, 12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize = 16)
        plt.yticks(fontsize=16)
        plt.legend(title='Water Year Type', fontsize=12, title_fontsize=14)
        plt.grid(True)
        plt.show()
        
        
        # Create a boxplot for each calendar month
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='WY_Type', y='Bias', data=bias_pct, order=wy_type_order)
        plt.title(f'Bias (OverEstimation) Boxplot for {res_name}\n'
                  f'Mean Over-estimation: {mean_bias:.2f} GWh ({mean_bias_pct:.2f} %)', fontsize=20)
        plt.xlabel('Water Year Type', fontsize=16)
        plt.ylabel('Bias (%)', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.show()
        


# %%  Shasta


def get_shasta_generation(release, storage, discharge_capacity = 9999999):
    
    # If discharge_capacity is a scalar
    if isinstance(discharge_capacity, (float, int)):
        release = release.apply(lambda x: min(x, discharge_capacity))
    elif len(discharge_capacity) == len(release):
       release = pd.Series(np.minimum(release, discharge_capacity))
    else:
        raise ValueError("discharge_capacity must be either a scalar or have the same length as release.")

    
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


#Get the releases and storage 
Releases = historic['SHA-QT']*1.983/1000 #TAF 1.983 is the conversion factor from CFS to TAF
Storage = historic['SHA-LS']/1000 #TAF
shasta_discharge_capacity = 34.909 # TAc-ft/day https://web.archive.org/web/20121003060702/http://www.usbr.gov/pmts/hydraulics_lab/pubs/PAP/PAP-0845.pdf


#Compute the Generation  with Discharge Capacity
shasta_gen = get_shasta_generation(Releases, Storage, shasta_discharge_capacity*Releases.index.day)
get_calfews_gen_plots(shasta_gen, 'Shasta')
get_eia_validation(shasta_gen, eia['Shasta']/1000, 'Shasta')
monthly_bias_plot(shasta_gen, eia['Shasta']/1000, 'Shasta', 676)
wy_bias_plot(shasta_gen, eia['Shasta']/1000, wy_types['SAC_Index'], 'Shasta', 676)



# %%
###New Melones

def get_new_melones_generation(release, storage, discharge_capacity = 9999999):
    
    # If discharge_capacity is a scalar
    if isinstance(discharge_capacity, (float, int)):
        release = release.apply(lambda x: min(x, discharge_capacity))
    elif len(discharge_capacity) == len(release):
       release = pd.Series(np.minimum(release, discharge_capacity))
    else:
        raise ValueError("discharge_capacity must be either a scalar or have the same length as release.")

    
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

#Get the releases and storage 
Releases = historic['NML-QT']*1.983/1000 #TAF 1.983 is the conversion factor from CFS to TAF
Storage = historic['NML-LS']/1000 #TAF
new_melones_discharge_capacity = 16.450 #TAC-ft/day https://www.waterboards.ca.gov/waterrights/water_issues/programs/hearings/auburn_dam/exhibits/x_8.pdf


#Compute the Generation 
new_melones_gen = get_new_melones_generation(Releases, Storage, new_melones_discharge_capacity*Releases.index.day)



#Generation and Validation Plots
get_calfews_gen_plots(new_melones_gen, 'New Melones')
get_eia_validation(new_melones_gen, eia['New Melones']/1000, 'New Melones')
monthly_bias_plot(new_melones_gen, eia['New Melones']/1000, 'New Melones', 300)
wy_bias_plot(new_melones_gen, eia['New Melones']/1000, wy_types['SJ_Index'], 'New Melones', 300)



# %%
###Folsom
def get_folsom_generation(release, storage, discharge_capacity = 9999999):
    
    # If discharge_capacity is a scalar
    if isinstance(discharge_capacity, (float, int)):
        release = release.apply(lambda x: min(x, discharge_capacity))
    elif len(discharge_capacity) == len(release):
       release = pd.Series(np.minimum(release, discharge_capacity))
    else:
        raise ValueError("discharge_capacity must be either a scalar or have the same length as release.")

    
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


#Get Release and Storage
Releases = historic['FOL-QT']*1.983/1000
Storage = historic['FOL-LS']/1000 #TAF
folsom_discharge_capacity = 13.682 # TAc-ft/day https://lowimpacthydro.org/wp-content/uploads/2021/05/Folsom-Nimbus-LIHI-Application-2021-March-5-2021-final.pdf


#Compute the Generation 
folsom_gen = get_folsom_generation(Releases, Storage, folsom_discharge_capacity*Releases.index.day)

#Generation and Validation Plots
get_calfews_gen_plots(folsom_gen, 'Folsom')
get_eia_validation(folsom_gen, eia['Folsom']/1000, 'Folsom')
monthly_bias_plot(folsom_gen, eia['Folsom']/1000, 'Folsom', 198)
wy_bias_plot(folsom_gen, eia['Folsom']/1000, wy_types['SAC_Index'], 'Folsom', 198)



# %%
###Keswick

def get_keswick_generation(release, discharge_capacity = 9999999):
    
    # If discharge_capacity is a scalar
    if isinstance(discharge_capacity, (float, int)):
        release = release.apply(lambda x: min(x, discharge_capacity))
    elif len(discharge_capacity) == len(release):
       release = pd.Series(np.minimum(release, discharge_capacity))
    else:
        raise ValueError("discharge_capacity must be either a scalar or have the same length as release.")

    

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

#Get Release and Storage
Releases = historic['KES-QT']*1.983/1000
Storage = historic['KES-LS']/1000 #TAF
keswick_discharge_capacity = 31.73 # TAc-ft/day https://web.archive.org/web/20121003060702/http://www.usbr.gov/pmts/hydraulics_lab/pubs/PAP/PAP-0845.pdf


#Compute the Generation 
keswick_gen = get_keswick_generation(Releases, keswick_discharge_capacity*Releases.index.day)

#Generation and Validation Plots
get_calfews_gen_plots(keswick_gen, 'Keswick')
get_eia_validation(keswick_gen, eia['Keswick']/1000, 'Keswick')
monthly_bias_plot(keswick_gen, eia['Keswick']/1000, 'Keswick', 117)
wy_bias_plot(keswick_gen, eia['Keswick']/1000, wy_types['SAC_Index'], 'Keswick', 117)


# %%
###Trinity

def get_trinity_generation(release, storage, discharge_capacity = 9999999):
    
    # If discharge_capacity is a scalar
    if isinstance(discharge_capacity, (float, int)):
        release = release.apply(lambda x: min(x, discharge_capacity))
    elif len(discharge_capacity) == len(release):
       release = pd.Series(np.minimum(release, discharge_capacity))
    else:
        raise ValueError("discharge_capacity must be either a scalar or have the same length as release.")

    

    tailwater_elevation = 1901.5
    
    
    def get_forebay_elev_Trinity(df):
        df_scaled = df * 1000
        result = (2028.137624 +
                  (0.0005032905066 * df_scaled) -
                  (0.0000000005134105556 * df_scaled ** 2) +
                  (3.281723733E-16 * df_scaled ** 3) -
                  (1.048527299E-22 * df_scaled ** 4) +
                  (1.292215614E-29 * df_scaled ** 5))
        return result
    forebay_elevation = get_forebay_elev_Trinity(storage)
    
    # Compute the Gross Head 
    gross_head = forebay_elevation - tailwater_elevation
    
    # Compute the power generation potential (kwh/AF)
    def get_power_per_kwh(df):
        return 1.19285 * df - 142.1086

    kwh_per_AF = get_power_per_kwh(gross_head)
    
    return kwh_per_AF * release / 10 ** 3



#Get Release and Storage
Releases = historic['TRN-QT']*1.983/1000
Storage = historic['TRN-LS']/1000 #TAF
trinity_discharge_capacity = 7.32 # TAc-ft/day https://web.archive.org/web/20121003060702/http://www.usbr.gov/pmts/hydraulics_lab/pubs/PAP/PAP-0845.pdf


#Compute the Generation 
trinity_gen = get_trinity_generation(Releases, Storage, trinity_discharge_capacity*Releases.index.day)

#Generation and Validation Plots
get_calfews_gen_plots(trinity_gen, 'Trinity')
get_eia_validation(trinity_gen, eia['Trinity']/1000, 'Trinity')
monthly_bias_plot(trinity_gen, eia['Trinity']/1000, 'Trinity', 140)
wy_bias_plot(trinity_gen, eia['Trinity']/1000, wy_types['SAC_Index'], 'Trinity', 140)





# %%
###Comparitive Bias for Shasta, Folsom, Trinity and New Melones
#The bias is in terms of Days (GWhr/MWh) 


#Inputs
#1. Pandas Dataframe -- Estimated Generation in GWh (est_gen)
#2. Pandas Dataframe -- EIA Generation in Gwh (eia_gen)
#3. Pandas Dataframe -- Water Year Type
#4. Dictonary -- Capacities -- 
#Reservoir Names (res_name) not provided since they are taken from the column names

#Output:- 
#1. Plot with the bias for each calendar month. 


def comp_bias_plot(est_gen, eia_gen, wy_type, capacities):
    
    
    # Convert the index to datetime
    est_gen.index = pd.to_datetime(est_gen.index)
    eia_gen.index = pd.to_datetime(eia_gen.index)
    wy_type.index = pd.to_datetime(wy_type.index)
    
    
    # Slice the data to the range between 2001 and 2023
    start_date = pd.to_datetime('2000-12-31')
    end_date = pd.to_datetime('2023-09-30')
    est_gen = est_gen[(est_gen.index > start_date) & (est_gen.index <= end_date)]
    eia_gen = eia_gen[(eia_gen.index > start_date) & (eia_gen.index <= end_date)]
    wy_type = wy_type[(wy_type.index > start_date) & (wy_type.index <= end_date)]
    
    #Convert to Month-Year
    est_gen.index = est_gen.index.strftime('%m-%Y')
    eia_gen.index = eia_gen.index.strftime('%m-%Y')
    wy_type.index = wy_type.index.strftime('%m-%Y')
    
    # Calculate the bias
    bias = est_gen - eia_gen
    scaled_bias = 100*1000*bias/(pd.Series(capacities)*24*30) #HOURS
    
    #Combine to a single dataset
    bias_melted = bias.reset_index().melt(id_vars=['OBSERV_DATE'], var_name='Basin', value_name='Value')
    scaled_bias_melted = scaled_bias.reset_index().melt(id_vars=['OBSERV_DATE'], var_name='Basin', value_name='Scaled_Value')
    wy_type_melted = wy_type.reset_index().melt(id_vars=['Month'], var_name='Basin', value_name='Water Year Type')
    wy_type_melted = wy_type_melted.rename(columns={'Month': 'OBSERV_DATE'})
    combined_df = pd.merge(bias_melted, wy_type_melted, on=['OBSERV_DATE', 'Basin'])
    combined_df = pd.merge(combined_df, scaled_bias_melted, on = ['OBSERV_DATE', 'Basin'])
    combined_df = combined_df.set_index('OBSERV_DATE')


    # Create a DataFrame for bias
    bias_df = pd.DataFrame({'Month-Year': combined_df.index, 
                            'Basin': combined_df['Basin'],
                            'Bias': combined_df['Value'], 
                            'Scaled_Bias': combined_df['Scaled_Value'], 
                            'WY_Type': combined_df['Water Year Type']})
    bias_df['Month'] = pd.to_datetime(bias_df['Month-Year'], format='%m-%Y').dt.month
    
    
    #---------Plot 1 -- Bias by Month and Water Year Type---------------------#
    basins = bias_df['Basin'].unique()
    wy_type_order = ['W', 'AN', 'BN', 'D', 'C']

    fig, axes = plt.subplots(2, 2, figsize=(18, 14), sharey=True)

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    for i, basin in enumerate(basins):
        ax = axes[i]
        basin_df = bias_df[bias_df['Basin'] == basin]
        
        sns.boxplot(x='Month', y='Scaled_Bias', hue='WY_Type', 
                    data=basin_df, hue_order=wy_type_order, ax=ax)
        
        mean_bias = basin_df['Bias'].mean()
        mean_scaled_bias = basin_df['Scaled_Bias'].mean()
        ax.set_title(f'\n{basin}\nMean Bias: {mean_bias:.1f} GWh ({mean_scaled_bias:.1f} %)', fontsize=24)
        ax.set_xlabel('Month', fontsize=20)
        ax.set_ylabel('Bias (%)', fontsize=20)
        ax.set_xticks(range(12))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 
                            'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 
                           fontsize=18)
        ax.legend(title='Water Year Type', fontsize=18, title_fontsize=20)
        ax.grid(True)
        ax.tick_params(axis='y', labelsize=18)  # Increase y-axis text size

    plt.tight_layout()
    plt.show()
    
    ####-------------PLOT 2---------------------------------------------------#
    #Bias by WATER YEAR TYPE
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), sharey=True)
    axes = axes.flatten()

    for i, basin in enumerate(basins):
        ax = axes[i]
        basin_df = bias_df[bias_df['Basin'] == basin]
        
        sns.boxplot(x='WY_Type', y='Scaled_Bias', data=basin_df, order=wy_type_order, ax=ax)
        
        mean_bias = basin_df['Bias'].mean()
        mean_scaled_bias = basin_df['Scaled_Bias'].mean()
        ax.set_title(f'\n{basin}\nMean Bias: {mean_bias:.1f} GWh ({mean_scaled_bias:.1f} %)', fontsize=24)
        ax.set_xlabel('Water Year Type', fontsize=20)
        ax.set_ylabel('Bias (%)', fontsize=20)
        ax.grid(True)
        ax.tick_params(axis='y', labelsize=18)  
        ax.tick_params(axis='x', labelsize=18)  

    plt.tight_layout()
    plt.show()
    
    
    ####-------------PLOT 3---------------------------------------------------#
    #Bias by WATER YEAR TYPE
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), sharey=True)
    axes = axes.flatten()

    for i, basin in enumerate(basins):
        ax = axes[i]
        basin_df = bias_df[bias_df['Basin'] == basin]
        
        sns.boxplot(x='Month', y='Scaled_Bias', data=basin_df, ax=ax)
        
        mean_bias = basin_df['Bias'].mean()
        mean_scaled_bias = basin_df['Scaled_Bias'].mean()
        ax.set_title(f'\n{basin}\nMean Bias: {mean_bias:.1f} GWh ({mean_scaled_bias:.1f} %)', fontsize=24)
        ax.set_xlabel('Month', fontsize=20)
        ax.set_ylabel('Bias (%)', fontsize=20)
        ax.set_xticks(range(12))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 
                            'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 
                           fontsize=18)
        ax.grid(True)
        ax.tick_params(axis='y', labelsize=18)  
        ax.tick_params(axis='x', labelsize=18)  

    plt.tight_layout()
    plt.show()
    

####Create the variables
est_gen = pd.DataFrame({'Shasta': shasta_gen,
                        'Folsom': folsom_gen,
                        'Trinity': trinity_gen,
                        'New Melones': new_melones_gen})


eia_gen = pd.DataFrame({'Shasta': eia['Shasta']/1000,
                        'Folsom': eia['Folsom']/1000,
                        'Trinity': eia['Trinity']/1000,
                        'New Melones': eia['New Melones']/1000})


wy_type = pd.DataFrame({'Shasta': wy_types['SAC_Index'],
                        'Folsom': wy_types['SAC_Index'],
                        'Trinity': wy_types['SAC_Index'],
                        'New Melones': wy_types['SJ_Index']})


capacities = {'Shasta': 676,
              'Folsom': 198,
              'Trinity': 140,
              'New Melones': 300}


comp_bias_plot(est_gen, eia_gen, wy_type, capacities)





# %%
####FOLSOM Monthly and Folsom Daily. 
#Note:- The time periods are different. 

#Read the daily data -- FOLSOM OUTFLOW
Folsom_Outflow = pd.read_excel('Yash/Misc_Data/Folsom_Outflow.xlsx')
Folsom_Outflow = Folsom_Outflow[['OBS DATE', 'VALUE']]
Folsom_Outflow['VALUE'] = pd.to_numeric(Folsom_Outflow['VALUE'].str.replace(',', ''))
Folsom_Outflow['OBS DATE'] = pd.to_datetime(Folsom_Outflow['OBS DATE'])
Folsom_Outflow.set_index('OBS DATE', inplace=True)


#Read the daily data -- SHASTA OUTFLOW
Shasta_Outflow = pd.read_excel('Yash/Misc_Data/Shasta_Outflow.xlsx')
Shasta_Outflow = Shasta_Outflow[['OBS DATE', 'VALUE']]
Shasta_Outflow['VALUE'] = pd.to_numeric(Shasta_Outflow['VALUE'].str.replace(',', ''))
Shasta_Outflow['OBS DATE'] = pd.to_datetime(Shasta_Outflow['OBS DATE'])
Shasta_Outflow.set_index('OBS DATE', inplace=True)



#Read the daily data -- FOLSOM Storage
Folsom_Storage = pd.read_excel('Yash/Misc_Data/Folsom_Storage.xlsx')
Folsom_Storage = Folsom_Storage[['OBS DATE', 'VALUE']]
Folsom_Storage['VALUE'] = pd.to_numeric(Folsom_Storage['VALUE'].str.replace(',', ''))
Folsom_Storage['OBS DATE'] = pd.to_datetime(Folsom_Storage['OBS DATE'])
Folsom_Storage.set_index('OBS DATE', inplace=True)



#Read the daily data -- SHASTA Storage
Shasta_Storage = pd.read_excel('Yash/Misc_Data/Shasta_Storage.xlsx')
Shasta_Storage = Shasta_Storage[['OBS DATE', 'VALUE']]
Shasta_Storage['VALUE'] = pd.to_numeric(Shasta_Storage['VALUE'].str.replace(',', ''))
Shasta_Storage['OBS DATE'] = pd.to_datetime(Shasta_Storage['OBS DATE'])
Shasta_Storage.set_index('OBS DATE', inplace=True)


#Compute the folsom daily gen
folsom_daily_gen = get_folsom_generation(Folsom_Outflow['VALUE']*1.983/1000, 
                                         Folsom_Storage['VALUE']/1000, 
                                         folsom_discharge_capacity)

#Compute the shasta daily gen
shasta_daily_gen = get_shasta_generation(Shasta_Outflow['VALUE']*1.983/1000, 
                                         Shasta_Storage['VALUE']/1000, 
                                         shasta_discharge_capacity)






def aggregate_and_merge(gen, daily_gen, eia_gen, WY):
    # Aggregate daily data to monthly data
    daily_gen_monthly = daily_gen.resample('M').sum()

    # Adjust EIA data to the end of the month
    eia_gen.index = pd.to_datetime(eia_gen.index)
    eia_gen.index = eia_gen.index + pd.offsets.MonthEnd(0)
    
    #Adjust the Water Year Types to end of the month.
    WY.index = pd.to_datetime(WY.index)
    WY.index = WY.index + pd.offsets.MonthEnd(0)

    # Find the overlapping dates
    overlapping_dates = gen.index.intersection(daily_gen_monthly.index).intersection(eia_gen.index).intersection(WY.index)

    # Filter all series to include only the overlapping dates
    gen_filtered = gen.loc[overlapping_dates]
    daily_gen_filtered = daily_gen_monthly.loc[overlapping_dates]
    eia_gen_filtered = eia_gen.loc[overlapping_dates]
    wy_type = WY.loc[overlapping_dates]

    # Create a DataFrame from the series
    agg_df = pd.DataFrame({
        'Monthly_gen': gen_filtered,
        'Daily_gen': daily_gen_filtered,
        'Eia_gen': eia_gen_filtered,
        "WY_type":wy_type
    })
    
    #Create a Bias Dataset
    bias_df = pd.DataFrame({
        'Monthly_Assump': agg_df['Monthly_gen'] - agg_df['Eia_gen'],
        'Daily_Assump': agg_df['Daily_gen'] - agg_df['Eia_gen'],
        "WY_type":agg_df['WY_type']
    })
    
    # Set the plotting style
    sns.set(style="whitegrid")

    # Set the water year type order
    wy_type_order = ['W', 'AN', 'BN', 'D', 'C']

    # Compute the mean of both columns
    mean_monthly_assump = bias_df['Monthly_Assump'].mean()
    mean_daily_assump = bias_df['Daily_Assump'].mean()

    # Create the 2x1 plot layout
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot Monthly_Assump
    sns.boxplot(data=bias_df.reset_index(), x=bias_df.index.month, y='Monthly_Assump', hue='WY_type', ax=axes[0], hue_order=wy_type_order)
    axes[0].set_title(f'Perfect Monthly Forecast - Bias: {mean_monthly_assump:.2f} GWh', fontsize=22)
    axes[0].set_xlabel('Month', fontsize=18)
    axes[0].set_ylabel('Bias (GWh)', fontsize=18)
    axes[0].grid(True)

    # Plot Daily_Assump
    sns.boxplot(data=bias_df.reset_index(), x=bias_df.index.month, y='Daily_Assump', hue='WY_type', ax=axes[1], hue_order=wy_type_order)
    axes[1].set_title(f'Perfect Daily Forecast - Bias: {mean_daily_assump:.2f} GWh', fontsize=22)
    axes[1].set_xlabel('Month', fontsize=18)
    axes[1].set_ylabel('Bias (GWh)', fontsize=18)
    axes[1].grid(True)

    # Set the font size of tick labels
    plt.setp(axes[0].get_xticklabels(), fontsize=18)
    plt.setp(axes[0].get_yticklabels(), fontsize=18)
    plt.setp(axes[1].get_xticklabels(), fontsize=18)
    plt.setp(axes[1].get_yticklabels(), fontsize=18)

    # Make sure the y-axis is the same for both plots
    y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(y_min, y_max)
    axes[1].set_ylim(y_min, y_max)

    # Remove individual legends
    axes[0].get_legend().remove()
    axes[1].get_legend().remove()

    # Create a single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Water Year Type', bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=5, fontsize=18, title_fontsize=18)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Show the plot
    plt.show()


aggregate_and_merge(folsom_gen, folsom_daily_gen, eia['Folsom']/1000, wy_types['SAC_Index'])
aggregate_and_merge(shasta_gen, shasta_daily_gen, eia['Shasta']/1000, wy_types['SAC_Index'])




# %% ANALYSIS USING THE SAME DATA SET FOR FOLSOM AND SHASTA

'''
Function takes in daily reservoir releases and daily reservoir storages and
computes the LT GEN bias for Uniform Monthly Release and Uniform Daily Release
with perfect forecasts

INPUTS
1. Daily Reservoir Release in cfs (pandas.core.series.Series with index as date) -- release
2. Daily Reservoir Storage in AF (pandas.core.series.Series with index as date) -- storage
3. Daily Penstock Capacity in AF (penstock_capacity)
4. Hydropower Generation Module (Function itself) - hydro_module
5. Historic Water Year Types
6. EIA Generation Data
7. Reservoir Name (res_name)
8. Capacity (capacity) - Plant Capacity in MWhr
'''



def compare_bias_timestep(release, storage, hydro_module, penstock_capacity, WY, 
                          eia_gen, res_name, capacity):
    
    #Hyper-Parameters
    cfs_to_af = 1.983
    
    # Adjust EIA data to the end of the month
    eia_gen.index = pd.to_datetime(eia_gen.index)
    eia_gen.index = eia_gen.index + pd.offsets.MonthEnd(0)
    
    #Adjust the Water Year Types to end of the month.
    WY.index = pd.to_datetime(WY.index)
    WY.index = WY.index + pd.offsets.MonthEnd(0)
    
    #_________________________________________________________________________#
    #---------------------DAILY TIMESTEP--------------------------------------#
    #Compute the daily hydropower generation
    daily_gen = hydro_module(release*cfs_to_af/1000,
                             storage/1000,
                             penstock_capacity)
    
    # Aggregate daily to monthly hydropower gen
    daily_gen_monthly = daily_gen.resample('M').sum()
    
    #_________________________________________________________________________#
    #---------------------MONTHLY TIMESTEP------------------------------------#
    monthly_release = release.resample('M').sum()
    monthly_storage = storage.resample('M').mean()
    days_in_month = monthly_release.index.day
    
    #Compute the daily hydropower generation
    monthly_gen = hydro_module(monthly_release*cfs_to_af/1000,
                               monthly_storage/1000,
                               penstock_capacity*days_in_month)
    # Find the overlapping dates
    overlapping_dates = monthly_gen.index.intersection(daily_gen_monthly.index).intersection(eia_gen.index).intersection(WY.index)

    # Filter all series to include only the overlapping dates
    monthly_gen_filtered = monthly_gen.loc[overlapping_dates]
    daily_gen_filtered = daily_gen_monthly.loc[overlapping_dates]
    eia_gen_filtered = eia_gen.loc[overlapping_dates]
    wy_type = WY.loc[overlapping_dates]

    # Create a DataFrame from the series
    agg_df = pd.DataFrame({
        'Monthly_gen': monthly_gen_filtered,
        'Daily_gen': daily_gen_filtered,
        'Eia_gen': eia_gen_filtered,
        "WY_type":wy_type
    })
    
    
    #Create the generation plot
    plt.figure(figsize=(10, 6))
    agg_df['Monthly_gen'].plot(color='black', label='Monthly Forecast')
    agg_df['Daily_gen'].plot(color='blue', label='Daily Forecast', alpha=0)
    agg_df['Eia_gen'].plot(color='red', label='EIA Gen', alpha=0)
    plt.title(f'{res_name}', fontsize=18)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('GWh', fontsize=14)
    plt.xticks(fontsize=12, rotation=45)  
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12, bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3, title='Legend', title_fontsize='13')
    plt.tight_layout()
    plt.show()
    
    #Create the generation plot
    plt.figure(figsize=(10, 6))
    agg_df['Monthly_gen'].plot(color='black', label='Monthly Forecast')
    agg_df['Daily_gen'].plot(color='blue', label='Daily Forecast')
    agg_df['Eia_gen'].plot(color='red', label='EIA Gen', alpha=0)
    plt.title(f'{res_name}', fontsize=18)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('GWh', fontsize=14)
    plt.xticks(fontsize=12, rotation=45)  
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12, bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3, title='Legend', title_fontsize='13')
    plt.tight_layout()
    plt.show()
    
    #Create the generation plot
    plt.figure(figsize=(10, 6))
    agg_df['Monthly_gen'].plot(color='black', label='Monthly Forecast')
    agg_df['Daily_gen'].plot(color='blue', label='Daily Forecast')
    agg_df['Eia_gen'].plot(color='red', label='EIA Gen')
    plt.title(f'{res_name}', fontsize=18)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('GWh', fontsize=14)
    plt.xticks(fontsize=12, rotation=45)  
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12, bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3, title='Legend', title_fontsize='13')
    plt.tight_layout()
    plt.show()
    
    #Create a Bias Dataset
    bias_df = pd.DataFrame({
        'Monthly_Assump': 100*(agg_df['Monthly_gen'] - agg_df['Eia_gen'])/(capacity*24*31/1000),
        'Daily_Assump': 100*(agg_df['Daily_gen'] - agg_df['Eia_gen'])/(capacity*24*31/1000),
        "WY_type":agg_df['WY_type']
    })
    
    # Set the plotting style
    sns.set(style="whitegrid")

    # Set the water year type order
    wy_type_order = ['W', 'AN', 'BN', 'D', 'C']

    # Compute the mean of both columns
    mean_monthly_assump = bias_df['Monthly_Assump'].mean()
    mean_daily_assump = bias_df['Daily_Assump'].mean()

    # Create the 2x1 plot layout
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot Monthly_Assump
    sns.boxplot(data=bias_df.reset_index(), x=bias_df.index.month, y='Monthly_Assump', hue='WY_type', ax=axes[0], hue_order=wy_type_order)
    axes[0].set_title(f'Perfect Monthly Forecast - Bias: {mean_monthly_assump:.2f} %', fontsize=22)
    axes[0].set_xlabel('Month', fontsize=18)
    axes[0].set_ylabel('Bias (%)', fontsize=18)
    axes[0].grid(True)

    # Plot Daily_Assump
    sns.boxplot(data=bias_df.reset_index(), x=bias_df.index.month, y='Daily_Assump', hue='WY_type', ax=axes[1], hue_order=wy_type_order)
    axes[1].set_title(f'Perfect Daily Forecast - Bias: {mean_daily_assump:.2f} %', fontsize=22)
    axes[1].set_xlabel('Month', fontsize=18)
    axes[1].set_ylabel('Bias (%)', fontsize=18)
    axes[1].grid(True)

    # Set the font size of tick labels
    plt.setp(axes[0].get_xticklabels(), fontsize=18)
    plt.setp(axes[0].get_yticklabels(), fontsize=18)
    plt.setp(axes[1].get_xticklabels(), fontsize=18)
    plt.setp(axes[1].get_yticklabels(), fontsize=18)

    # Make sure the y-axis is the same for both plots
    y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(y_min, y_max)
    axes[1].set_ylim(y_min, y_max)

    # Remove individual legends
    axes[0].get_legend().remove()
    axes[1].get_legend().remove()

    # Create a single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Water Year Type', bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=5, fontsize=18, title_fontsize=18)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Show the plot
    plt.show()


compare_bias_timestep(Folsom_Outflow['VALUE'], 
                      Folsom_Storage['VALUE'],
                      get_folsom_generation,
                      folsom_discharge_capacity,
                      wy_types['SAC_Index'],
                      eia['Folsom']/1000, 
                      'Folsom', 
                      198)


compare_bias_timestep(Shasta_Outflow['VALUE'], 
                      Shasta_Storage['VALUE'],
                      get_shasta_generation,
                      shasta_discharge_capacity,
                      wy_types['SAC_Index'],
                      eia['Shasta']/1000,
                      'Shasta', 
                      676)







# %% ANALYSIS OF BIAS IN FLOOD SPILLS# -- ADDED 7/24

'''
Function takes in daily reservoir releases and just computes the spills on the
daily and monthly timestep over the penstock capacity. 


INPUTS
1. Daily Reservoir Release in cfs (pandas.core.series.Series with index as date) -- release
2. Daily Penstock Capacity in TAF (penstock_capacity)
3. Historic Water Year Types
4. Reservoir Name
'''


def compare_bias_flow(release, penstock_capacity, WY, res_name):
    
    #Hyper-Parameters
    cfs_to_af = 1.983
    
    #Adjust the Water Year Types to end of the month.
    WY.index = pd.to_datetime(WY.index)
    WY.index = WY.index + pd.offsets.MonthEnd(0)
    
    #Convert releases to TAF.
    release = release*cfs_to_af/1000
    
    #_________________________________________________________________________#
    #---------------------DAILY TIMESTEP--------------------------------------#
    #Compute the daily spill
    daily_spill = release.apply(lambda x: max(x - penstock_capacity, 0))
    
    # Aggregate daily to monthly hydropower gen
    daily_spill_agg = daily_spill.resample('M').sum()
    
    #_________________________________________________________________________#
    #---------------------MONTHLY TIMESTEP------------------------------------#
    monthly_release = release.resample('M').sum()
    
    #Compute the daily hydropower generation
    days_in_month = monthly_release.index.day
    monthly_spill = monthly_release - days_in_month*penstock_capacity
    monthly_spill = monthly_spill.apply(lambda x: max(x, 0))
    
    # Find the overlapping dates
    overlapping_dates = monthly_spill.index.intersection(daily_spill_agg.index).intersection(WY.index)

    # Filter all series to include only the overlapping dates
    monthly_spill_filtered = monthly_spill.loc[overlapping_dates]
    daily_spill_filtered = daily_spill_agg.loc[overlapping_dates]
    wy_type = WY.loc[overlapping_dates]
    Diff = daily_spill_filtered - monthly_spill_filtered

    # Create a DataFrame from the series
    agg_df = pd.DataFrame({
        'Monthly_spill': monthly_spill_filtered,
        'Daily_spill': daily_spill_filtered,
        "WY_type": wy_type,
        "Diff": Diff
    })
    
    
    # Set the plotting style
    sns.set(style="whitegrid")

    # Set the water year type order
    wy_type_order = ['W', 'AN', 'BN', 'D', 'C']

    # Compute the mean of both columns
    mean_monthly_assump = agg_df['Monthly_spill'].mean()    
    mean_daily_assump = agg_df['Daily_spill'].mean()
    mean_diff = agg_df['Diff'].mean()

    # Create the 3x1 plot layout
    fig, axes = plt.subplots(3, 1, figsize=(15, 11.25), sharex=True)

    # Plot Monthly_Assump
    sns.boxplot(data=agg_df.reset_index(), x=agg_df.index.month, y='Monthly_spill', hue='WY_type', ax=axes[0], hue_order=wy_type_order)
    axes[0].set_title(f'{res_name} - Perfect Monthly Forecast - Mean Spill: {mean_monthly_assump:.2f} TAF', fontsize=22)
    axes[0].set_xlabel('Month', fontsize=18)
    axes[0].set_ylabel('Spill (TAF)', fontsize=18)
    axes[0].grid(True)

    # Plot Daily_Assump
    sns.boxplot(data=agg_df.reset_index(), x=agg_df.index.month, y='Daily_spill', hue='WY_type', ax=axes[1], hue_order=wy_type_order)
    axes[1].set_title(f'Perfect Daily Forecast - Mean Spill: {mean_daily_assump:.2f} TAF', fontsize=22)
    axes[1].set_xlabel('Month', fontsize=18)
    axes[1].set_ylabel('Spill (TAF)', fontsize=18)
    axes[1].grid(True)
    
    # Plot Difference
    sns.boxplot(data=agg_df.reset_index(), x=agg_df.index.month, y='Diff', hue='WY_type', ax=axes[2], hue_order=wy_type_order)
    axes[2].set_title(f'Difference - Mean Spill: {mean_diff:.2f} TAF', fontsize=22)
    axes[2].set_xlabel('Month', fontsize=18)
    axes[2].set_ylabel('Spill (TAF)', fontsize=18)
    axes[2].grid(True)
    
    # Set the font size of tick labels
    for ax in axes:
        plt.setp(ax.get_xticklabels(), fontsize=18)
        plt.setp(ax.get_yticklabels(), fontsize=18)
        
        # Make sure the y-axis is the same for all plots if needed
    y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0], axes[2].get_ylim()[0])
    y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1], axes[2].get_ylim()[1])
    for ax in axes:
        ax.set_ylim(y_min, y_max)
            
    # Remove individual legends
    for ax in axes:
        ax.get_legend().remove()
                
    # Create a single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Water Year Type', bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=5, fontsize=18, title_fontsize=18)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Show the plot
    plt.show()


compare_bias_flow(Folsom_Outflow['VALUE'],
                  folsom_discharge_capacity,
                  wy_types['SAC_Index'],
                  'Folsom')


compare_bias_flow(Shasta_Outflow['VALUE'],
                  shasta_discharge_capacity,
                  wy_types['SAC_Index'],
                  'Shasta')

