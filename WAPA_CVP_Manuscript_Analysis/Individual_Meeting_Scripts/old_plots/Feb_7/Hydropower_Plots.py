# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 16:31:57 2025

@author: amonkar
"""

#%% CVP Hydropower generation plots for the Feb-7th meeting. 
#Run this after running the Validation_Plots.py file

from scipy import stats
from numpy import mean

#### Start with monthly generation 
cvp_gen_daily = calfews_cvp_gen[['Shasta', 'Trinity', 'Carr', 'Spring_Creek', \
                                 'Keswick', 'Folsom', 'Nimbus', 'New_Melones']]
#cvp_gen_daily = cvp_gen_daily[cvp_gen_daily.index > pd.Timestamp('2002-12-31')]
cvp_gen_daily['CVP_Gen'] = cvp_gen_daily.sum(axis=1)


#Subset to necessary plots
cvp_gen_plants = calfews_cvp_gen_monthly[['Shasta', 'Trinity', 'Carr', 'Spring_Creek', \
                                          'Keswick', 'Folsom', 'Nimbus', 'New_Melones']]
cvp_gen_plants['CVP_Gen'] = cvp_gen_plants.sum(axis=1)

eia_gen_plants = eia_monthly[['Shasta', 'Trinity', 'Carr', 'Spring_Creek', \
                              'Keswick', 'Folsom', 'Nimbus', 'New_Melones']]
eia_gen_plants['CVP_Gen'] = eia_gen_plants.sum(axis=1)
eia_gen_plants.index = pd.to_datetime(eia_gen_plants.index).to_period('M').to_timestamp('M')



#%% Daily time step plots
#Start with the daily simulation. 
input_subset = cvp_gen_daily[cvp_gen_daily.index < pd.Timestamp("1996-09-30")]

plt.figure(figsize = (15, 8))
plt.plot(cvp_gen_daily.index, cvp_gen_daily['CVP_Gen'], 'r-',  linewidth=0)
plt.plot(input_subset.index, input_subset['CVP_Gen'], 'r-', label = "CALFEWS Generation",  linewidth=2)
plt.ylabel("Daily Generation (GWh)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title("CVP Daily Generation \n ", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.tight_layout()



plt.figure(figsize = (15, 8))
plt.plot(cvp_gen_daily.index, cvp_gen_daily['CVP_Gen'], 'r-', label = "CALFEWS Generation",  linewidth=2)
plt.ylabel("Daily Generation (GWh)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title("CVP Daily Generation \n ", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.tight_layout()


# %% Monthly Time Step

plt.figure(figsize = (15, 8))
plt.plot(eia_gen_plants.index, eia_gen_plants['CVP_Gen'], 'b-', label = "EIA Generation",  linewidth=0)
plt.plot(cvp_gen_plants.index, cvp_gen_plants['CVP_Gen'], 'r-', label = "CALFEWS Generation", linewidth = 2)
plt.ylabel("Monthly Generation (GWh)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title("CVP Monthly Generation \n ", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.tight_layout()


plt.figure(figsize = (15, 8))
plt.plot(eia_gen_plants.index, eia_gen_plants['CVP_Gen'], 'b-', label = "EIA Generation",  linewidth=2)
plt.plot(cvp_gen_plants.index, cvp_gen_plants['CVP_Gen'], 'r-', label = "CALFEWS Generation", linewidth = 0)
plt.ylabel("Monthly Generation (GWh)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title("CVP Monthly Generation \n ", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.tight_layout()


#Compute the correlation between EIA and CALFEWS total gen
correlation = stats.pearsonr(eia_gen_plants['CVP_Gen'], 
                           cvp_gen_plants['CVP_Gen'])[0]

mean_error = (-eia_gen_plants['CVP_Gen'] + cvp_gen_plants['CVP_Gen']).mean()

plt.figure(figsize = (15, 8))
plt.plot(eia_gen_plants.index, eia_gen_plants['CVP_Gen'], 'b-', label = "EIA Generation",  linewidth=2)
plt.plot(cvp_gen_plants.index, cvp_gen_plants['CVP_Gen'], 'r-', label = "CALFEWS Generation", linewidth = 2)
plt.ylabel("Monthly Generation (GWh)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title(f"CVP Monthly Generation \n R = {round(correlation, 2)} & Average Error = {round(mean_error, 2)} GWh", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.tight_layout()

# %% San luis


###Shasta Monthly plot
correlation = stats.pearsonr(eia_monthly['San_Luis'],
                          calfews_cvp_gen_monthly['San_Luis'])[0]



plt.figure(figsize = (15, 8))
plt.plot(eia_monthly.index, eia_monthly['San_Luis'], 'b-', label = "EIA Generation",  linewidth=2)
plt.plot(calfews_cvp_gen_monthly.index, calfews_cvp_gen_monthly['San_Luis'], 'r-', label = "CALFEWS Generation", linewidth = 2)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
plt.ylabel("Monthly Operations (GWh)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title(f"San Luis Monthly Generation and Consumption \n R = {round(correlation, 2)}", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.tight_layout()


# %% Tracy Pumping Plants 

#Compute the correlation between EIA and CALFEWS tracy pumping
correlation = stats.pearsonr(datDaily['delta_TRP_pump'].resample('M').sum(), 
                           cfs_tafd*input_data['TRP_pump'].resample('M').sum())[0]

plt.figure(figsize=(15,8))
plt.plot(cfs_tafd*input_data['TRP_pump'].resample('M').sum(), 'b-', label='True pumping',  linewidth=2)
plt.plot(datDaily['delta_TRP_pump'].resample('M').sum(), 'r-', label='CALFEWS pumping',  linewidth=2)
plt.ylabel("Pumping (TAF)", fontsize = 24)
plt.xlabel("Date", fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title(f"Tracy Pumping Plant - Monthly Operations \n R = {round(correlation, 2)}", fontsize = 28)
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', \
           fontsize=20, ncol = 2, frameon = False)
plt.tight_layout()
