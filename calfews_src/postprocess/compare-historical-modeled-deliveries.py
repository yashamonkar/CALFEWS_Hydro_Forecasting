import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import cm

cmap = cm.get_cmap('viridis')
cols = [cmap(0.1),cmap(0.3),cmap(0.6),cmap(0.8)]

### Read in modeled turnout flows (instead of contract deliveries)
canal_results = pd.read_csv('calfews_src/data/results/canal_results_validation.csv', index_col=0, parse_dates=True)

# change AEC to ARV to match historical data
canal_results['FKC_ARV_turnout'] = canal_results['FKC_AEC_turnout']
del canal_results['FKC_AEC_turnout']

# aggregate monthly sum
canal_results['Date__'] = canal_results.index
canal_results['Year__'] = canal_results.index.year
canal_results['Month__'] = canal_results.index.month
canal_results['Wateryear__'] = canal_results.Year__
canal_results.Wateryear__.loc[canal_results.Month__ > 9] = canal_results.Wateryear__.loc[canal_results.Month__ > 9] + 1

### get district and water delivery type
waterway = canal_results.columns.map(lambda x: x.split('_')[0])
turnout = canal_results.columns.map(lambda x: x.split('_')[1])
flow_type = canal_results.columns.map(lambda x: x.split('_')[2])

### only keep FKC & MDC, and turnout flow type
canal_results = canal_results.loc[:, (~(flow_type == 'flow')) & ((waterway == 'FKC') | (waterway == 'MDC') | (flow_type == ''))]
waterway = canal_results.columns.map(lambda x: x.split('_')[0])
turnout = canal_results.columns.map(lambda x: x.split('_')[1])
flow_type = canal_results.columns.map(lambda x: x.split('_')[2])

canal_results['Date'] = canal_results['Date__']
canal_results['Year'] = canal_results['Year__']
canal_results['Month'] = canal_results['Month__']
canal_results['Wateryear'] = canal_results['Wateryear__']
del canal_results['Date__'], canal_results['Year__'], canal_results['Month__'], canal_results['Wateryear__']


# get friant contractors
contractor = canal_results.columns.map(lambda x: 'other')
contractor.values[(turnout == 'CWC')|(turnout == 'MAD')|(turnout == 'FRS')|
                      (turnout == 'COF')|(turnout == 'OFK')|(turnout == 'TUL')|
                      (turnout == 'EXE')|(turnout == 'LDS')|(turnout == 'LND')|
                      (turnout == 'PRT')|(turnout == 'LWT')|(turnout == 'TPD')|
                      (turnout == 'SAU')|(turnout == 'TBA')|(turnout == 'OXV')|
                      (turnout == 'DLE')|(turnout == 'KRT')|(turnout == 'SSJ')|
                      (turnout == 'SFW')|(turnout == 'ARV')] = 'friant'

# get aggregated deliveries - annual
canal_results_aggregate_Wateryear = canal_results.groupby(by = 'Wateryear').sum()
canal_results_aggregate_Wateryear['Wateryear'] = canal_results.groupby(by = 'Wateryear')['Wateryear'].first()
canal_results_aggregate_Wateryear['Date'] = canal_results.groupby(by = 'Wateryear')['Date'].first()

# get aggregated deliveries - annual/monthly
canal_results['Wateryear__Month'] = canal_results.Wateryear.map(int).map(str) + '__' + canal_results.Month.map(int).map(str)
canal_results_aggregate_Wateryear_Month = canal_results.groupby(by = 'Wateryear__Month').sum()
canal_results_aggregate_Wateryear_Month['Wateryear'] = canal_results.groupby(by = 'Wateryear__Month')['Wateryear'].first()
canal_results_aggregate_Wateryear_Month['Month'] = canal_results.groupby(by = 'Wateryear__Month')['Month'].first()
canal_results_aggregate_Wateryear_Month['Date'] = canal_results.groupby(by = 'Wateryear__Month')['Date'].first()
canal_results_aggregate_Wateryear_Month = canal_results_aggregate_Wateryear_Month.sort_values('Date')

# get aggregated deliveries - aggregate months across years
canal_results_aggregate_Month = canal_results.groupby(by = 'Month').sum()
canal_results_aggregate_Month['Wateryear'] = canal_results.groupby(by = 'Month')['Wateryear'].first()
canal_results_aggregate_Month['Month'] = canal_results.groupby(by = 'Month')['Month'].first()
canal_results_aggregate_Month['Date'] = canal_results.groupby(by = 'Month')['Date'].first()
canal_results_aggregate_Month = canal_results_aggregate_Month.sort_values('Date')











### read in and aggregate historical CVP deliveries
cvp_historical = pd.read_csv('calfews_src/data/input/CVP_delivery_cleaned.csv')
cvp_historical = cvp_historical.loc[(cvp_historical.Wateryear < 2017) & (cvp_historical.Wateryear > 1996)]
cvp_historical_pumpin = cvp_historical.loc[cvp_historical.pumpin==True,:]
cvp_historical = cvp_historical.loc[cvp_historical.pumpin==False,:]

# aggregate by Canal__Date
cvp_historical['Canal__Date'] = cvp_historical.Canal + '__' + cvp_historical.Date.map(str)
cvp_historical_aggregate_canal = cvp_historical.groupby(by = ['Canal__Date']).sum()
cvp_historical_aggregate_canal['Year'] = cvp_historical.groupby(by = ['Canal__Date'])['Year'].first()
cvp_historical_aggregate_canal['Month'] = cvp_historical.groupby(by = ['Canal__Date'])['Month'].first()
cvp_historical_aggregate_canal['Wateryear'] = cvp_historical.groupby(by = ['Canal__Date'])['Wateryear'].first()
cvp_historical_aggregate_canal['Date'] = cvp_historical.groupby(by = ['Canal__Date'])['Date'].first()
cvp_historical_aggregate_canal['Canal'] = cvp_historical.groupby(by = ['Canal__Date'])['Canal'].first()
cvp_historical_aggregate_canal['Project'] = 'CVP'
cvp_historical_aggregate_canal = cvp_historical_aggregate_canal.reset_index(drop=True)

# cvp_historical_pumpin['Canal__Date'] = cvp_historical_pumpin.Canal + '__' + cvp_historical_pumpin.Date.map(str)
# cvp_historical_pumpin_aggregate = cvp_historical_pumpin.groupby(by = ['Canal__Date']).sum()
# cvp_historical_pumpin_aggregate['Year'] = cvp_historical_pumpin.groupby(by = ['Canal__Date'])['Year'].first()
# cvp_historical_pumpin_aggregate['Month'] = cvp_historical_pumpin.groupby(by = ['Canal__Date'])['Month'].first()
# cvp_historical_pumpin_aggregate['Wateryear'] = cvp_historical_pumpin.groupby(by = ['Canal__Date'])['Wateryear'].first()
# cvp_historical_pumpin_aggregate['Date'] = cvp_historical_pumpin.groupby(by = ['Canal__Date'])['Date'].first()
# cvp_historical_pumpin_aggregate['Canal'] = cvp_historical_pumpin.groupby(by = ['Canal__Date'])['Canal'].first()
# cvp_historical_pumpin_aggregate['Project'] = 'CVP'
# cvp_historical_pumpin_aggregate = cvp_historical_pumpin_aggregate.reset_index(drop=True)

# aggregate by Contractor_Date
cvp_historical['Contractor__Date'] = cvp_historical.contractor + '__' + cvp_historical.Date.map(str)
cvp_historical_aggregate_contractor = cvp_historical.groupby(by = ['Contractor__Date']).sum()
cvp_historical_aggregate_contractor['Year'] = cvp_historical.groupby(by = ['Contractor__Date'])['Year'].first()
cvp_historical_aggregate_contractor['Month'] = cvp_historical.groupby(by = ['Contractor__Date'])['Month'].first()
cvp_historical_aggregate_contractor['Wateryear'] = cvp_historical.groupby(by = ['Contractor__Date'])['Wateryear'].first()
cvp_historical_aggregate_contractor['Date'] = cvp_historical.groupby(by = ['Contractor__Date'])['Date'].first()
cvp_historical_aggregate_contractor['Canal'] = cvp_historical.groupby(by = ['Contractor__Date'])['Canal'].first()
cvp_historical_aggregate_contractor['contractor'] = cvp_historical.groupby(by = ['Contractor__Date'])['contractor'].first()
cvp_historical_aggregate_contractor['Project'] = 'CVP'
cvp_historical_aggregate_contractor = cvp_historical_aggregate_contractor.reset_index(drop=True)



### get water-year aggregated deliveries
cvp_historical['WaterUserCode__Wateryear'] = cvp_historical.WaterUserCode + '__' + cvp_historical.Wateryear.map(int).map(str)
cvp_historical_aggregate_WaterUserCode_Wateryear =  cvp_historical.groupby(by = ['WaterUserCode__Wateryear']).sum()
cvp_historical_aggregate_WaterUserCode_Wateryear['WaterUserCode'] = cvp_historical.groupby(by = ['WaterUserCode__Wateryear'])['WaterUserCode'].first()
cvp_historical_aggregate_WaterUserCode_Wateryear['Wateryear'] = cvp_historical.groupby(by = ['WaterUserCode__Wateryear'])['Wateryear'].first()
cvp_historical_aggregate_WaterUserCode_Wateryear['Canal'] = cvp_historical.groupby(by = ['WaterUserCode__Wateryear'])['Canal'].first()

cvp_historical['Contractor__Wateryear'] = cvp_historical.contractor + '__' + cvp_historical.Wateryear.map(int).map(str)
cvp_historical_aggregate_contractor_Wateryear =  cvp_historical.groupby(by = ['Contractor__Wateryear']).sum()
cvp_historical_aggregate_contractor_Wateryear['contractor'] = cvp_historical.groupby(by = ['Contractor__Wateryear'])['contractor'].first()
cvp_historical_aggregate_contractor_Wateryear['Wateryear'] = cvp_historical.groupby(by = ['Contractor__Wateryear'])['Wateryear'].first()
cvp_historical_aggregate_contractor_Wateryear['Date'] = cvp_historical.groupby(by = ['Contractor__Wateryear'])['Date'].first()

cvp_historical['Canal__Wateryear'] = cvp_historical.Canal + '__' + cvp_historical.Wateryear.map(int).map(str)
cvp_historical_aggregate_Canal_Wateryear =  cvp_historical.groupby(by = ['Canal__Wateryear']).sum()
cvp_historical_aggregate_Canal_Wateryear['Canal'] = cvp_historical.groupby(by = ['Canal__Wateryear'])['Canal'].first()
cvp_historical_aggregate_Canal_Wateryear['Wateryear'] = cvp_historical.groupby(by = ['Canal__Wateryear'])['Wateryear'].first()
cvp_historical_aggregate_Canal_Wateryear['Date'] = cvp_historical.groupby(by = ['Canal__Wateryear'])['Date'].first()



### get monthly aggregated deliveries
cvp_historical['WaterUserCode__Wateryear__Month'] = cvp_historical.WaterUserCode + '__' + cvp_historical.Wateryear.map(int).map(str) + '__' + cvp_historical.Month.map(int).map(str)
cvp_historical_aggregate_WaterUserCode_Wateryear_Month =  cvp_historical.groupby(by = ['WaterUserCode__Wateryear__Month']).sum()
cvp_historical_aggregate_WaterUserCode_Wateryear_Month['WaterUserCode'] = cvp_historical.groupby(by = ['WaterUserCode__Wateryear__Month'])['WaterUserCode'].first()
cvp_historical_aggregate_WaterUserCode_Wateryear_Month['Wateryear'] = cvp_historical.groupby(by = ['WaterUserCode__Wateryear__Month'])['Wateryear'].first()
cvp_historical_aggregate_WaterUserCode_Wateryear_Month['Month'] = cvp_historical.groupby(by = ['WaterUserCode__Wateryear__Month'])['Month'].first()
cvp_historical_aggregate_WaterUserCode_Wateryear_Month['Date'] = cvp_historical.groupby(by = ['WaterUserCode__Wateryear__Month'])['Date'].first()
cvp_historical_aggregate_WaterUserCode_Wateryear_Month['Canal'] = cvp_historical.groupby(by = ['WaterUserCode__Wateryear__Month'])['Canal'].first()
cvp_historical_aggregate_WaterUserCode_Wateryear_Month = cvp_historical_aggregate_WaterUserCode_Wateryear_Month.sort_values('Date')

cvp_historical['Contractor__Wateryear__Month'] = cvp_historical.contractor + '__' + cvp_historical.Wateryear.map(int).map(str) + '__' + cvp_historical.Month.map(int).map(str)
cvp_historical_aggregate_contractor_Wateryear_Month =  cvp_historical.groupby(by = ['Contractor__Wateryear__Month']).sum()
cvp_historical_aggregate_contractor_Wateryear_Month['contractor'] = cvp_historical.groupby(by = ['Contractor__Wateryear__Month'])['contractor'].first()
cvp_historical_aggregate_contractor_Wateryear_Month['Wateryear'] = cvp_historical.groupby(by = ['Contractor__Wateryear__Month'])['Wateryear'].first()
cvp_historical_aggregate_contractor_Wateryear_Month['Month'] = cvp_historical.groupby(by = ['Contractor__Wateryear__Month'])['Month'].first()
cvp_historical_aggregate_contractor_Wateryear_Month['Date'] = cvp_historical.groupby(by = ['Contractor__Wateryear__Month'])['Date'].first()
cvp_historical_aggregate_contractor_Wateryear_Month = cvp_historical_aggregate_contractor_Wateryear_Month.sort_values('Date')

cvp_historical['Canal__Wateryear__Month'] = cvp_historical.Canal + '__' + cvp_historical.Wateryear.map(int).map(str) + '__' + cvp_historical.Month.map(int).map(str)
cvp_historical_aggregate_Canal_Wateryear_Month =  cvp_historical.groupby(by = ['Canal__Wateryear__Month']).sum()
cvp_historical_aggregate_Canal_Wateryear_Month['Canal'] = cvp_historical.groupby(by = ['Canal__Wateryear__Month'])['Canal'].first()
cvp_historical_aggregate_Canal_Wateryear_Month['Wateryear'] = cvp_historical.groupby(by = ['Canal__Wateryear__Month'])['Wateryear'].first()
cvp_historical_aggregate_Canal_Wateryear_Month['Month'] = cvp_historical.groupby(by = ['Canal__Wateryear__Month'])['Month'].first()
cvp_historical_aggregate_Canal_Wateryear_Month['Date'] = cvp_historical.groupby(by = ['Canal__Wateryear__Month'])['Date'].first()
cvp_historical_aggregate_Canal_Wateryear_Month = cvp_historical_aggregate_Canal_Wateryear_Month.sort_values('Date')



### get monthly aggregated deliveries, across years
cvp_historical['WaterUserCode__Month'] = cvp_historical.WaterUserCode + '__' + cvp_historical.Month.map(int).map(str)
cvp_historical_aggregate_WaterUserCode_Month =  cvp_historical.groupby(by = ['WaterUserCode__Month']).sum()
cvp_historical_aggregate_WaterUserCode_Month['WaterUserCode'] = cvp_historical.groupby(by = ['WaterUserCode__Month'])['WaterUserCode'].first()
cvp_historical_aggregate_WaterUserCode_Month['Month'] = cvp_historical.groupby(by = ['WaterUserCode__Month'])['Month'].first()
cvp_historical_aggregate_WaterUserCode_Month['Canal'] = cvp_historical.groupby(by = ['WaterUserCode__Month'])['Canal'].first()

cvp_historical['Contractor__Month'] = cvp_historical.contractor + '__' + cvp_historical.Month.map(int).map(str)
cvp_historical_aggregate_contractor_Month =  cvp_historical.groupby(by = ['Contractor__Month']).sum()
cvp_historical_aggregate_contractor_Month['contractor'] = cvp_historical.groupby(by = ['Contractor__Month'])['contractor'].first()
cvp_historical_aggregate_contractor_Month['Month'] = cvp_historical.groupby(by = ['Contractor__Month'])['Month'].first()
cvp_historical_aggregate_contractor_Month['Date'] = cvp_historical.groupby(by = ['Contractor__Month'])['Date'].first()

cvp_historical['Canal__Month'] = cvp_historical.Canal + '__' + cvp_historical.Month.map(int).map(str)
cvp_historical_aggregate_Canal_Month =  cvp_historical.groupby(by = ['Canal__Month']).sum()
cvp_historical_aggregate_Canal_Month['Canal'] = cvp_historical.groupby(by = ['Canal__Month'])['Canal'].first()
cvp_historical_aggregate_Canal_Month['Month'] = cvp_historical.groupby(by = ['Canal__Month'])['Month'].first()
cvp_historical_aggregate_Canal_Month['Date'] = cvp_historical.groupby(by = ['Canal__Month'])['Date'].first()


### plot historical & modeled deliveries (based on turnouts) - friant contractors
fig = plt.figure(figsize=(18, 9))
ax = plt.subplot2grid((2,2), (0,0))
# ind = (friantDelivery_modeled == True) | (cvcDelivery_modeled == True) | (otherDelivery_modeled == True) | (subtractDoubleCount_modeled == True)
ind = ((waterway == 'FKC') | (waterway == 'MDC')) & (contractor == 'friant')
ax.plot_date(cvp_historical_aggregate_contractor_Wateryear_Month['Date'].loc[
                    (cvp_historical_aggregate_contractor_Wateryear_Month.contractor == 'friant')],
                  cvp_historical_aggregate_contractor_Wateryear_Month['delivery_taf'].loc[
                    (cvp_historical_aggregate_contractor_Wateryear_Month.contractor == 'friant')],
                  fmt='-', c=cols[0],alpha=0.7)
print(canal_results_aggregate_Wateryear_Month.Date)
print(canal_results_aggregate_Wateryear_Month.loc[:, ind])
ax.plot_date(canal_results_aggregate_Wateryear_Month.Date,
                  np.sum(canal_results_aggregate_Wateryear_Month.loc[:, ind], axis=1), fmt='--', c=cols[3],alpha=0.7)
ax.legend(['Historical', 'Modeled'])
ax.set_ylabel('Monthly deliveries (taf/month)')
ax.set_xlabel('Date')
ax = plt.subplot2grid((2,2), (0,1))
ax.scatter(cvp_historical_aggregate_contractor_Wateryear_Month['delivery_taf'].loc[(cvp_historical_aggregate_contractor_Wateryear_Month.contractor == 'friant')],
           np.sum(canal_results_aggregate_Wateryear_Month.loc[:, ind], axis=1),c=cols[0],alpha=0.7)
ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".7")
ax.annotate('friant contractors', xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.94,
                          ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.95))
ax.annotate(round(np.corrcoef(cvp_historical_aggregate_contractor_Wateryear_Month['delivery_taf'].loc[(cvp_historical_aggregate_contractor_Wateryear_Month.contractor == 'friant')],
                              np.sum(canal_results_aggregate_Wateryear_Month.loc[:, ind], axis=1))[1][0], 2),
            xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.94 ,
                ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.9))
ax.set_ylabel('Modeled deliveries (taf/month)')
ax.set_xlabel('Historical deliveries (taf/month)')
ax = plt.subplot2grid((2,2), (1,0))
ax.plot_date(cvp_historical_aggregate_contractor_Wateryear['Date'].loc[
                    (cvp_historical_aggregate_contractor_Wateryear.contractor == 'friant')],
                  cvp_historical_aggregate_contractor_Wateryear['delivery_taf'].loc[
                    (cvp_historical_aggregate_contractor_Wateryear.contractor == 'friant')],
                  fmt='-', c=cols[0],alpha=0.7)
ax.plot_date(canal_results_aggregate_Wateryear.Date,
                  np.sum(canal_results_aggregate_Wateryear.loc[:, ind], axis=1), fmt='--', c=cols[3],alpha=0.7)
ax.legend(['Historical', 'Modeled'])
ax.set_ylabel('Water year deliveries (taf/yr)')
ax.set_xlabel('Date')
ax = plt.subplot2grid((2,2), (1,1))
ax.scatter(cvp_historical_aggregate_contractor_Wateryear['delivery_taf'].loc[(cvp_historical_aggregate_contractor_Wateryear.contractor == 'friant')],
           np.sum(canal_results_aggregate_Wateryear.loc[:, ind], axis=1),c=cols[0],alpha=0.7)
ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".7")
ax.annotate('friant contractors', xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.94,
                          ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.95))
ax.annotate(round(np.corrcoef(cvp_historical_aggregate_contractor_Wateryear['delivery_taf'].loc[(cvp_historical_aggregate_contractor_Wateryear.contractor == 'friant')],
                              np.sum(canal_results_aggregate_Wateryear.loc[:, ind], axis=1))[1][0], 2),
            xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.94,
                ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.9))
ax.set_ylabel('Modeled deliveries (taf/yr)')
ax.set_xlabel('Historical deliveries (taf/yr)')
fig.savefig('calfews_src/figs/compareCVP/friant_allContractors.png', dpi=150)


### plot historical & modeled deliveries (based on turnouts) - FKC & MDC, contractors & non-contractors
fig = plt.figure(figsize=(18, 9))
ax = plt.subplot2grid((2,2), (0,0))
# ind = (friantDelivery_modeled == True) | (cvcDelivery_modeled == True) | (otherDelivery_modeled == True) | (subtractDoubleCount_modeled == True)
ind = (waterway == 'FKC') | (waterway == 'MDC')
ax.plot_date(cvp_historical_aggregate_Canal_Wateryear_Month['Date'].loc[
               (cvp_historical_aggregate_Canal_Wateryear_Month.Canal == 'MADERACANAL')],
                  cvp_historical_aggregate_Canal_Wateryear_Month['delivery_taf'].loc[
                    (cvp_historical_aggregate_Canal_Wateryear_Month.Canal == 'MADERACANAL')].values +\
             cvp_historical_aggregate_Canal_Wateryear_Month['delivery_taf'].loc[
                    (cvp_historical_aggregate_Canal_Wateryear_Month.Canal == 'FRIANT-KERNCANAL')].values,
                  fmt='-', c=cols[0],alpha=0.7)
ax.plot_date(canal_results_aggregate_Wateryear_Month.Date,
                  np.sum(canal_results_aggregate_Wateryear_Month.loc[:, ind], axis=1), fmt='--', c=cols[3],alpha=0.7)
ax.legend(['Historical', 'Modeled'])
ax.set_ylabel('Monthly deliveries (taf/month)')
ax.set_xlabel('Date')
ax = plt.subplot2grid((2,2), (0,1))
ax.scatter(cvp_historical_aggregate_Canal_Wateryear_Month['delivery_taf'].loc[
                    (cvp_historical_aggregate_Canal_Wateryear_Month.Canal == 'MADERACANAL')].values +\
             cvp_historical_aggregate_Canal_Wateryear_Month['delivery_taf'].loc[
                    (cvp_historical_aggregate_Canal_Wateryear_Month.Canal == 'FRIANT-KERNCANAL')].values,
           np.sum(canal_results_aggregate_Wateryear_Month.loc[:, ind], axis=1),c=cols[0],alpha=0.7)
ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".7")
ax.annotate('FKC/MDC all', xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.94,
                          ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.95))
ax.annotate(round(np.corrcoef(cvp_historical_aggregate_Canal_Wateryear_Month['delivery_taf'].loc[
                    (cvp_historical_aggregate_Canal_Wateryear_Month.Canal == 'MADERACANAL')].values +\
             cvp_historical_aggregate_Canal_Wateryear_Month['delivery_taf'].loc[
                    (cvp_historical_aggregate_Canal_Wateryear_Month.Canal == 'FRIANT-KERNCANAL')].values,
                              np.sum(canal_results_aggregate_Wateryear_Month.loc[:, ind], axis=1))[1][0], 2),
            xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.94 ,
                ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.9))
ax.set_ylabel('Modeled deliveries (taf/month)')
ax.set_xlabel('Historical deliveries (taf/month)')
ax = plt.subplot2grid((2,2), (1,0))
ax.plot_date(cvp_historical_aggregate_Canal_Wateryear['Date'].loc[
                    (cvp_historical_aggregate_Canal_Wateryear.Canal == 'MADERACANAL')],
             cvp_historical_aggregate_Canal_Wateryear['delivery_taf'].loc[
               (cvp_historical_aggregate_Canal_Wateryear.Canal == 'MADERACANAL')].values + \
             cvp_historical_aggregate_Canal_Wateryear['delivery_taf'].loc[
               (cvp_historical_aggregate_Canal_Wateryear.Canal == 'FRIANT-KERNCANAL')].values,
                  fmt='-', c=cols[0],alpha=0.7)
ax.plot_date(canal_results_aggregate_Wateryear.Date,
                  np.sum(canal_results_aggregate_Wateryear.loc[:, ind], axis=1), fmt='--', c=cols[3],alpha=0.7)
ax.legend(['Historical', 'Modeled'])
ax.set_ylabel('Water year deliveries (taf/yr)')
ax.set_xlabel('Date')
ax = plt.subplot2grid((2,2), (1,1))
ax.scatter(cvp_historical_aggregate_Canal_Wateryear['delivery_taf'].loc[
               (cvp_historical_aggregate_Canal_Wateryear.Canal == 'MADERACANAL')].values + \
             cvp_historical_aggregate_Canal_Wateryear['delivery_taf'].loc[
               (cvp_historical_aggregate_Canal_Wateryear.Canal == 'FRIANT-KERNCANAL')].values,
           np.sum(canal_results_aggregate_Wateryear.loc[:, ind], axis=1),c=cols[0],alpha=0.7)
ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".7")
ax.annotate('FKC/MDC all', xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.94,
                          ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.95))
ax.annotate(round(np.corrcoef(cvp_historical_aggregate_Canal_Wateryear['delivery_taf'].loc[
               (cvp_historical_aggregate_Canal_Wateryear.Canal == 'MADERACANAL')].values + \
             cvp_historical_aggregate_Canal_Wateryear['delivery_taf'].loc[
               (cvp_historical_aggregate_Canal_Wateryear.Canal == 'FRIANT-KERNCANAL')].values,
                              np.sum(canal_results_aggregate_Wateryear.loc[:, ind], axis=1))[1][0], 2),
            xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.94,
                ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.9))
ax.set_ylabel('Modeled deliveries (taf/yr)')
ax.set_xlabel('Historical deliveries (taf/yr)')
fig.savefig('calfews_src/figs/compareCVP/friant_all.png', dpi=150)




### plot annual deliveries for different districts - scatter
d_list = [[['MAD','CWC','FRS'],
           ['COF','TUL','EXE']],
          [['LND','LDS','LWT'],
           ['PRT','TPD','SAU']],
          [['TBA','DLE','KRT'],
           ['SSJ','SFW','ARV']]]
name_list = ['north','central','south']
for k in range(3):
  d = d_list[k]
  name = name_list[k]
  fig = plt.figure(figsize=(18,9))
  gs1 = gridspec.GridSpec(2,3)
  # gs1.update(wspace=0.3,hspace=0.3)
  for i in range(2):
    for j in range(3):
      ax = plt.subplot(gs1[i,j])
      ax.tick_params(axis='y', which='both', labelleft=True, labelright=False)
      ax.tick_params(axis='x', which='both', labelbottom=True, labeltop=False)
      if (d[i][j] != 'na'):
        ind = (turnout == d[i][j])
        ax.scatter(cvp_historical_aggregate_WaterUserCode_Wateryear['delivery_taf'].loc[cvp_historical_aggregate_WaterUserCode_Wateryear.WaterUserCode == d[i][j]],
                    np.sum(canal_results_aggregate_Wateryear.loc[:, ind],axis=1), c=cols[0], alpha=0.7)
        ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".7")
        ax.annotate(d[i][j], xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.91 ,
                                 ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.95))
        ax.annotate(round(np.corrcoef(cvp_historical_aggregate_WaterUserCode_Wateryear['delivery_taf'].loc[cvp_historical_aggregate_WaterUserCode_Wateryear.WaterUserCode == d[i][j]],
                                      np.sum(canal_results_aggregate_Wateryear.loc[:, ind], axis=1))[1][0], 2),
                    xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.91 ,
                        ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.88))
      ax.set_xlabel('')
      ax.set_ylabel('')
      # if ((i == 1) & (j == 0)):
      ax.set_ylabel('Modeled deliveries (taf/yr)')
      ax.set_xlabel('Historical deliveries (taf/yr)')
  plt.savefig('calfews_src/figs/compareCVP/annual_scatter_%s.png' % name, dpi=150)

### plot annual deliveries for different districts - time series
d_list = [[['MAD','CWC','FRS'],
           ['COF','TUL','EXE']],
          [['LND','LDS','LWT'],
           ['PRT','TPD','SAU']],
          [['TBA','DLE','KRT'],
           ['SSJ','SFW','ARV']]]
name_list = ['north','central','south']
for k in range(3):
  d = d_list[k]
  name = name_list[k]
  fig = plt.figure(figsize=(18,9))
  gs1 = gridspec.GridSpec(2,3)
  # gs1.update(wspace=0.3,hspace=0.3)
  for i in range(2):
    for j in range(3):
      ax = plt.subplot(gs1[i,j])
      ax.tick_params(axis='y', which='both', labelleft=True, labelright=False)
      ax.tick_params(axis='x', which='both', labelbottom=True, labeltop=False)
      ind = (turnout == d[i][j])
      ax.plot(cvp_historical_aggregate_WaterUserCode_Wateryear['Wateryear'].loc[
                      cvp_historical_aggregate_WaterUserCode_Wateryear.WaterUserCode == d[i][j]],
                    cvp_historical_aggregate_WaterUserCode_Wateryear['delivery_taf'].loc[
                      cvp_historical_aggregate_WaterUserCode_Wateryear.WaterUserCode == d[i][j]], c=cols[0])
      ax.plot(np.sum(canal_results_aggregate_Wateryear.loc[:, ind], axis=1), c=cols[3], ls='--')
      ax.annotate(d[i][j], xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.91,
                               ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.95))
      ax.set_xlabel('')
      ax.set_ylabel('')
      if ((i == 0) & (j == 2)):
        ax.legend(['Historical','Modeled'])
      ax.set_ylabel('Deliveries (taf/yr)')
      ax.set_xlabel('Date')
  plt.savefig('calfews_src/figs/compareCVP/annual_timeseries_%s.png' % name, dpi=150)





### plot monthly deliveries for different districts - scatter
d_list = [[['MAD','CWC','FRS'],
           ['COF','TUL','EXE']],
          [['LND','LDS','LWT'],
           ['PRT','TPD','SAU']],
          [['TBA','DLE','KRT'],
           ['SSJ','SFW','ARV']]]
name_list = ['north','central','south']
for k in range(3):
  d = d_list[k]
  name = name_list[k]
  fig = plt.figure(figsize=(18,9))
  gs1 = gridspec.GridSpec(2,3)
  # gs1.update(wspace=0.3,hspace=0.3)
  for i in range(2):
    for j in range(3):
      ax = plt.subplot(gs1[i,j])
      ax.tick_params(axis='y', which='both', labelleft=True, labelright=False)
      ax.tick_params(axis='x', which='both', labelbottom=True, labeltop=False)
      ind = (turnout == d[i][j])
      ax.scatter(cvp_historical_aggregate_WaterUserCode_Wateryear_Month['delivery_taf'].loc[
                      (cvp_historical_aggregate_WaterUserCode_Wateryear_Month.WaterUserCode == d[i][j])],
                 np.sum(canal_results_aggregate_Wateryear_Month.loc[:, ind], axis=1), c=cols[0], alpha=0.7)
      ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".7")
      ax.annotate(d[i][j], xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.91,
                               ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.95))
      ax.annotate(round(np.corrcoef(cvp_historical_aggregate_WaterUserCode_Wateryear_Month['delivery_taf'].loc[
                      (cvp_historical_aggregate_WaterUserCode_Wateryear_Month.WaterUserCode == d[i][j])],
                              np.sum(canal_results_aggregate_Wateryear_Month.loc[:, ind], axis=1))[1][0], 2),
                  xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.91,
                      ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.88))
      ax.set_xlabel('')
      ax.set_ylabel('')
      # if ((i == 1) & (j == 0)):
      ax.set_ylabel('Modeled deliveries (taf/yr)')
      ax.set_xlabel('Historical deliveries (taf/yr)')
  plt.savefig('calfews_src/figs/compareCVP/monthly_scatter_%s.png' % name, dpi=150)




### plot monthly deliveries for different districts - time series
d_list = [[['MAD','CWC','FRS'],
           ['COF','TUL','EXE']],
          [['LND','LDS','LWT'],
           ['PRT','TPD','SAU']],
          [['TBA','DLE','KRT'],
           ['SSJ','SFW','ARV']]]
name_list = ['north','central','south']
for k in range(3):
  d = d_list[k]
  name = name_list[k]
  fig = plt.figure(figsize=(18,9))
  gs1 = gridspec.GridSpec(2,3)
  # gs1.update(wspace=0.3,hspace=0.3)
  for i in range(2):
    for j in range(3):
      ax = plt.subplot(gs1[i,j])
      ax.tick_params(axis='y', which='both', labelleft=True, labelright=False)
      ax.tick_params(axis='x', which='both', labelbottom=True, labeltop=False)
      ax.plot_date(cvp_historical_aggregate_WaterUserCode_Wateryear_Month['Date'].loc[
                      (cvp_historical_aggregate_WaterUserCode_Wateryear_Month.WaterUserCode == d[i][j])],
                    cvp_historical_aggregate_WaterUserCode_Wateryear_Month['delivery_taf'].loc[
                      (cvp_historical_aggregate_WaterUserCode_Wateryear_Month.WaterUserCode == d[i][j])],
                    fmt='-', c=cols[0])
      ind = (turnout == d[i][j])
      ax.plot_date(canal_results_aggregate_Wateryear_Month.Date,
                    np.sum(canal_results_aggregate_Wateryear_Month.loc[:, ind], axis=1), fmt='--', c=cols[3])
      ax.annotate(d[i][j], xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.91,
                               ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.95))
      ax.set_xlabel('')
      ax.set_ylabel('')
      if ((i == 0) & (j == 2)):
        ax.legend(['Historical','Modeled'])
      # if ((i == 1) & (j == 0)):
      ax.set_ylabel('Deliveries (taf/month)')
      ax.set_xlabel('Date')
  plt.show()
  plt.savefig('calfews_src/figs/compareCVP/monthly_timeseries_%s.png' % name, dpi=150)




### plot historical & modeled deliveries (based on turnouts) - friant contractors
fig = plt.figure(figsize=(18, 9))
gs1 = gridspec.GridSpec(3,4)
months = [10,11,12,1,2,3,4,5,6,7,8,9]
for i in range(3):
  for j in range(4):
    month = months[4*i + j]
    ax = plt.subplot2grid((3,4), (i,j))
    ind_turnout = ((waterway == 'FKC') | (waterway == 'MDC')) & (contractor == 'friant')
    ind_month = canal_results_aggregate_Wateryear_Month.Month == month
    ax.scatter(cvp_historical_aggregate_contractor_Wateryear_Month['delivery_taf'].loc[
                        (cvp_historical_aggregate_contractor_Wateryear_Month.contractor == 'friant') & (cvp_historical_aggregate_contractor_Wateryear_Month.Month == month)],
               np.sum(canal_results_aggregate_Wateryear_Month.loc[ind_month, ind_turnout], axis=1),c=cols[0],alpha=0.7)
    ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".7")
    ax.annotate('m='+str(month), xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.8,
                              ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.12))
    ax.annotate(r'$\rho=$' + str(round(np.corrcoef(cvp_historical_aggregate_contractor_Wateryear_Month['delivery_taf'].loc[
                        (cvp_historical_aggregate_contractor_Wateryear_Month.contractor == 'friant') & (cvp_historical_aggregate_contractor_Wateryear_Month.Month == month)],
                                               np.sum(canal_results_aggregate_Wateryear_Month.loc[ind_month, ind_turnout],axis=1))[1][0], 2)),
                xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.8,
                    ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.03))
    ax.set_ylabel('Modeled deliveries (taf/month)')
    ax.set_xlabel('Historical deliveries (taf/month)')
fig.savefig('calfews_src/figs/compareCVP/scatter_byMonth_allContractors.png', dpi=150)



### plot historical & modeled deliveries (based on turnouts) - friant contractors
d_list = ['MAD','CWC','FRS','COF','TUL','EXE','LND','LDS','LWT','PRT','TPD','SAU','TBA','DLE','KRT','SSJ','SFW','ARV']
months = [10,11,12,1,2,3,4,5,6,7,8,9]
for d in d_list:
  fig = plt.figure(figsize=(18, 9))
  gs1 = gridspec.GridSpec(3, 4)
  for i in range(3):
    for j in range(4):
      month = months[4*i + j]
      ax = plt.subplot2grid((3,4), (i,j))
      ind_turnout = (turnout == d)
      ind_month = canal_results_aggregate_Wateryear_Month.Month == month
      ax.scatter(cvp_historical_aggregate_WaterUserCode_Wateryear_Month['delivery_taf'].loc[
                          (cvp_historical_aggregate_WaterUserCode_Wateryear_Month.WaterUserCode == d)&(cvp_historical_aggregate_WaterUserCode_Wateryear_Month.Month == month)],
                 np.sum(canal_results_aggregate_Wateryear_Month.loc[ind_month, ind_turnout], axis=1),c=cols[0],alpha=0.7)
      ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".7")
      ax.annotate(d, xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.65,
                         ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02))
      ax.annotate('m=' + str(month), xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.8,
                                         ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.12))
      ax.annotate(r'$\rho=$' + str(round(np.corrcoef(cvp_historical_aggregate_WaterUserCode_Wateryear_Month['delivery_taf'].loc[
                          (cvp_historical_aggregate_WaterUserCode_Wateryear_Month.WaterUserCode == d)&(cvp_historical_aggregate_WaterUserCode_Wateryear_Month.Month == month)],
                                                     np.sum(canal_results_aggregate_Wateryear_Month.loc[
                                                              ind_month, ind_turnout], axis=1))[1][0], 2)),
                  xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.8,
                      ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.03))
      ax.set_ylabel('Modeled deliveries (taf/month)')
      ax.set_xlabel('Historical deliveries (taf/month)')
  fig.savefig('calfews_src/figs/compareCVP/scatter_byMonth_%s.png' %d, dpi=150)


#############################################################################
# GRAVEYARD
#########################################################################
# ### Read in modeled daily district delivery data
# df_modeled = pd.read_csv('calfews_src/data/results/district_results_full_validation.csv', index_col=0, parse_dates=True)
#
# # aggregate monthly sum
# df_modeled['Date__'] = df_modeled.index
# df_modeled['Year__'] = df_modeled.index.year
# df_modeled['Month__'] = df_modeled.index.month
# df_modeled['Wateryear__'] = df_modeled.Year__
# df_modeled.Wateryear__.loc[df_modeled.Month__ > 9] = df_modeled.Wateryear__.loc[df_modeled.Month__ > 9] + 1
#
# ### get district and water delivery type
# districts_modeled = df_modeled.columns.map(lambda x: x.split('_')[0])
# WT_modeled = df_modeled.columns.map(lambda x: x.split('_',1)[1])
# source_modeled = WT_modeled.map(lambda x: x.split('_',1)[0])
# WT_modeled = df_modeled.columns.map(lambda x: x.split('_')[-1])
#
# # binary variable to say whether it is friant delivery
# friantDelivery_modeled = df_modeled.columns.map(lambda x: False)
# cvcDelivery_modeled = df_modeled.columns.map(lambda x: False)
# swpDelivery_modeled = df_modeled.columns.map(lambda x: False)
# cvpdeltaDelivery_modeled = df_modeled.columns.map(lambda x: False)
# localDelivery_modeled = df_modeled.columns.map(lambda x: False)
# otherDelivery_modeled = df_modeled.columns.map(lambda x: False)
# subtractDoubleCount_modeled = df_modeled.columns.map(lambda x: False)
# friantDelivery_modeled.values[((source_modeled == 'friant1')|(source_modeled=='friant2')) & ((WT_modeled == 'delivery')|(WT_modeled == 'flood'))] = True
# cvcDelivery_modeled.values[((source_modeled=='cvc')) & ((WT_modeled == 'delivery')|(WT_modeled == 'flood'))] = True
# swpDelivery_modeled.values[((source_modeled == 'tableA')) & ((WT_modeled == 'delivery')|(WT_modeled == 'flood'))] = True
# cvpdeltaDelivery_modeled.values[((source_modeled == 'cvpdelta')) & ((WT_modeled == 'delivery')|(WT_modeled == 'flood'))] = True
# localDelivery_modeled.values[((source_modeled == 'kings') | (source_modeled == 'kaweah') | (source_modeled == 'tule') | (source_modeled == 'kern')) & ((WT_modeled == 'delivery')|(WT_modeled == 'flood'))] = True
# otherDelivery_modeled.values[[i for i,v in enumerate(WT_modeled) if v in
#                 ['banked','SW']]] = True
# otherDelivery_modeled.values[[i for i,v in enumerate(source_modeled) if v in
#                 ['recover','inleiu']]] = True
# subtractDoubleCount_modeled.values[[i for i,v in enumerate(WT_modeled) if v in
#                 ['recharged']]] = True
#
# df_modeled['Date'] = df_modeled['Date__']
# df_modeled['Year'] = df_modeled['Year__']
# df_modeled['Month'] = df_modeled['Month__']
# df_modeled['Wateryear'] = df_modeled['Wateryear__']
# del df_modeled['Date__'], df_modeled['Year__'], df_modeled['Month__'], df_modeled['Wateryear__']
#
# # deliveries are cumulative over water year, so difference to get monthly deliveries
# ind = np.where((friantDelivery_modeled == True) | (cvcDelivery_modeled == True) | (swpDelivery_modeled == True) | (cvpdeltaDelivery_modeled == True) | (localDelivery_modeled == True) | (otherDelivery_modeled == True) | (subtractDoubleCount_modeled == True))[0]
# for i in ind:
#   for wy in range(min(df_modeled.Wateryear), max(df_modeled.Wateryear)+1):
#     startDay = np.where(df_modeled.Wateryear == wy)[0][0]
#     df_modeled.iloc[(startDay+1):(startDay+12), i] = np.diff(df_modeled.iloc[(startDay):(startDay+12), i])
#
# # make recharged water negative, so as not to double count deliveries of project water to water bank
# ind = (subtractDoubleCount_modeled == True)
# df_modeled.loc[:,ind] = -df_modeled.loc[:,ind]
#
# # get friant contractors
# contractor = df_modeled.columns.map(lambda x: 'other')
# contractor.values[(districts_modeled == 'CWC')|(districts_modeled == 'MAD')|(districts_modeled == 'FRS')|
#                       (districts_modeled == 'COF')|(districts_modeled == 'OFK')|(districts_modeled == 'TUL')|
#                       (districts_modeled == 'EXE')|(districts_modeled == 'LDS')|(districts_modeled == 'LND')|
#                       (districts_modeled == 'PRT')|(districts_modeled == 'LWT')|(districts_modeled == 'TPD')|
#                       (districts_modeled == 'SAU')|(districts_modeled == 'TBA')|(districts_modeled == 'OXV')|
#                       (districts_modeled == 'DLE')|(districts_modeled == 'KRT')|(districts_modeled == 'SSJ')|
#                       (districts_modeled == 'SFW')|(districts_modeled == 'ARV')] = 'friant'
#
# # get aggregated deliveries - annual
# df_modeled_aggregate_Wateryear = df_modeled.groupby(by = 'Wateryear').sum()
# df_modeled_aggregate_Wateryear['Wateryear'] = df_modeled.groupby(by = 'Wateryear')['Wateryear'].first()
# df_modeled_aggregate_Wateryear['Date'] = df_modeled.groupby(by = 'Wateryear')['Date'].first()
#
# # get aggregated deliveries - monthly
# df_modeled['Wateryear__Month'] = df_modeled.Wateryear.map(int).map(str) + '__' + df_modeled.Month.map(int).map(str)
# df_modeled_aggregate_Wateryear_Month = df_modeled.groupby(by = 'Wateryear__Month').sum()
# df_modeled_aggregate_Wateryear_Month['Wateryear'] = df_modeled.groupby(by = 'Wateryear__Month')['Wateryear'].first()
# df_modeled_aggregate_Wateryear_Month['Month'] = df_modeled.groupby(by = 'Wateryear__Month')['Month'].first()
# df_modeled_aggregate_Wateryear_Month['Date'] = df_modeled.groupby(by = 'Wateryear__Month')['Date'].first()
# df_modeled_aggregate_Wateryear_Month = df_modeled_aggregate_Wateryear_Month.sort_values('Date')
#
# # get shortened version for SWP comparison
# df_modeled_aggregate_Wateryear_swp = df_modeled_aggregate_Wateryear.loc[df_modeled_aggregate_Wateryear.Wateryear > 2000, :]
# df_modeled_aggregate_Wateryear_Month_swp = df_modeled_aggregate_Wateryear_Month.loc[df_modeled_aggregate_Wateryear_Month.Wateryear > 2000, :]

# ### plots based on modeled deliveries, not turnouts
# ### plot historical & modeled deliveries - FKC
# ax = plt.subplot2grid((2,2), (0,0))
# ind = (friantDelivery_modeled == True) | (cvcDelivery_modeled == True) | (otherDelivery_modeled == True) | (subtractDoubleCount_modeled == True)
# ax.plot_date(cvp_historical_aggregate_contractor_Wateryear_Month['Date'].loc[
#                     (cvp_historical_aggregate_contractor_Wateryear_Month.contractor == 'friant')],
#                   cvp_historical_aggregate_contractor_Wateryear_Month['delivery_taf'].loc[
#                     (cvp_historical_aggregate_contractor_Wateryear_Month.contractor == 'friant')],
#                   fmt='-', c=cols[0],alpha=0.7)
# ax.plot_date(df_modeled_aggregate_Wateryear_Month.Date,
#                   np.sum(df_modeled_aggregate_Wateryear_Month.loc[:, ind], axis=1), fmt='--', c=cols[3],alpha=0.7)
# ax.legend(['Historical', 'Modeled'])
# ax.set_ylabel('Monthly deliveries (taf/month)')
# ax.set_xlabel('Date')
# ax = plt.subplot2grid((2,2), (0,1))
# ax.scatter(cvp_historical_aggregate_contractor_Wateryear_Month['delivery_taf'].loc[(cvp_historical_aggregate_contractor_Wateryear_Month.contractor == 'friant')],
#            np.sum(df_modeled_aggregate_Wateryear_Month.loc[:, ind], axis=1),c=cols[0],alpha=0.7)
# ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".7")
# ax.annotate('friant', xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.94,
#                           ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.95))
# ax.annotate(round(np.corrcoef(cvp_historical_aggregate_contractor_Wateryear_Month['delivery_taf'].loc[(cvp_historical_aggregate_contractor_Wateryear_Month.contractor == 'friant')],
#                               np.sum(df_modeled_aggregate_Wateryear_Month.loc[:, ind], axis=1))[1][0], 2),
#             xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.94 ,
#                 ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.9))
# ax.set_ylabel('Modeled deliveries (taf/month)')
# ax.set_xlabel('Historical deliveries (taf/month)')
# ax = plt.subplot2grid((2,2), (1,0))
# ax.plot_date(cvp_historical_aggregate_contractor_Wateryear['Date'].loc[
#                     (cvp_historical_aggregate_contractor_Wateryear.contractor == 'friant')],
#                   cvp_historical_aggregate_contractor_Wateryear['delivery_taf'].loc[
#                     (cvp_historical_aggregate_contractor_Wateryear.contractor == 'friant')],
#                   fmt='-', c=cols[0],alpha=0.7)
# ax.plot_date(df_modeled_aggregate_Wateryear.Date,
#                   np.sum(df_modeled_aggregate_Wateryear.loc[:, ind], axis=1), fmt='--', c=cols[3],alpha=0.7)
# ax.legend(['Historical', 'Modeled'])
# ax.set_ylabel('Water year deliveries (taf/yr)')
# ax.set_xlabel('Date')
# ax = plt.subplot2grid((2,2), (1,1))
# ax.scatter(cvp_historical_aggregate_contractor_Wateryear['delivery_taf'].loc[(cvp_historical_aggregate_contractor_Wateryear.contractor == 'friant')],
#            np.sum(df_modeled_aggregate_Wateryear.loc[:, ind], axis=1),c=cols[0],alpha=0.7)
# ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".7")
# ax.annotate('friant', xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.94,
#                           ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.95))
# ax.annotate(round(np.corrcoef(cvp_historical_aggregate_contractor_Wateryear['delivery_taf'].loc[(cvp_historical_aggregate_contractor_Wateryear.contractor == 'friant')],
#                               np.sum(df_modeled_aggregate_Wateryear.loc[:, ind], axis=1))[1][0], 2),
#             xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.94,
#                 ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.9))
# ax.set_ylabel('Modeled deliveries (taf/yr)')
# ax.set_xlabel('Historical deliveries (taf/yr)')
#
#
#
#
#
#
#
#
# ### plot annual deliveries for different districts - scatter
# d = [['MAD','CWC','FRS'],
#      ['COF','TUL','EXE']]
# # d = [['LND','LDS','LWT'],
# #      ['PRT','TPD','SAU']]
# # d = [['TBA','DLE','KRT'],
# #      ['SSJ','SFW','ARV']]
# gs1 = gridspec.GridSpec(2,3)
# # gs1.update(wspace=0.3,hspace=0.3)
# for i in range(2):
#   for j in range(3):
#     ax = plt.subplot(gs1[i,j])
#     ax.tick_params(axis='y', which='both', labelleft=True, labelright=False)
#     ax.tick_params(axis='x', which='both', labelbottom=True, labeltop=False)
#     if (d[i][j] != 'na'):
#       ind = ((districts_modeled == d[i][j]) &
#              ((friantDelivery_modeled == True) | (cvcDelivery_modeled == True) | (otherDelivery_modeled == True) | (subtractDoubleCount_modeled == True)))
#       ax.scatter(cvp_historical_aggregate_WaterUserCode_Wateryear['delivery_taf'].loc[cvp_historical_aggregate_WaterUserCode_Wateryear.WaterUserCode == d[i][j]],
#                   np.sum(df_modeled_aggregate_Wateryear.loc[:, ind],axis=1), c=cols[0], alpha=0.7)
#       ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".7")
#       ax.annotate(d[i][j], xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.91 ,
#                                ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.95))
#       ax.annotate(round(np.corrcoef(cvp_historical_aggregate_WaterUserCode_Wateryear['delivery_taf'].loc[cvp_historical_aggregate_WaterUserCode_Wateryear.WaterUserCode == d[i][j]],
#                                     np.sum(df_modeled_aggregate_Wateryear.loc[:, ind], axis=1))[1][0], 2),
#                   xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.91 ,
#                       ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.88))
#     ax.set_xlabel('')
#     ax.set_ylabel('')
#     # if ((i == 1) & (j == 0)):
#     ax.set_ylabel('Modeled deliveries (taf/yr)')
#     ax.set_xlabel('Historical deliveries (taf/yr)')
#
#
#
#
# ### plot annual deliveries for different districts - time series
# # d = [['MAD','CWC','FRS'],
# #      ['COF','TUL','EXE']]
# # d = [['LND','LDS','LWT'],
# #      ['PRT','TPD','SAU']]
# d = [['TBA','DLE','KRT'],
#      ['SSJ','SFW','ARV']]
# gs1 = gridspec.GridSpec(2,3)
# # gs1.update(wspace=0.3,hspace=0.3)
# for i in range(2):
#   for j in range(3):
#     ax = plt.subplot(gs1[i,j])
#     ax.tick_params(axis='y', which='both', labelleft=True, labelright=False)
#     ax.tick_params(axis='x', which='both', labelbottom=True, labeltop=False)
#     ind = ((districts_modeled == d[i][j]) &
#            ((friantDelivery_modeled == True) | (cvcDelivery_modeled == True) | (otherDelivery_modeled == True)))
#     ax.plot(cvp_historical_aggregate_WaterUserCode_Wateryear['Wateryear'].loc[
#                     cvp_historical_aggregate_WaterUserCode_Wateryear.WaterUserCode == d[i][j]],
#                   cvp_historical_aggregate_WaterUserCode_Wateryear['delivery_taf'].loc[
#                     cvp_historical_aggregate_WaterUserCode_Wateryear.WaterUserCode == d[i][j]], c=cols[0])
#     ax.plot(np.sum(df_modeled_aggregate_Wateryear.loc[:, ind], axis=1), c=cols[3])
#     ax.annotate(d[i][j], xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.91,
#                              ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.95))
#     ax.set_xlabel('')
#     ax.set_ylabel('')
#     if ((i == 0) & (j == 2)):
#       ax.legend(['Historical','Modeled'])
#     ax.set_ylabel('Deliveries (taf/yr)')
#     ax.set_xlabel('Date')
#
#
#
#
#
# ### plot monthly deliveries for different districts - scatter
# d = [['MAD','CWC','FRS'],
#      ['COF','TUL','EXE']]
# # d = [['LND','LDS','LWT'],
# #      ['PRT','TPD','SAU']]
# # d = [['TBA','DLE','KRT'],
# #      ['SSJ','SFW','ARV']]
# gs1 = gridspec.GridSpec(2,3)
# # gs1.update(wspace=0.3,hspace=0.3)
# for i in range(2):
#   for j in range(3):
#     ax = plt.subplot(gs1[i,j])
#     ax.tick_params(axis='y', which='both', labelleft=True, labelright=False)
#     ax.tick_params(axis='x', which='both', labelbottom=True, labeltop=False)
#     ind = ((districts_modeled == d[i][j]) & ((friantDelivery_modeled == True) | (cvcDelivery_modeled == True) | (otherDelivery_modeled == True)))
#     ax.scatter(cvp_historical_aggregate_WaterUserCode_Wateryear_Month['delivery_taf'].loc[
#                     (cvp_historical_aggregate_WaterUserCode_Wateryear_Month.WaterUserCode == d[i][j])],
#                np.sum(df_modeled_aggregate_Wateryear_Month.loc[:, ind], axis=1), c=cols[0], alpha=0.7)
#     ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".7")
#     ax.annotate(d[i][j], xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.91,
#                              ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.95))
#     ax.annotate(round(np.corrcoef(cvp_historical_aggregate_WaterUserCode_Wateryear_Month['delivery_taf'].loc[
#                     (cvp_historical_aggregate_WaterUserCode_Wateryear_Month.WaterUserCode == d[i][j])],
#                             np.sum(df_modeled_aggregate_Wateryear_Month.loc[:, ind], axis=1))[1][0], 2),
#                 xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.91,
#                     ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.88))
#     ax.set_xlabel('')
#     ax.set_ylabel('')
#     # if ((i == 1) & (j == 0)):
#     ax.set_ylabel('Modeled deliveries (taf/yr)')
#     ax.set_xlabel('Historical deliveries (taf/yr)')
#
#
#
#
#
#
#
# ### plot monthly deliveries for different districts - time series
# d = [['MAD','CWC','FRS'],
#      ['COF','TUL','EXE']]
# # d = [['LND','LDS','LWT'],
# #      ['PRT','TPD','SAU']]
# # d = [['TBA','DLE','KRT'],
# #      ['SSJ','SFW','ARV']]
# gs1 = gridspec.GridSpec(2,3)
# # gs1.update(wspace=0.3,hspace=0.3)
# for i in range(2):
#   for j in range(3):
#     ax = plt.subplot(gs1[i,j])
#     ax.tick_params(axis='y', which='both', labelleft=True, labelright=False)
#     ax.tick_params(axis='x', which='both', labelbottom=True, labeltop=False)
#     ax.plot_date(cvp_historical_aggregate_WaterUserCode_Wateryear_Month['Date'].loc[
#                     (cvp_historical_aggregate_WaterUserCode_Wateryear_Month.WaterUserCode == d[i][j])],
#                   cvp_historical_aggregate_WaterUserCode_Wateryear_Month['delivery_taf'].loc[
#                     (cvp_historical_aggregate_WaterUserCode_Wateryear_Month.WaterUserCode == d[i][j])],
#                   fmt='-', c=cols[0])
#     ind = ((districts_modeled == d[i][j]) & ((friantDelivery_modeled == True) | (cvcDelivery_modeled == True) | (otherDelivery_modeled == True)))
#     ax.plot_date(df_modeled_aggregate_Wateryear_Month.Date,
#                   np.sum(df_modeled_aggregate_Wateryear_Month.loc[:, ind], axis=1), fmt='--', c=cols[3])
#     ax.annotate(d[i][j], xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.91,
#                              ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.95))
#     ax.set_xlabel('')
#     ax.set_ylabel('')
#     if ((i == 0) & (j == 2)):
#       ax.legend(['Historical','Modeled'])
#     # if ((i == 1) & (j == 0)):
#     ax.set_ylabel('Deliveries (taf/month)')
#     ax.set_xlabel('Date')




# ### read in & aggregate SWP historical data
# swp_historical = pd.read_csv('calfews_src/data/input/SWP_delivery_cleaned.csv')
# swp_historical = swp_historical.loc[(swp_historical.Wateryear >= 2001) & (swp_historical.Wateryear <= 2016), :]
# # only use is_delivery==1 (should be approximately total deliveries, but not sure about 'other' water types)
# swp_historical = swp_historical.loc[swp_historical.is_delivery == 1]
#
#
# # aggregate by Wateryear-Agency_Group
# swp_historical['Wateryear__Agency_Group'] = swp_historical.Wateryear.map(str) + '__' + swp_historical.Agency_Group
# swp_historical_aggregate_AgencyGroup_Wateryear = swp_historical.groupby(by = ['Wateryear__Agency_Group']).sum()
# swp_historical_aggregate_AgencyGroup_Wateryear['Wateryear'] = swp_historical.groupby(by = ['Wateryear__Agency_Group'])['Wateryear'].first()
# swp_historical_aggregate_AgencyGroup_Wateryear['Year'] = swp_historical_aggregate_AgencyGroup_Wateryear['Wateryear']
# swp_historical_aggregate_AgencyGroup_Wateryear['Month'] = 10
# swp_historical_aggregate_AgencyGroup_Wateryear['Day'] = 1
# swp_historical_aggregate_AgencyGroup_Wateryear['Agency_Group'] = swp_historical.groupby(by = ['Wateryear__Agency_Group'])['Agency_Group'].first()
# swp_historical_aggregate_AgencyGroup_Wateryear['Date'] = pd.to_datetime(swp_historical_aggregate_AgencyGroup_Wateryear[['Year','Month','Day']])
# swp_historical_aggregate_AgencyGroup_Wateryear = swp_historical_aggregate_AgencyGroup_Wateryear.sort_values(by=['Date'])
#
# swp_historical['Wateryear__Month__Agency_Group'] = swp_historical.Wateryear.map(str) + '__' + swp_historical.Month.map(str) + '__' + swp_historical.Agency_Group
# swp_historical_aggregate_AgencyGroup_Wateryear_Month = swp_historical.groupby(by = ['Wateryear__Month__Agency_Group']).sum()
# swp_historical_aggregate_AgencyGroup_Wateryear_Month['Wateryear'] = swp_historical.groupby(by = ['Wateryear__Month__Agency_Group'])['Wateryear'].first()
# swp_historical_aggregate_AgencyGroup_Wateryear_Month['Year'] = swp_historical.groupby(by = ['Wateryear__Month__Agency_Group'])['Year'].first()
# swp_historical_aggregate_AgencyGroup_Wateryear_Month['Month'] = swp_historical.groupby(by = ['Wateryear__Month__Agency_Group'])['Month'].first()
# swp_historical_aggregate_AgencyGroup_Wateryear_Month['Day'] = 1
# swp_historical_aggregate_AgencyGroup_Wateryear_Month['Agency_Group'] = swp_historical.groupby(by = ['Wateryear__Month__Agency_Group'])['Agency_Group'].first()
# swp_historical_aggregate_AgencyGroup_Wateryear_Month['Date'] = pd.to_datetime(swp_historical_aggregate_AgencyGroup_Wateryear_Month[['Year','Month','Day']])
# swp_historical_aggregate_AgencyGroup_Wateryear_Month = swp_historical_aggregate_AgencyGroup_Wateryear_Month.sort_values(by=['Date'])
#
# # aggregate by Wateryear-Month-Agency_Group-WT_Group
# swp_historical['Wateryear__Agency_Group__WT_Group'] = swp_historical.Wateryear.map(str) + '__' + swp_historical.Agency_Group + '__' + swp_historical.WT_Group
# swp_historical_aggregate_AgencyGroup_WTGroup_Wateryear = swp_historical.groupby(by = ['Wateryear__Agency_Group__WT_Group']).sum()
# swp_historical_aggregate_AgencyGroup_WTGroup_Wateryear['Wateryear'] = swp_historical.groupby(by = ['Wateryear__Agency_Group__WT_Group'])['Wateryear'].first()
# swp_historical_aggregate_AgencyGroup_WTGroup_Wateryear['Year'] = swp_historical_aggregate_AgencyGroup_WTGroup_Wateryear['Wateryear']
# swp_historical_aggregate_AgencyGroup_WTGroup_Wateryear['Month'] = 10
# swp_historical_aggregate_AgencyGroup_WTGroup_Wateryear['Day'] = 1
# swp_historical_aggregate_AgencyGroup_WTGroup_Wateryear['Agency_Group'] = swp_historical.groupby(by = ['Wateryear__Agency_Group__WT_Group'])['Agency_Group'].first()
# swp_historical_aggregate_AgencyGroup_WTGroup_Wateryear['WT_Group'] = swp_historical.groupby(by = ['Wateryear__Agency_Group__WT_Group'])['WT_Group'].first()
# swp_historical_aggregate_AgencyGroup_WTGroup_Wateryear['Date'] = pd.to_datetime(swp_historical_aggregate_AgencyGroup_WTGroup_Wateryear[['Year','Month','Day']])
# swp_historical_aggregate_AgencyGroup_WTGroup_Wateryear = swp_historical_aggregate_AgencyGroup_WTGroup_Wateryear.sort_values(by=['Date'])
#
# swp_historical['Wateryear__Month__Agency_Group__WT_Group'] = swp_historical.Wateryear.map(str) + '__' + swp_historical.Month.map(str) + '__' + swp_historical.Agency_Group + '__' + swp_historical.WT_Group
# swp_historical_aggregate_AgencyGroup_WTGroup_Wateryear_Month = swp_historical.groupby(by = ['Wateryear__Month__Agency_Group__WT_Group']).sum()
# swp_historical_aggregate_AgencyGroup_WTGroup_Wateryear_Month['Wateryear'] = swp_historical.groupby(by = ['Wateryear__Month__Agency_Group__WT_Group'])['Wateryear'].first()
# swp_historical_aggregate_AgencyGroup_WTGroup_Wateryear_Month['Year'] = swp_historical.groupby(by = ['Wateryear__Month__Agency_Group__WT_Group'])['Year'].first()
# swp_historical_aggregate_AgencyGroup_WTGroup_Wateryear_Month['Month'] = swp_historical.groupby(by = ['Wateryear__Month__Agency_Group__WT_Group'])['Month'].first()
# swp_historical_aggregate_AgencyGroup_WTGroup_Wateryear_Month['Day'] = 1
# swp_historical_aggregate_AgencyGroup_WTGroup_Wateryear_Month['Agency_Group'] = swp_historical.groupby(by = ['Wateryear__Month__Agency_Group__WT_Group'])['Agency_Group'].first()
# swp_historical_aggregate_AgencyGroup_WTGroup_Wateryear_Month['WT_Group'] = swp_historical.groupby(by = ['Wateryear__Month__Agency_Group__WT_Group'])['WT_Group'].first()
# swp_historical_aggregate_AgencyGroup_WTGroup_Wateryear_Month['Date'] = pd.to_datetime(swp_historical_aggregate_AgencyGroup_WTGroup_Wateryear_Month[['Year','Month','Day']])
# swp_historical_aggregate_AgencyGroup_WTGroup_Wateryear_Month = swp_historical_aggregate_AgencyGroup_WTGroup_Wateryear_Month.sort_values(by=['Date'])
#
#
#
#
# ### plot annual deliveries for different districts - scatter
# d = [['WSL','TLB','DLR'],
#      ['LHL','BDM','BLR']]
# d = [['WSL','TLB','DLR'],
#      ['SMI','BVA','WRM']]
# gs1 = gridspec.GridSpec(2,3)
# # gs1.update(wspace=0.3,hspace=0.3)
# for i in range(2):
#   for j in range(3):
#     ax = plt.subplot(gs1[i,j])
#     ax.tick_params(axis='y', which='both', labelleft=True, labelright=False)
#     ax.tick_params(axis='x', which='both', labelbottom=True, labeltop=False)
#     if (d[i][j] != 'na'):
#       ind = ((districts_modeled == d[i][j]) &
#              ((swpDelivery_modeled == True) | (cvpdeltaDelivery_modeled == True) | (otherDelivery_modeled == True)))
#       ax.scatter(swp_historical_aggregate_AgencyGroup_Wateryear['delivery_taf'].loc[swp_historical_aggregate_AgencyGroup_Wateryear['Agency_Group'] == d[i][j]],
#                   np.sum(df_modeled_aggregate_Wateryear_swp.loc[:, ind],axis=1), c=cols[0], alpha=0.7)
#       ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".7")
#       ax.annotate(d[i][j], xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.91 ,
#                                ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.95))
#       ax.annotate(round(np.corrcoef(swp_historical_aggregate_AgencyGroup_Wateryear['delivery_taf'].loc[swp_historical_aggregate_AgencyGroup_Wateryear['Agency_Group'] == d[i][j]],
#                                     np.sum(df_modeled_aggregate_Wateryear_swp.loc[:, ind], axis=1))[1][0], 2),
#                   xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.91 ,
#                       ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.88))
#     ax.set_xlabel('')
#     ax.set_ylabel('')
#     # if ((i == 1) & (j == 0)):
#     ax.set_ylabel('Modeled deliveries (taf/yr)')
#     ax.set_xlabel('Historical deliveries (taf/yr)')
#
#
#
# ### plot annual deliveries for different districts - scatter
# d = [['TLB','DLR','LHL'],
#      ['BDM','BLR','SMI']]
# d = [['WSL','TLB','DLR'],
#      ['SMI','BVA','WRM']]
# gs1 = gridspec.GridSpec(2,3)
# # gs1.update(wspace=0.3,hspace=0.3)
# for i in range(2):
#   for j in range(3):
#     ax = plt.subplot(gs1[i,j])
#     ax.tick_params(axis='y', which='both', labelleft=True, labelright=False)
#     ax.tick_params(axis='x', which='both', labelbottom=True, labeltop=False)
#     if (d[i][j] != 'na'):
#       ind = ((districts_modeled == d[i][j]) &
#              ((swpDelivery_modeled == True) | (otherDelivery_modeled == True)))
#       ax.scatter(swp_historical_aggregate_AgencyGroup_Wateryear_Month['delivery_taf'].loc[swp_historical_aggregate_AgencyGroup_Wateryear_Month['Agency_Group'] == d[i][j]],
#                   np.sum(df_modeled_aggregate_Wateryear_Month_swp.loc[:, ind],axis=1), c=cols[0], alpha=0.7)
#       ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".7")
#       ax.annotate(d[i][j], xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.91 ,
#                                ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.95))
#       ax.annotate(round(np.corrcoef(swp_historical_aggregate_AgencyGroup_Wateryear_Month['delivery_taf'].loc[swp_historical_aggregate_AgencyGroup_Wateryear_Month['Agency_Group'] == d[i][j]],
#                                     np.sum(df_modeled_aggregate_Wateryear_Month_swp.loc[:, ind], axis=1))[1][0], 2),
#                   xy=(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0])*0.91 ,
#                       ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.88))
#     ax.set_xlabel('')
#     ax.set_ylabel('')
#     # if ((i == 1) & (j == 0)):
#     ax.set_ylabel('Modeled deliveries (taf/yr)')
#     ax.set_xlabel('Historical deliveries (taf/yr)')