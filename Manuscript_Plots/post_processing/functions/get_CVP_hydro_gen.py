# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:07:39 2025

@author: amonkar

Code to compute the total CVP Hydropower Generation
"""

import pandas as pd
import numpy as np



#%%
def get_CVP_hydro_gen(storage_data, releases_data):
    
    """
    Both releases and storage values have to be in TAF.
    
    It requires the input of storage data and releases data with specific column
    names shown below.
    
    Storage data at: (in TAF)
        1. Shasta
        2. Trinity
        3. Folsom
        4. New Melones
        5. San Luis
    
    Release data at: (in TAF)
        1. Shasta
        2. Trinity
        3. Trinity Diversions
        4. Folsom
        5. New Melones
        6. San Luis Federal Releases
    """
    
    #-------------------------------------------------------------------------#
    #####--------------------------Shasta---------------------------------#####
    #-------------------------------------------------------------------------#
    #Hyper-parameters
    shasta_capacity = 676 #MW
    shasta_discharge_capacity = 34.909 # TAc-ft/day https://web.archive.org/web/20121003060702/http://www.usbr.gov/pmts/hydraulics_lab/pubs/PAP/PAP-0845.pdf
    
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
        def get_kwh_per_af(df):
            return 1.045 * ((0.83522 * df) + 30.5324)
        kwh_per_AF = get_kwh_per_af(gross_head)
        
        return kwh_per_AF*release/10**3 #GWh
    
    #Compute the Generation  with Discharge Capacity
    shasta_gen = get_shasta_generation(releases_data['Shasta'], #TAF 
                                       storage_data['Shasta'],  #TAF
                                       shasta_discharge_capacity) #TAF
    
    #-------------------------------------------------------------------------#
    #####--------------------------Trinity---------------------------------#####
    #-------------------------------------------------------------------------#
    #Hyper-parameters
    trinity_capacity = 140 #MW
    trinity_discharge_capacity = 7.317 # TAc-ft/day  
    
    def get_trinity_generation(release, storage, discharge_capacity = 9999999):
        
        #Curtail Releases
        release = release.apply(lambda x: min(x, discharge_capacity))
        
        #Tailwater Elevation
        tailwater_elevation = 1901.5
        
        #Covert Storage to Forebay Elevation 
        def get_forebay_elev_trinity(df):
            return (2028.137624 + 
                    (0.0005032905066 * (df * 1000)) - 
                    (0.0000000005134105556 * (df * 1000) ** 2) + 
                    (3.281723733E-16 * (df * 1000) ** 3) - 
                    (1.048527299E-22 * (df * 1000) ** 4) + 
                    (1.292215614E-29 * (df * 1000) ** 5))
        forebay_elevation = get_forebay_elev_trinity(storage)
        
        #Compute the Gross Head 
        gross_head = forebay_elevation-tailwater_elevation
        
        #Compute the power generation potential (kwh/AF)
        def get_kwh_per_af(df):
            return (1.19285 * df) - 142.1086
        kwh_per_AF = get_kwh_per_af(gross_head)
        
        return kwh_per_AF*release/10**3 #GWh
    
    #Compute the Generation  with Discharge Capacity
    trinity_gen = get_trinity_generation(releases_data['Trinity'], #TAF 
                                       storage_data['Trinity'],  #TAF
                                       trinity_discharge_capacity) #TAF
    
    
    #-------------------------------------------------------------------------#
    #####--------------------------Carr---------------------------------#####
    #-------------------------------------------------------------------------#
    #Assumes all water from the diversions transfers here. 
    #Hyper-parameters
    carr_capacity = 184 #MW
    carr_discharge_capacity = 7.333 # TAc-ft/day 
    
    def get_carr_generation(release, discharge_capacity = 9999999):
        
        #Curtail Releases
        release = release.apply(lambda x: min(x, discharge_capacity))
        
        #Tailwater Elevation
        def get_tailwater_elev_carr(df):
            return (1061.53868 + 
                    (0.00173068872 * (df * 1000)) - 
                    (0.00000001320368111 * (df * 1000) ** 2) + 
                    (6.942087896E-14 * (df * 1000) ** 3) - 
                    (1.903353493E-19 * (df * 1000) ** 4) + 
                    (2.072893412E-25 * (df * 1000) ** 5))
        tailwater_elevation = 1200 #get_tailwater_elev_carr(storage)
        
        #Covert Storage to Forebay Elevation 
        forebay_elevation = 1901.5
        
        #Compute the Gross Head 
        gross_head = forebay_elevation-tailwater_elevation
        
        #Compute the power generation potential (kwh/AF)
        def get_kwh_per_af(df):
            return (1.19285 * df) - 142.1086
        kwh_per_AF = get_kwh_per_af(gross_head)
        
        return kwh_per_AF*release/10**3 #GWh
    
    #Compute the Generation  with Discharge Capacity
    carr_gen = get_carr_generation(releases_data['Diversions'], #TAF
                                   carr_discharge_capacity) #TAF
    
    #-------------------------------------------------------------------------#
    #####--------------------------Spring Creek---------------------------#####
    #-------------------------------------------------------------------------#
    #Assumes all water from the diversions transfers here. 
    #Hyper-parameters
    spring_creek_capacity = 200 #MW
    spring_creek_discharge_capacity = 8.6 # TAc-ft/day 
    
    def get_spring_creek_generation(release, discharge_capacity = 9999999):
        
        #Curtail Releases
        release = release.apply(lambda x: min(x, discharge_capacity))
        
        #Tailwater Elevation
        tailwater_elevation = 583.5 #get_tailwater_elev_carr(storage)
        
        #Covert Storage to Forebay Elevation 
        def get_forebay_elev_spring_creek(df):
            return (1061.53868 + 
                    (0.00173068872 * (df * 1000)) - 
                    (0.00000001320368111 * (df * 1000) ** 2) + 
                    (6.942087896E-14 * (df * 1000) ** 3) - 
                    (1.903353493E-19 * (df * 1000) ** 4) + 
                    (2.072893412E-25 * (df * 1000) ** 5))
        forebay_elevation = 1200 #get_forebay_elev_spring_creek(storage)
        
        #Compute the Gross Head 
        gross_head = forebay_elevation-tailwater_elevation
        
        #Compute the power generation potential (kwh/AF)
        def get_kwh_per_af(df):
            return (1.19285 * df) - 142.1086
        kwh_per_AF = get_kwh_per_af(gross_head)
        
        return kwh_per_AF*release/10**3 #GWh
    
    #Compute the Generation  with Discharge Capacity
    spring_creek_gen = get_spring_creek_generation(releases_data['Diversions'], #TAF 
                                                   spring_creek_discharge_capacity) #TAF
    
    #-------------------------------------------------------------------------#
    #####--------------------------Keswick--------------------------------#####
    #-------------------------------------------------------------------------#
    #Hyper-parameters
    keswick_capacity = 117 #MW
    keswick_discharge_capacity = 31.73 # TAc-ft/day https://web.archive.org/web/20121003060702/http://www.usbr.gov/pmts/hydraulics_lab/pubs/PAP/PAP-0845.pdf
    
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
        def get_kwh_per_af(df):
            return 0.70399 * df + 9.4772

        kwh_per_AF = get_kwh_per_af(gross_head)
        
        return kwh_per_AF * release / 10 ** 3
    
    #Compute the Keswick Generation
    keswick_gen = get_keswick_generation(releases_data['Shasta'] + releases_data['Diversions'],
                                         keswick_discharge_capacity)


    
    #-------------------------------------------------------------------------#
    #####--------------------------Folsom---------------------------------#####
    #-------------------------------------------------------------------------#
    #Hyper-parameters
    folsom_capacity = 198.7 #MW
    folsom_discharge_capacity = 13.682 # TAc-ft/day https://lowimpacthydro.org/wp-content/uploads/2021/05/Folsom-Nimbus-LIHI-Application-2021-March-5-2021-final.pdf
    
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
        def get_kwh_per_af(df):
            return 0.92854 * df - 16.282
        kwh_per_AF = get_kwh_per_af(gross_head)
        
        return kwh_per_AF*release/10**3 #GWh
    
    #Compute the Generation 
    folsom_gen = get_folsom_generation(releases_data['Folsom'], 
                                       storage_data['Folsom'],
                                       folsom_discharge_capacity)
    
    #####--------------------------Nimbus---------------------------------#####
    #Hyperparameters 
    nimbus_capacity = 15 #MW
    nimbus_discharge_capacity = 10.11 # TAc-ft/day https://lowimpacthydro.org/wp-content/uploads/2021/05/Folsom-Nimbus-LIHI-Application-2021-March-5-2021-final.pdf
    
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
        def get_kwh_per_af(df):
            return 0.11191 * df + 29.8156
        kwh_per_AF = get_kwh_per_af(gross_head)
        
        return kwh_per_AF*release/10**3 #GWh
    
    #Compute the Generation 
    nimbus_gen = get_nimbus_generation(releases_data['Folsom'], #TAF
                                       nimbus_discharge_capacity)

    #-------------------------------------------------------------------------#
    #####------------------------New Melones------------------------------#####
    #-------------------------------------------------------------------------#
    #Hyper-parameters
    new_melones_capacity = 300 #MW
    new_melones_discharge_capacity = 16.450 #TAC-ft/day https://www.waterboards.ca.gov/waterrights/water_issues/programs/hearings/auburn_dam/exhibits/x_8.pdf
    
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
        def get_kwh_per_af(df):
            return 0.62455 * df + 142.3077

        kwh_per_AF = get_kwh_per_af(gross_head)
        
        return kwh_per_AF * release / 10 ** 3
    
    #Compute the Generation 
    new_melones_gen = get_new_melones_generation(releases_data['New Melones'],
                                                 storage_data['New Melones'], 
                                                 new_melones_discharge_capacity)
    
    #-------------------------------------------------------------------------#
    #####-------------------------San Luis--------------------------------#####
    #-------------------------------------------------------------------------#
    #Hyper-parameters
    #Hyper-parameters
    san_luis_capacity = 424 #MW (Technically half of it)
    san_luis_discharge_capacity = 26 #https://en.wikipedia.org/wiki/San_Luis_Dam#:~:text=The%20San%20Luis%20Pumping%2DGenerating,when%20the%20water%20is%20discharged.
    san_luis_pumping_capacity = 21.8
    
    def get_san_luis_generation(release, storage, discharge_capacity = 9999999 , pumping_capacity = 9999999):
        
        #Tailwater elevation for San Luis is fixed
        tailwater_elevation = 220
        
        #Curtail Releases
        release = release.apply(lambda x: min(x, discharge_capacity))
        release = release.apply(lambda x: max(x, -pumping_capacity))
        
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
        def get_kwh_per_af(df):
            return 0.917325 * df - 11.0233

        kwh_per_AF = get_kwh_per_af(gross_head)
        
        return kwh_per_AF * release / 10 ** 3 
    
    #Compute the Generation 
    san_luis_gen = get_san_luis_generation(releases_data['San Luis'],
                                           storage_data['San Luis'],
                                           san_luis_discharge_capacity,
                                           san_luis_pumping_capacity)
    
    
    #####-------------------------O'Neill--------------------------------#####
    
    def get_oneill_generation(release, discharge_capacity = 9999999 , pumping_capacity = 9999999):
        
        #Curtail Releases
        release = release.apply(lambda x: min(x, discharge_capacity))
        release = release.apply(lambda x: max(x, -pumping_capacity))
        
        #Tailwater elevation for San Luis is fixed
        tailwater_elevation = 172
        
        forebay_elevation = 220
        
        # Compute the Gross Head 
        gross_head = forebay_elevation - tailwater_elevation
        
        # Compute the power generation potential (kwh/AF)
        def get_kwh_per_af(df):
            return df/1.3714

        kwh_per_AF = get_kwh_per_af(gross_head)
        
        return kwh_per_AF * release / 10 ** 3
    
    #Compute the Generation 
    oneill_gen = get_oneill_generation(releases_data['San Luis'],
                                         san_luis_discharge_capacity,
                                         san_luis_pumping_capacity)
    
    #Compile and return the values
    df = pd.DataFrame({
        'Shasta': shasta_gen,
        'Trinity': trinity_gen,
        'Carr': carr_gen,
        'Spring_Creek': spring_creek_gen,
        'Keswick': keswick_gen,
        'Folsom': folsom_gen,
        'Nimbus': nimbus_gen,
        'New_Melones': new_melones_gen,
        'San_Luis': san_luis_gen,
        'Oneill': oneill_gen
        })
    
    #Total CVP Generation
    df['CVP_Gen'] = df.sum(axis=1)
    
    return df






