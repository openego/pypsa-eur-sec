#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:57:21 2020

@author: bw0928

*****************************************************************************
This script calculates cost-energy_saving-curves for retrofitting
for the EU-37 countries, based on the building stock data from hotmaps and
the EU building stock database
*****************************************************************************

Structure:

    (1) set assumptions and parameters
    (2) read and prepare data
    (3) calculate (€-dE-curves)
    (4) save in csv / plot

*****************************************************************************
"""

import pandas as pd
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None
# %%  ******** (1) ASSUMPTIONS - PARAMETERS **********************************

k = 0.035   # thermal conductivity standard value
interest_rate = 0.04

annualise_cost = True   # annualise the investment costs
tax_weighting = False   # weight costs depending on taxes in countries
construction_index = False   # weight costs depending on costruction_index
plot = False

l_strength = ["0.04", "0.08"]  # additional insulation thickness
# strenght of relative retrofitting depending on the component
l_weight = pd.DataFrame({"weight": [1, 1, 1, 1]},  #[4,2,2,0.5]
                        index=["Roof", "Walls", "Floor", "Windows"])

# mapping missing countries by neighbours
map_for_missings = {"AL": ["BG", "RO", "ES"],
                    "BA": ["HR"],
                    "RS": ["BG", "RO", "HR", "HU"],
                    "MK": ["BG", "ES"],
                    "ME": ["BA", "AL", "RS", "HR"],
                    "CH": ["SE", "DE"],
                    "NO": ["SE"],
                    "PL": ["DE", "CZ", "HR"]}  # TODO: missing u-values of Poland should be added from eurostat

# %% ************ (2) DATA ***************************************************

# building data --------------------------------------------------------------
building_data = pd.read_csv(snakemake.input.building_stock,
                            usecols=list(range(13)))

# standardize data
building_data["type"].replace({'Covered area: heated  [Mm²]': 'Heated area [Mm²]',
                               'Windows ': 'Windows', 'Walls ': 'Walls',
                               'Roof ': 'Roof', 'Floor ': 'Floor'},
                              inplace=True)
building_data.country_code = building_data.country_code.str.upper()
building_data["subsector"].replace({'Hotels and Restaurants': 'Hotels and restaurants'},
                                   inplace=True)
building_data["sector"].replace({'Residential sector': 'residential',
                                 'Service sector': 'services'},
                                inplace=True)
u_values = building_data[(building_data.feature.str.contains("U-values"))
                         & (building_data.subsector != "Total")]

components = list(u_values.type.unique())
building_types = list(u_values.subsector.unique())

country_iso_dic = building_data.set_index("country")["country_code"].to_dict()
# add missing /rename countries
country_iso_dic.update({'Norway': 'NO',
                        'Iceland': 'IS',
                        'Montenegro': 'ME',
                        'Serbia': 'RS',
                        'Albania': 'AL',
                        'United Kingdom': 'GB',
                        'Bosnia and Herzegovina': 'BA'})

# average component surface --------------------------------------------------
# TODO average component surface from service sector
average_surface = (pd.read_csv(snakemake.input.average_surface,
                               nrows=3, header=1, index_col=0)
                   .rename({'Single/two family house': 'Single family- Terraced houses',
                            'Large apartment house': 'Multifamily houses',
                            'Apartment house': 'Appartment blocks'},
                           axis="index")).iloc[:, :6]
average_surface.columns = ["surface", "height", "Roof",
                           "Walls", "Floor", "Windows"]
average_surface_w = average_surface[components].apply(lambda x: x / x.sum(),
                                                      axis=1)

# heated floor area ----------------------------------------------------------
area = building_data[(building_data.type == 'Heated area [Mm²]') &
                     (building_data.subsector != "Total")]
area_tot = area.groupby(["country", "sector"]).sum()
area["weight"] = area.apply(lambda x: x.value /
                            area_tot.value.loc[(x.country, x.sector)], axis=1)
area = area.groupby(['country', 'sector', 'subsector', 'bage']).sum()
area_tot.rename(index=country_iso_dic, inplace=True)

# add for some missing countries floor area from other data sources
area_missing = pd.read_csv(snakemake.input.floor_area_missing,
                           index_col=[0, 1], usecols=[0, 1, 2, 3])
area_tot = area_tot.append(area_missing.unstack(level=-1).dropna().stack())
area_tot = area_tot.loc[~area_tot.index.duplicated(keep='last')]

# for other missing countries calculate floor area by population size
pop_layout = pd.read_csv(snakemake.input.clustered_pop_layout, index_col=0)
pop_layout["ct"] = pop_layout.index.str[:2]
ct_total = pop_layout.total.groupby(pop_layout["ct"]).sum()

area_per_pop = area_tot.unstack().apply(lambda x: x/ct_total[x.index])
missing_area_ct = ct_total.index.difference(area_tot.index.levels[0])
for ct in missing_area_ct:
    averaged_data = pd.DataFrame(
                    area_per_pop.value.reindex(map_for_missings[ct]).mean()
                    * ct_total[ct],
                    columns=["value"])
    index = pd.MultiIndex.from_product([[ct], averaged_data.index.to_list()])
    averaged_data.index = index
    averaged_data["estimated"] = 1
    if ct not in area_tot.index.levels[0]:
        area_tot = area_tot.append(averaged_data, sort=True)
    else:
        area_tot.loc[averaged_data.index] = averaged_data

#  only take considered countries into account
countries = ct_total.index
area_tot = area_tot.loc[countries]
# get share of residential and sevice floor area
sec_w = (area_tot / area_tot.groupby(["country"]).sum())["value"]

# costs for retrofitting -----------------------------------------------------
cost_retro = pd.read_csv(snakemake.input.cost_germany,
                         nrows=4, index_col=0, usecols=[0, 1, 2, 3])
cost_retro.index = cost_retro.index.str.capitalize()
cost_retro.rename(index={"Window": "Windows", "Wall": "Walls"}, inplace=True)

if annualise_cost:
    cost_retro = (cost_retro[["cost_fix", "cost_var"]]
                  .apply(lambda x: x * interest_rate /
                         (1 - (1 + interest_rate)
                          ** -cost_retro.loc[x.index, "life_time"])))

if construction_index:
    cost_w = pd.read_csv(snakemake.input.construction_index,
                         skiprows=3, nrows=31, index_col=0)
    # since German retrofitting costs are assumed
    cost_w = ((cost_w["2018"] / cost_w.loc["Germany", "2018"])
              .rename(index=country_iso_dic))

if tax_weighting:
    tax_w = pd.read_csv(snakemake.input.tax_w,
                        header=12, nrows=39, index_col=0, usecols=[0, 4])
    tax_w.rename(index=country_iso_dic, inplace=True)
    tax_w = tax_w.apply(pd.to_numeric, errors='coerce').iloc[:, 0]
    tax_w.dropna(inplace=True)

# %% ********** (3) CALCULATE COST-ENERGY-CURVES ****************************

energy_saved = u_values[['country', 'sector', 'subsector', 'bage', 'type']]
costs = u_values[['country', 'sector', 'subsector', 'bage', 'type']]

# for missing weighting of surfaces of building types assume Apartment blocks
u_values["assumed_subsector"] = u_values.subsector
u_values.assumed_subsector[~u_values.subsector.isin(average_surface.index)] = 'Appartment blocks'

for l in l_strength:
    u_values[l] = u_values.apply(lambda x:
                                 k/((k/x.value) +
                                    (float(l)*l_weight.loc[x.type][0])),
                                 axis=1)
    energy_saved[l] = u_values.apply(lambda x:
                                     x[l] / x.value *
                                     average_surface_w.loc[x.assumed_subsector, x.type],
                                     axis=1)
    costs[l] = u_values.apply(lambda x:
         (cost_retro.loc[x.type, "cost_var"] * 100 * float(l) *
         l_weight.loc[x.type][0] + cost_retro.loc[x.type, "cost_fix"])
        * average_surface.loc[x.assumed_subsector, x.type]
        / average_surface.loc[x.assumed_subsector, "surface"],
        axis=1)

# energy and costs per country, sector, subsector and year
e_tot = energy_saved.groupby(['country', 'sector', 'subsector', 'bage']).sum()
cost_tot = costs.groupby(['country', 'sector', 'subsector', 'bage']).sum()

# weighting by area -> energy and costs per country and sector
# in case of missing data first concat
energy_saved = pd.concat([e_tot, area.weight], axis=1)
cost_res = pd.concat([cost_tot, area.weight], axis=1)
energy_saved = (energy_saved.apply(lambda x: x*x.weight, axis=1)
                .groupby(level=[0, 1]).sum())
cost_res = (cost_res.apply(lambda x: x*x.weight, axis=1)
            .groupby(level=[0, 1]).sum())

# %%
res = pd.concat([energy_saved[l_strength], cost_res[l_strength]],
                axis=1, keys=["dE", "cost"])
res.rename(index=country_iso_dic, inplace=True)
res = res.loc[countries]
# map missing countries
for ct in map_for_missings.keys():
    averaged_data = pd.DataFrame(res.loc[map_for_missings[ct], :].mean(level=1))
    index = pd.MultiIndex.from_product([[ct], averaged_data.index.to_list()])
    averaged_data.index = index
    if ct not in res.index.levels[0]:
        res = res.append(averaged_data)
    else:
        res.loc[averaged_data.index] = averaged_data

# weights costs after construction index
if construction_index:
    for ct in list(map_for_missings.keys() - cost_w.index):
        cost_w.loc[ct] = cost_w.loc[map_for_missings[ct]].mean()
    res.cost = res.cost.apply(lambda x: x*cost_w[x.index.levels[0]])

# weights cost depending on country taxes
if tax_weighting:
    for ct in list(map_for_missings.keys() - tax_w.index):
        tax_w[ct] = tax_w.loc[map_for_missings[ct]].mean()
    res.cost = res.cost.apply(lambda x: x*tax_w[x.index.levels[0]])

# get the total cost-energy-savings weight by sector area
tot = res.apply(lambda col: col*sec_w, axis=0).groupby(level=0).sum()
tot.set_index(pd.MultiIndex.from_product([list(tot.index), ["tot"]]),
              inplace=True)
res = res.append(tot).unstack().stack()

summed_area = pd.DataFrame(area_tot.groupby("country").sum())
summed_area.set_index(pd.MultiIndex.from_product(
                      [list(summed_area.index), ["tot"]]), inplace=True)
area_tot = area_tot.append(summed_area).unstack().stack()

# %% ******* (4) SAVE + PLOT  ************************************************
res.to_csv(snakemake.output.retro_cost)
area_tot.to_csv(snakemake.output.floor_area)

#%% --- plot -------------------------------------------------------------
if plot:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ct in ['DE', 'SE', 'BG']: #res.index.levels[0]:
        dE = (res.loc[(ct, "tot"), "dE"]*100)
        cost = res.loc[(ct, "tot"), "cost"]
        df = pd.concat([dE, cost], axis=1)
        df.columns = ["dE", "cost/m²"]
        df.plot(x="dE", y="cost/m²", grid=True, label=ct, ax=ax)
        plt.xlim([0, 100])
#        plt.ylim([0, 7])
        plt.ylabel("Euro/m²")
        plt.xlabel("energy demand in % of unrefurbished")
    ax.legend(bbox_to_anchor=(1.5, 1.0), ncol=2)

# %% for testing
if 'snakemake' not in globals():
    import yaml
    import os
    from vresutils.snakemake import MockSnakemake
    os.chdir("/home/ws/bw0928/Dokumente/pypsa-eur-sec/")
    snakemake = MockSnakemake(
        wildcards=dict(
            network='elec',
            simpl='',
            clusters='38',
            lv='1',
            opts='Co2L-3H',
            sector_opts="[Co2L0p0-24H-T-H-B-I]"),
            input=dict(
                building_stock="data/retro/data_building_stock.csv",
                tax_w="data/retro/electricity_taxes_eu.csv",
                construction_index="data/retro/comparative_level_investment.csv",
                average_surface="data/retro/average_surface_components.csv",
                floor_area_missing="data/retro/floor_area_missing.csv",
                clustered_pop_layout="resources/pop_layout_{network}_s{simpl}_{clusters}.csv",
                cost_germany="data/retro/retro_cost_germany.csv"),
        output=dict(
                retro_cost="resources/retro_cost_{network}_s{simpl}_{clusters}.csv",
                floor_area="resources/floor_area_{network}_s{simpl}_{clusters}.csv")
        )
    with open('/home/ws/bw0928/Dokumente/pypsa-eur-sec/config.yaml', encoding='utf8') as f:
        snakemake.config = yaml.safe_load(f)
