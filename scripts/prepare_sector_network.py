# coding: utf-8
import numpy as np
import xarray as xr
import re
import os
import pypsa
import yaml
import pytz
import logging
import pandas as pd
from vresutils.costdata import annuity
from six import iteritems, string_types

logger = logging.getLogger(__name__)
idx = pd.IndexSlice


# First tell PyPSA that links can have multiple outputs by
# overriding the component_attrs. This can be done for
# as many buses as you need with format busi for i = 2,3,4,5,....
# See https://pypsa.org/doc/components.html#
# link-with-multiple-outputs-or-inputs
override_component_attrs = pypsa.descriptors.Dict({k: v.copy() for k, v in
                                                   pypsa.components.component_attrs.items()})
override_component_attrs["Link"].loc["bus2"] = ["string", np.nan, np.nan,
                                                "2nd bus", "Input (optional)"]
override_component_attrs["Link"].loc["bus3"] = ["string", np.nan, np.nan,
                                                "3rd bus", "Input (optional)"]
override_component_attrs["Link"].loc["efficiency2"] = ["static or series",
                                                       "per unit", 1.,
                                                       "2nd bus efficiency",
                                                       "Input (optional)"]
override_component_attrs["Link"].loc["efficiency3"] = ["static or series",
                                                       "per unit", 1.,
                                                       "3rd bus efficiency",
                                                       "Input (optional)"]
override_component_attrs["Link"].loc["p2"] = ["series", "MW", 0.,
                                              "2nd bus output", "Output"]
override_component_attrs["Link"].loc["p3"] = ["series", "MW", 0.,
                                              "3rd bus output", "Output"]


# -------------- FUNCTIONS ----------------------------------------------------
def space_heat_retro(demand, years, r_rate=None, dE=None, option=None):
    if option == "all":
        """assuming that already retrofitted buildings can get retrofitted
           again depends also on the observed period(=years) e.g lifetime
           windows 25 years"""
        demand_future = demand
        for i in range(1, years):
            demand_future = (1 - r_rate) * demand_future + \
                r_rate * dE * demand_future
    elif option == "once":
        """ first only not retrofitted buildings are renovated, if the observed
        period is longer than the needed time to renovate the whole stock,
        already renovated building are renovated again"""
        for i in range(int(r_rate * years // 1) + 1):
            demand_future = (1 - r_rate * years) * demand + \
                r_rate * years * dE * demand
            demand = demand_future
    elif option == "retro_steps":
        """after
        http://www.europarl.europa.eu/RegData/etudes/STUD/2016/587326/IPOL_STU(2016)587326_EN.pdf p.22
        and own calculation on the assumption that 1% of the building stock
        is renovated extensively,
        - > 1.5% of the building stock are yearly renovated
        wherat 85% undergo moderate renovations (10% energy savings),
        10% moderate (30% energy savings) and 5% extensive (60% energy savings)
        """
        demand_future = demand
        for i in range(1, years):
            demand_future = 0.985 * demand_future + 0.015 * demand_future * \
                            (0.85 * 0.9 + 0.1 * 0.7 + 0.05 * 0.4)
    elif option == "EU-target":
        """
        the EU parliament targets a space heat reduction of 80% until 2050
        """
        demand_future = 0.2 * demand
    return demand_future


def attach_wind_costs(n, costs):
    """update pypsa eur costs for offwind """
    for tech in ["onwind"]:  # ,"offwind-ac", "offwind-dc"]:

        with xr.open_dataset('../pypsa-eur/resources/profile_{}.nc'
                             .format(tech)) as ds:
            #            if ds.indexes['bus'].empty: continue
            suptech = tech.split('-', 2)[0]
            if suptech == 'offwind':
                underwater_fraction = ds['underwater_fraction'].to_pandas()
                connection_cost = (1.25 *
                                   ds['average_distance'].to_pandas() *
                                   (underwater_fraction *
                                    costs.at[tech + '-connection-submarine',
                                             'fixed'] +
                                    (1. - underwater_fraction) *
                                    costs.at[tech + '-connection-underground',
                                             'fixed']))
                capital_cost = (costs.at['offwind', 'fixed'] +
                                costs.at[tech + '-station', 'fixed'] +
                                connection_cost)
                logger.info("Added connection cost of {:0.0f}-{:0.0f} Eur/MW/a\
                            to {}".format(connection_cost.min(),
                                          connection_cost.max(), tech))
            elif suptech == 'onwind':
                # +  ' land-cost are included in DEA
                capital_cost = costs.at['onwind', 'fixed']
                # costs.at['onwind-landcosts', 'fixed'])
            else:
                capital_cost = costs.at[tech, 'capital_cost']

            gens = n.generators[n.generators.carrier == tech].index
            n.generators.loc[gens, "capital_cost"] = capital_cost.mean()

            return n


def insert_electricity_distribution_grid(network):
    f_costs = options['electricity_distribution_grid_cost_factor']
    print("Inserting electricity distribution grid with investment cost\
          factor of", f_costs)

    nodes = pop_layout.index

    network.madd("Bus",
                 nodes + " low voltage",
                 carrier="low voltage")

    network.madd("Link",
                 nodes + " electricity distribution grid",
                 bus0=nodes,
                 bus1=nodes + " low voltage",
                 p_nom_extendable=True,
                 p_min_pu=-1,
                 carrier="electricity distribution grid",
                 efficiency=1,
                 marginal_cost=0,
                 # TODO add costs to cost.csv
                 capital_cost=47505*f_costs)

    loads = network.loads.index[network.loads.carrier == "electricity"]
    network.loads.loc[loads, "bus"] += " low voltage"

    bevs = network.links.index[network.links.carrier == "BEV charger"]
    network.links.loc[bevs, "bus0"] += " low voltage"

    v2gs = network.links.index[network.links.carrier == "V2G"]
    network.links.loc[v2gs, "bus1"] += " low voltage"

    hps = network.links.index[network.links.carrier.str.contains("heat pump")]
    network.links.loc[hps, "bus0"] += " low voltage"

    rh = network.links.index[network.links.carrier.str.contains("resistive")]
    network.links.loc[rh, "bus0"] += " low voltage"

    mchp = network.links.index[network.links.carrier.str.contains("micro gas")]
    network.links.loc[mchp, "bus1"] += " low voltage"

    #set existing solar to cost of utility cost rather the 50-50 rooftop-utility
    solar = network.generators.index[network.generators.carrier == "solar"]
    network.generators.loc[solar, "capital_cost"] = costs.at['solar-utility',
                                                             'fixed']

    network.madd("Generator",
                 solar,
                 suffix=" rooftop",
                 bus=network.generators.loc[solar, "bus"] + " low voltage",
                 carrier="solar rooftop",
                 p_nom_extendable=True,
                 p_nom_max=network.generators.loc[solar, "p_nom_max"],
                 marginal_cost=network.generators.loc[solar, 'marginal_cost'],
                 capital_cost=costs.at['solar-rooftop', 'fixed'],
                 efficiency=network.generators.loc[solar, 'efficiency'],
                 p_max_pu=network.generators_t.p_max_pu[solar])

    network.add("Carrier", "home battery")

    network.madd("Bus",
                 nodes + " home battery",
                 carrier="home battery")

    network.madd("Store",
                 nodes + " home battery",
                 bus=nodes + " home battery",
                 e_cyclic=True,
                 e_nom_extendable=True,
                 carrier="home battery",
                 capital_cost=costs.at['battery storage', 'fixed'])

    network.madd("Link",
                 nodes + " home battery charger",
                 bus0=nodes + " low voltage",
                 bus1=nodes + " home battery",
                 carrier="home battery charger",
                 efficiency=costs.at['battery inverter', 'efficiency']**0.5,
                 capital_cost=costs.at['battery inverter', 'fixed'],
                 p_nom_extendable=True)

    network.madd("Link",
                 nodes + " home battery discharger",
                 bus0=nodes + " home battery",
                 bus1=nodes + " low voltage",
                 carrier="home battery discharger",
                 efficiency=costs.at['battery inverter', 'efficiency']**0.5,
                 marginal_cost=options['marginal_cost_storage'],
                 p_nom_extendable=True)


def create_network_topology(n, prefix):
    """
    create a network topology as the electric network,
    returns a pandas dataframe with bus0, bus1 and length
    """

    topo = pd.DataFrame(columns=["bus0", "bus1", "length"])
    connector = " -> "
    attrs = ["bus0", "bus1", "length"]

    candidates = pd.concat([n.lines[attrs],
                            n.links.loc[n.links.carrier == "DC", attrs]])

    positive_order = candidates.bus0 < candidates.bus1
    candidates_p = candidates[positive_order]
    candidates_n = (candidates[~ positive_order]
                    .rename(columns={"bus0": "bus1", "bus1": "bus0"}))
    candidates = pd.concat((candidates_p, candidates_n), sort=False)

    topo = candidates.groupby(["bus0", "bus1"], as_index=False).mean()
    topo.rename(index=lambda x: prefix + topo.at[x, "bus0"]
                + connector + topo.at[x, "bus1"],
                inplace=True)
    return topo


def remove_elec_base_techs(n):
    """
    remove conventional generators (e.g. OCGT) and storage units
    (e.g. batteries and H2) from base electricity-only network,
    since they're added here differently using links
    """
    to_keep = {"generators": snakemake.config["plotting"]["vre_techs"],
               "storage_units": snakemake.config["plotting"]["renewable_storage_techs"]}

    n.carriers = n.carriers.loc[to_keep["generators"] +
                                to_keep["storage_units"]]

    for components, techs in iteritems(to_keep):
        df = getattr(n, components)
        to_remove = df.carrier.value_counts().index ^ techs
        print("removing {} with carrier {}".format(components, to_remove))
        df.drop(df.index[df.carrier.isin(to_remove)], inplace=True)


def update_elec_costs(n, costs):
    """update the old cost assumptions from pypsa-eur to the new ones,
    this function keeps the old DC line costs and the old wind costs"""

    print("updating old pypsa-eur cost assumptions")

    for c in n.iterate_components(n.one_port_components):
        if c.name != "Load":
            print(c.name)
            cost_to_replace = costs  # .loc[~costs.index.str.contains("wind")]
            cap_dict = cost_to_replace["fixed"].to_dict()
            vom_dict = (
                cost_to_replace["VOM"] +
                cost_to_replace["fuel"] /
                cost_to_replace["efficiency"]).to_dict()
            eff_dict = cost_to_replace["efficiency"].to_dict()
#            c.df["capital_cost"] = c.df["carrier"].map(
#                cap_dict).combine_first(c.df["capital_cost"])
            c.df["marginal_cost"] = c.df["carrier"].map(
                vom_dict).combine_first(c.df["marginal_cost"])
            if c.name == "Generator":
                c.df["efficiency"] = c.df["carrier"].map(
                    eff_dict).combine_first(c.df["efficiency"])

    n.generators.loc[n.generators.carrier=="solar", "capital_cost"] = costs.loc[
        "solar", "fixed"]

#    n = attach_wind_costs(n, costs)


def add_co2_tracking(n):

    # minus sign because opposite to how fossil fuels used:
    # CH4 burning puts CH4 down, atmosphere up
    n.add("Carrier", "co2",
          co2_emissions=-1.)

    # this tracks CO2 in the atmosphere
    n.add("Bus", "co2 atmosphere",
          carrier="co2")

    # NB: can also be negative
    n.madd("Store", ["co2 atmosphere"],
           e_nom_extendable=True,
           e_min_pu=-1,
           carrier="co2",
           bus="co2 atmosphere")

    # this tracks CO2 stored, e.g. underground
    n.add("Bus", "co2 stored",
          carrier="co2 stored")

    # TODO move cost to data/costs.csv
    # TODO move maximum somewhere more transparent
    n.madd("Store", ["co2 stored"],
           e_nom_extendable=True,
           e_nom_max=2e8,  # 1e6
           capital_cost=20.,
           # e_cyclic=True,
           carrier="co2 stored",
           bus="co2 stored")


    if options['co2_vent']:
        n.madd("Link", ["co2 vent"],
               bus0="co2 stored",
               bus1="co2 atmosphere",
               carrier="co2 vent",
               efficiency=1.,
               p_nom_extendable=True)

    if options['dac']:
        # direct air capture consumes electricity to take CO2 from the air to the            underground store
        # TODO do with cost from Breyer - later use elec and heat and capital
        # cost
        n.madd("Link", ["DAC"],
               bus0="co2 atmosphere",
               bus1="co2 stored",
               carrier="DAC",
               marginal_cost=100.,
               efficiency=1.,
               p_nom_extendable=True)


def add_co2limit(n, Nyears=1., limit=0.):

    cts = pop_layout.ct.value_counts().index

    co2_limit = co2_totals.loc[cts, "electricity"].sum()

    if "T" in opts:
        co2_limit += co2_totals.loc[cts,
                                    [i + " non-elec" for i in ["rail",
                                                               "road"]]].sum().sum()
    if "H" in opts:
        co2_limit += co2_totals.loc[cts,
                                    [i + " non-elec" for i in ["residential",
                                                               "services"]]].sum().sum()
    if "I" in opts:
        co2_limit += co2_totals.loc[cts,
                                    ["industrial non-elec",
                                     "industrial processes",
                                     "domestic aviation",
                                     "international aviation",
                                     "domestic navigation",
                                     "international navigation"]].sum().sum()

    co2_limit *= limit * Nyears

    n.add("GlobalConstraint", "CO2Limit",
          carrier_attribute="co2_emissions", sense="<=",
          constant=co2_limit)


def add_emission_prices(n, emission_prices=None, exclude_co2=False):
    assert False, "Needs to be fixed, adds NAN"

    if emission_prices is None:
        emission_prices = snakemake.config['costs']['emission_prices']
    if exclude_co2:
        emission_prices.pop('co2')
    ep = (pd.Series(emission_prices).rename(lambda x: x +'_emissions')
          * n.carriers).sum(axis=1)
    n.generators['marginal_cost'] += n.generators.carrier.map(ep)
    n.storage_units['marginal_cost'] += n.storage_units.carrier.map(ep)


def set_line_s_max_pu(n):
    """
    set n-1 security margin to 0.5 for 37 clusters and to 0.7
    200 clusters 128 reproduces 98% of line volume in TWkm,
    but clustering distortions inside node
    """
    n_clusters = len(n.buses.index[n.buses.carrier == "AC"])
    s_max_pu = np.clip(0.5 + 0.2 * (n_clusters - 37) / (200 - 37), 0.5, 0.7)
    n.lines['s_max_pu'] = s_max_pu

    dc_b = n.links.carrier == 'DC'
    n.links.loc[dc_b, 'p_max_pu'] = snakemake.config['links']['p_max_pu']
    n.links.loc[dc_b, 'p_min_pu'] = - snakemake.config['links']['p_max_pu']


def set_line_volume_limit(n, lv):

    dc_b = n.links.carrier == 'DC'

    if lv != "opt":
        lv = float(lv)

        #  Either line_volume cap or cost
        n.lines['capital_cost'] = 0.
        n.links.loc[dc_b, 'capital_cost'] = 0.
    else:
        n.lines['capital_cost'] = (n.lines['length'] *
                                   costs.at['HVAC overhead', 'fixed'])

        # add HVDC inverter post factor, to maintain consistency with LV limit
        n.links.loc[dc_b, 'capital_cost'] = (n.links.loc[dc_b, 'length'] *
                                             costs.at['HVDC overhead', 'fixed'])
        # costs.at['HVDC inverter pair', 'fixed'])

    if lv != 1.0:
        lines_s_nom = n.lines.s_nom.where(n.lines.type == '',
                                          np.sqrt(3) * n.lines.num_parallel *
                                          n.lines.type.map(n.line_types.i_nom) *
                                          n.lines.bus0.map(n.buses.v_nom))

        n.lines['s_nom_min'] = lines_s_nom
        n.links.loc[dc_b, 'p_nom_min'] = n.links['p_nom']

        n.lines['s_nom_extendable'] = True
        n.links.loc[dc_b, 'p_nom_extendable'] = True

        if lv != "opt":
            n.line_volume_limit = lv * ((lines_s_nom * n.lines['length']).sum() +
                                        n.links.loc[dc_b].eval('p_nom * length').sum())

    return n


def average_every_nhours(n, offset):
    logger.info('Resampling the network to {}'.format(offset))
    m = n.copy(with_time=False)

    # fix copying of network attributes
    # copied from pypsa/io.py, should be in pypsa/components.py# Network.copy()
    allowed_types = (float, int, bool) + string_types + \
        tuple(np.typeDict.values())
    attrs = dict((attr, getattr(n, attr))
                 for attr in dir(n)
                 if (not attr.startswith("__") and
                     isinstance(getattr(n, attr), allowed_types)))
    for k, v in iteritems(attrs):
        setattr(m, k, v)

    snapshot_weightings = n.snapshot_weightings.resample(offset).sum()
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings

    for c in n.iterate_components():
        pnl = getattr(m, c.list_name + "_t")
        for k, df in iteritems(c.pnl):
            if not df.empty:
                if c.list_name == "stores" and k == "e_max_pu":
                    pnl[k] = df.resample(offset).min()
                elif c.list_name == "stores" and k == "e_min_pu":
                    pnl[k] = df.resample(offset).max()
                else:
                    pnl[k] = df.resample(offset).mean()

    return m


def generate_periodic_profiles(
    dt_index=pd.date_range(
        "2011-01-01 00:00",
        "2011-12-31 23:00",
        freq="H", tz="UTC"),
        nodes=[], weekly_profile=range(24 * 7)):
    """Give a 24*7 long list of weekly hourly profiles, generate this for
       each country for the period dt_index, taking account of time
       zones and Summer Time.

    """
    weekly_profile = pd.Series(weekly_profile, range(24 * 7))

    week_df = pd.DataFrame(index=dt_index, columns=nodes)

    for ct in nodes:
        week_df[ct] = [24 * dt.weekday() + dt.hour for dt in dt_index.tz_convert(
            pytz.timezone(timezone_mappings[ct[:2]]))]
        week_df[ct] = week_df[ct].map(weekly_profile)

    return week_df


def shift_df(df, hours=1):
    """Works both on Series and DataFrame"""
    df = df.copy()
    df.values[:] = np.concatenate([df.values[-hours:],
                                   df.values[:-hours]])
    return df


def transport_degree_factor(temperature, deadband_lower=15, deadband_upper=20,
                            lower_degree_factor=0.5,
                            upper_degree_factor=1.6):
    """Work out how much energy demand in vehicles increases due to heating and cooling.
    There is a deadband where there is no increase.
    Degree factors are % increase in demand compared to no heating/cooling fuel consumption.
    Returns per unit increase in demand for each place and time
    """

    dd = temperature.copy()

    dd[(temperature > deadband_lower) & (temperature < deadband_upper)] = 0.

    dd[temperature < deadband_lower] = lower_degree_factor / 100. * \
        (deadband_lower - temperature[temperature < deadband_lower])

    dd[temperature > deadband_upper] = upper_degree_factor / 100. * \
        (temperature[temperature > deadband_upper] - deadband_upper)

    return dd


def prepare_data(network):
    # # # # # # # # # # # # # #
    # Heating
    # # # # # # # # # # # # # #

    ashp_cop = xr.open_dataarray(
        snakemake.input.cop_air_total).T.to_pandas().reindex(
        index=network.snapshots)
    gshp_cop = xr.open_dataarray(
        snakemake.input.cop_soil_total).T.to_pandas().reindex(
        index=network.snapshots)

    solar_thermal = xr.open_dataarray(
        snakemake.input.solar_thermal_total).T.to_pandas().reindex(
        index=network.snapshots)
    # 1e3 converts from W/m^2 to MW/(1000m^2) = kW/m^2
    solar_thermal = options['solar_cf_correction'] * solar_thermal / 1e3

    energy_totals = pd.read_csv(
        snakemake.input.energy_totals_name,
        index_col=0)

    nodal_energy_totals = energy_totals.loc[pop_layout.ct].fillna(0.)
    nodal_energy_totals.index = pop_layout.index
    # district heat share not weighted by population
    dist_heat_share = round(
        nodal_energy_totals["district heat share"],
        ndigits=2)
    nodal_energy_totals = nodal_energy_totals.multiply(
        pop_layout.fraction, axis=0)

    # copy forward the daily average heat demand into each hour, so it can be
    # multipled by the intraday profile
    daily_space_heat_demand = xr.open_dataarray(
        snakemake.input.heat_demand_total).T.to_pandas().reindex(
        index=network.snapshots, method="ffill")

    intraday_profiles = pd.read_csv(snakemake.input.heat_profile, index_col=0)

    sectors = ["residential", "services"]
    uses = ["water", "space"]

    heat_demand = {}
    electric_heat_supply = {}
    for sector in sectors:
        for use in uses:
            intraday_year_profile = generate_periodic_profiles(daily_space_heat_demand.index.tz_localize("UTC"), nodes=daily_space_heat_demand.columns, weekly_profile=(
                list(intraday_profiles["{} {} weekday".format(sector, use)]) * 5 + list(intraday_profiles["{} {} weekend".format(sector, use)]) * 2)).tz_localize(None)

            if use == "space":
                heat_demand_shape = daily_space_heat_demand * intraday_year_profile
                factor = options['space_heating_fraction']
            else:
                heat_demand_shape = intraday_year_profile
                factor = 1.

            heat_demand["{} {}".format(sector, use)] = factor * (heat_demand_shape / heat_demand_shape.sum(
            )).multiply(nodal_energy_totals["total {} {}".format(sector, use)]) * 1e6
            electric_heat_supply["{} {}".format(sector, use)] = (heat_demand_shape / heat_demand_shape.sum(
            )).multiply(nodal_energy_totals["electricity {} {}".format(sector, use)]) * 1e6

    heat_demand = pd.concat(heat_demand, axis=1)
    electric_heat_supply = pd.concat(electric_heat_supply, axis=1)

    # subtract from electricity load since heat demand already in heat_demand
    electric_nodes = n.loads.index[n.loads.carrier == "electricity"]
    n.loads_t.p_set[electric_nodes] = n.loads_t.p_set[electric_nodes] - \
        electric_heat_supply.groupby(level=1, axis=1).sum()[electric_nodes]

    # # # # # # # # # # # # # #
    # Transport
    # # # # # # # # # # # # # #

    # #  Get overall demand curve for all vehicles

#     dir_name = "data/emobility/"
    traffic = pd.read_csv(
        snakemake.input.traffic_data +
        "KFZ__count",
        skiprows=2)["count"]

    # Generate profiles
    transport_shape = generate_periodic_profiles(
        dt_index=network.snapshots.tz_localize("UTC"),
        nodes=pop_layout.index,
        weekly_profile=traffic.values).tz_localize(None)
    transport_shape = transport_shape / transport_shape.sum()

    transport_data = pd.read_csv(snakemake.input.transport_name,
                                 index_col=0)

    nodal_transport_data = transport_data.loc[pop_layout.ct].fillna(0.)
    nodal_transport_data.index = pop_layout.index
    nodal_transport_data["number cars"] = pop_layout["fraction"] * \
        nodal_transport_data["number cars"]
    nodal_transport_data.loc[nodal_transport_data["average fuel efficiency"] == 0.,
                             "average fuel efficiency"] = transport_data["average fuel efficiency"].mean()

    # electric motors are more efficient, so alter transport demand

    # kWh/km from EPA https://www.fueleconomy.gov/feg/ for Tesla Model S
    plug_to_wheels_eta = 0.20
    battery_to_wheels_eta = plug_to_wheels_eta * 0.9

    efficiency_gain = nodal_transport_data["average fuel efficiency"] / \
        battery_to_wheels_eta

    # get heating demand for correction to demand time series
    temperature = xr.open_dataarray(
        snakemake.input.temp_air_total).T.to_pandas()

    # correction factors for vehicle heating
    dd_ICE = transport_degree_factor(
        temperature,
        options['transport_heating_deadband_lower'],
        options['transport_heating_deadband_upper'],
        options['ICE_lower_degree_factor'],
        options['ICE_upper_degree_factor'])

    dd_EV = transport_degree_factor(
        temperature,
        options['transport_heating_deadband_lower'],
        options['transport_heating_deadband_upper'],
        options['EV_lower_degree_factor'],
        options['EV_upper_degree_factor'])

    # divide out the heating/cooling demand from ICE totals
    ICE_correction = (transport_shape * (1 + dd_ICE)
                      ).sum() / transport_shape.sum()

    transport = (
        transport_shape.multiply(
            nodal_energy_totals["total road"] +
            nodal_energy_totals["total rail"] -
            nodal_energy_totals["electricity rail"]) *
        1e6 *
        Nyears).divide(
            efficiency_gain *
            ICE_correction)

    # multiply back in the heating/cooling demand for EVs
    transport = transport.multiply(1 + dd_EV)

    # #  derive plugged-in availability for PKW's (cars)

    traffic = pd.read_csv(
        snakemake.input.traffic_data +
        "Pkw__count",
        skiprows=2)["count"]

    avail_max = 0.95

    avail_mean = 0.8

    avail = avail_max - (avail_max - avail_mean) * (traffic - \
                         traffic.min()) / (traffic.mean() - traffic.min())

    avail_profile = generate_periodic_profiles(
        dt_index=network.snapshots.tz_localize("UTC"),
        nodes=pop_layout.index,
        weekly_profile=avail.values).tz_localize(None)

    dsm_week = np.zeros((24 * 7,))

    dsm_week[(np.arange(0, 7, 1) * 24 + options['dsm_restriction_time'])
             ] = options['dsm_restriction_value']

    dsm_profile = generate_periodic_profiles(
        dt_index=network.snapshots.tz_localize("UTC"),
        nodes=pop_layout.index,
        weekly_profile=dsm_week).tz_localize(None)

    # # # # # # # # # # # # # # #
    # CO2
    # # # # # # # # # # # # # # #

    # 1e6 to convert Mt to tCO2
    co2_totals = 1e6 * \
        pd.read_csv(snakemake.input.co2_totals_name, index_col=0)

    return (nodal_energy_totals, heat_demand, ashp_cop, gshp_cop,
            solar_thermal, transport, avail_profile, dsm_profile,
            co2_totals, nodal_transport_data, dist_heat_share)


def convert_units(costs):
    """""
    converts costs units to MW and EUR,
    for the conversion of the water tanks from EUR/m³ -> EUR/MWh
    temperature difference dT 0f 40°C is assumed.
    *****************************************
        E = c_p * m * dT -> E/m = c_p * dT
        c_p = 4.2 kJ/(kg°C) specific heat capacity of water
        1J = 1/3600 Wh
        dT = 40°C
        E = energy
        m = mass (kg)
    *****************************************
    """
#    costs.dropna(inplace=True)
    costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
    costs.loc[costs.unit.str.contains("EUR/tCO2/a"), "value"] *= 8760
    costs.loc[costs.unit.str.contains(
        "USD"), "value"] *= snakemake.config['costs']['USD2013_to_EUR2013']
    costs.loc[costs.unit.str.contains("EUR/m3"), "value"] /= 4.2 / 3600 * 40
    # TODO add solar thermal unit converting up here
    return costs



def prepare_costs():

    cost_year = snakemake.config['costs']['year']
    map_missings = {"CO2 intensity": 0,
                    "FOM": 0,
                    "VOM": 0,
                    "discount rate": snakemake.config['costs']['discountrate'],
                    "efficiency": 1,
                    "fuel": 0,
                    "investment": 0,
                    "lifetime": 25}

    # set all asset costs and other parameters
    costs = pd.read_csv(
        snakemake.input.costs +
        "costs_{}.csv".format(cost_year),
        index_col=list(
            range(2))).sort_index()

    # convert units
    costs = convert_units(costs)

    costs = costs["value"].unstack(
        level=1).groupby(
        level="technology").sum(
            min_count=1)
    costs = costs.fillna(map_missings)

    costs["fixed"] = [(annuity(v["lifetime"], v["discount rate"])
                       + v["FOM"] / 100.)
                      * v["investment"] * Nyears
                      for i, v in costs.iterrows()]

    # assuming for solar 50% utility and 50% rooftop
    costs.loc["solar"] = (0.5 * costs.loc["solar-rooftop"] +
                          0.5 * costs.loc["solar-utility"])

    # ----------------------------------------------------------------------
    costs_old = pd.read_csv(snakemake.input.costs_old,
                            index_col=list(range(3))).sort_index()
    # convert units
    costs_old = convert_units(costs_old)

    costs_old = (costs_old.loc[idx[:, 2030, :], "value"].unstack(level=2)
                 .groupby(level="technology").sum(min_count=1))

    costs_old = costs_old.fillna(map_missings)

    costs_old["fixed"] = [(annuity(v["lifetime"], v["discount rate"])
                           + v["FOM"] / 100.)
                          * v["investment"] * Nyears
                          for i, v in costs_old.iterrows()]

    costs_old.rename({"hydrogen storage": "hydrogen storage tank",
                      "hydrogen underground storage": "hydrogen storage underground"},
                     axis=0, inplace=True)
    # ------------------------------------------------------------------------

    missing = costs_old.index.difference(costs.index)
    # retrofitting costs are calculated seperately
    missing = missing[~missing.str.contains("retrofitting")]
    not_used = ["biomass", "decentral CHP"]
    missing = missing.drop(not_used)

    if len(missing):
        print("************************************************************")
        print("warning, in new cost assumptions the following components: ")
        for i in range(len(missing)):
            print("    ", i + 1, missing[i])
        print(" are missing and the old cost assumptions are assumed.")
        print("************************************************************")

    costs = pd.concat([costs, costs_old.loc[missing]], sort=False)

    for col in costs.columns:
        costs[col].fillna(costs_old[col], inplace=True)

    if options["costs_old"]:
        print("old costs are assumed")
        costs = costs_old

    if options["h2_costs_old"]:
        h2_costs = ['electrolysis', 'fuel cell', 'hydrogen storage tank',
                    'hydrogen storage underground']
        print("old costs assumed for ", *h2_costs)
        print("------------------------------------------")
        costs = pd.concat([costs.loc[~costs.index.isin(h2_costs)],
                           costs_old.loc[h2_costs]], sort=False)

    return costs


def add_generation(network):
    print("adding electricity generation")
    nodes = pop_layout.index

#    conventionals = [("OCGT", "gas")]

    for generator, carrier in [("OCGT", "gas")]:
        network.add("Carrier",
                    carrier)

        network.add("Bus",
                    "EU " + carrier,
                    carrier=carrier)

        # use madd to get carrier inserted
        network.madd("Store",
                     ["EU " + carrier + " Store"],
                     bus=["EU " + carrier],
                     e_nom_extendable=True,
                     e_cyclic=True,
                     carrier=carrier,
                     capital_cost=0.)  # could correct to e.g. 0.2 EUR/kWh * annuity and O&M

        network.add("Generator",
                    "EU fossil " + carrier,
                    bus="EU " + carrier,
                    p_nom_extendable=True,
                    carrier=carrier,
                    capital_cost=0.,
                    marginal_cost=costs.at[carrier, 'fuel'])

        network.madd("Link",
                      nodes + " " + generator,
                      bus0=["EU " + carrier] * len(nodes),
                      bus1=nodes,
                      bus2="co2 atmosphere",
                      marginal_cost=costs.at[generator,
                                            'efficiency'] * costs.at[generator,
                                                                      'VOM'],
                      # NB: VOM is per MWel
                      # NB: fixed cost is per MWel
                      capital_cost=costs.at[generator,
                                            'efficiency'] * costs.at[generator,
                                                                    'fixed'],
                      p_nom_extendable=True,
                      carrier=generator,
                      efficiency=costs.at[generator, 'efficiency'],
                      efficiency2=costs.at[carrier, 'CO2 intensity'])


def add_wave(network, wave_cost_factor):
    wave_fn = "data/WindWaveWEC_GLTB.xlsx"

    locations = ["FirthForth", "Hebrides"]

    # in kW
    capacity = pd.Series([750, 1000, 600], ["Attenuator", "F2HB", "MultiPA"])

    # in EUR/MW
    costs = wave_cost_factor * \
        pd.Series([2.5, 2, 1.5], ["Attenuator", "F2HB", "MultiPA"]) * 1e6

    sheets = {}

    for l in locations:
        sheets[l] = pd.read_excel(wave_fn,
                                  index_col=0, skiprows=[0], parse_dates=True,
                                  sheet_name=l)

    to_drop = ["Vestas 3MW", "Vestas 8MW"]
    wave = pd.concat([sheets[l].drop(to_drop, axis=1).divide(
        capacity, axis=1) for l in locations], keys=locations, axis=1)

    for wave_type in costs.index:
        n.add("Generator",
              "Hebrides " + wave_type,
              bus="GB4 0",
              p_nom_extendable=True,
              carrier="wave",
              capital_cost=(annuity(25, 0.07) + 0.03) * costs[wave_type],
              p_max_pu=wave["Hebrides", wave_type])


def add_storage(network):
    print("adding electricity storage")
    nodes = pop_layout.index

    network.add("Carrier", "H2")

    network.madd("Bus",
                 nodes + " H2",
                 carrier="H2")

    network.madd("Bus",
                 nodes + " gas",
                 carrier="gas")

    network.madd("Store",
                 nodes + " gas Store",
                 bus=nodes + " gas",
                 e_nom_extendable=True,
                 e_cyclic=True,
                 carrier="gas",
                 capital_cost=0.)

    network.madd("Generator",
                 nodes + " fossil gas",
                 bus=nodes + " gas",
                 p_nom_extendable=True,
                 carrier="gas",
                 capital_cost=0.,
                 marginal_cost=78)

    network.madd("Link",
                 nodes + " H2 Electrolysis",
                 bus1=nodes + " H2",
                 bus0=nodes,
                 p_nom_extendable=True,
                 carrier="H2 Electrolysis",
                 efficiency=costs.at["electrolysis", "efficiency"],
                 capital_cost=(costs.at["electrolysis", "fixed"] *
                               costs.at["electrolysis", "efficiency"])
                 )

    network.madd("Link",
                 nodes + " H2 Fuel Cell",
                 bus0=nodes + " H2",
                 bus1=nodes,
                 p_nom_extendable=True,
                 carrier="H2 Fuel Cell",
                 efficiency=costs.at["fuel cell",
                                     "efficiency"],
                 capital_cost=(costs.at["fuel cell", "fixed"] *
                               costs.at["fuel cell", "efficiency"])
                 )   # NB: fixed cost is per MWel

    network.madd("Link",
                 nodes + " " + "OCGT",
                 bus0=nodes + " gas",
                 bus1=nodes,
                 bus2="co2 atmosphere",
                 marginal_cost=(costs.at['OCGT', 'efficiency'] *
                                costs.at['OCGT', 'VOM']),
                 capital_cost=(costs.at['OCGT', 'efficiency'] *
                               costs.at['OCGT', 'fixed']),
                 p_nom_extendable=True,
                 carrier='OCGT',
                 efficiency=costs.at['OCGT', 'efficiency'],
                 efficiency2=costs.at['gas', 'CO2 intensity'])

    cavern_nodes = pd.DataFrame()
    if options['hydrogen_underground_storage']:

        h2_salt_cavern_potential = pd.read_csv(snakemake.input.h2_cavern,
                                               index_col=0, skiprows=[0],
                                               names=["potential", "TWh"])
        h2_cavern_ct = h2_salt_cavern_potential[h2_salt_cavern_potential.potential]
        cavern_nodes = pop_layout[pop_layout.ct.isin(h2_cavern_ct.index)]
        # assumptions: weight storage potential in a country by population
        h2_pot = (h2_cavern_ct.loc[cavern_nodes.ct, "TWh"].astype(float)
                  .reset_index().set_index(cavern_nodes.index))
        h2_pot = h2_pot.TWh * cavern_nodes.fraction

        h2_capital_cost = costs.at["hydrogen storage underground", "fixed"]

        network.madd("Store",
                     cavern_nodes.index + " H2 Store",
                     bus=cavern_nodes.index + " H2",
                     e_nom_extendable=True,
                     #             e_nom_max=h2_pot.values,
                     #             type="underground",
                     e_cyclic=True,
                     carrier="H2 Store",
                     capital_cost=h2_capital_cost)

    # hydrogen stored not underground
    h2_capital_cost = costs.at["hydrogen storage tank", "fixed"]
    nodes_upper = nodes ^ cavern_nodes.index

    network.madd("Store",
                 nodes_upper + " H2 Store",
                 bus=nodes_upper + " H2",
                 e_nom_extendable=True,
                 e_cyclic=True,
                 #                 type="upperground",
                 carrier="H2 Store",
                 capital_cost=h2_capital_cost)

    h2_links = create_network_topology(n, "H2 pipeline ")

    # TODO Add efficiency losses
    network.madd("Link",
                 h2_links.index,
                 bus0=h2_links.bus0 + " H2",
                 bus1=h2_links.bus1 + " H2",
                 p_min_pu=-1,
                 p_nom_extendable=True,
                 length=h2_links.length.values,
                 capital_cost=costs.at['H2 pipeline',
                                       'fixed'] * h2_links.length.values,
                 carrier="H2 pipeline")

    gas_links = create_network_topology(n, "gas pipeline ")

    # TODO Add efficiency losses
    network.madd("Link",
                 gas_links.index,
                 bus0=gas_links.bus0 + " gas",
                 bus1=gas_links.bus1 + " gas",
                 p_min_pu=-1,
                 p_nom_extendable=True,
                 length=gas_links.length.values,
                 capital_cost=5000,
                 carrier="gas pipeline")

    network.add("Carrier", "battery")

    network.madd("Bus",
                 nodes + " battery",
                 carrier="battery")

    network.madd("Store",
                 nodes + " battery",
                 bus=nodes + " battery",
                 e_cyclic=True,
                 e_nom_extendable=True,
                 carrier="battery",
                 capital_cost=costs.at['battery storage', 'fixed'])

    network.madd("Link",
                 nodes + " battery charger",
                 bus0=nodes,
                 bus1=nodes + " battery",
                 carrier="battery charger",
                 efficiency=costs.at['battery inverter', 'efficiency']**0.5,
                 capital_cost=costs.at['battery inverter', 'fixed'],
                 p_nom_extendable=True)

    network.madd("Link",
                 nodes + " battery discharger",
                 bus0=nodes + " battery",
                 bus1=nodes,
                 carrier="battery discharger",
                 efficiency=costs.at['battery inverter', 'efficiency']**0.5,
                 marginal_cost=options['marginal_cost_storage'],
                 p_nom_extendable=True)

    if options['methanation']:
        network.madd("Link",
                     nodes + " Sabatier",
                     bus0=nodes + " H2",
                     bus1=nodes + " gas",
                     bus2="co2 stored",
                     p_nom_extendable=True,
                     carrier="Sabatier",
                     efficiency=costs.at["methanation", "efficiency"],
                     efficiency2=(-costs.at["methanation", "efficiency"] *
                                  costs.at['gas', 'CO2 intensity']),
                     capital_cost=costs.at["methanation", "fixed"])

    if options['helmeth']:
        network.madd("Link",
                     nodes + " helmeth",
                     bus0=nodes,
                     bus1=nodes + " gas",
                     bus2="co2 stored",
                     carrier="helmeth",
                     p_nom_extendable=True,
                     efficiency=costs.at["helmeth", "efficiency"],
                     efficiency2=(-costs.at["helmeth", "efficiency"] *
                                  costs.at['gas', 'CO2 intensity']),
                     capital_cost=costs.at["helmeth", "fixed"])

    if options['SMR']:
        network.madd("Link",
                     nodes + " SMR CCS",
                     bus0=nodes + " gas",
                     bus1=nodes + " H2",
                     bus2="co2 atmosphere",
                     bus3="co2 stored",
                     p_nom_extendable=True,
                     carrier="SMR CCS",
                     efficiency=costs.at["SMR CCS", "efficiency"],
                     efficiency2=(costs.at['gas', 'CO2 intensity'] *
                                  (1 - options["ccs_fraction"])),
                     efficiency3=(costs.at['gas', 'CO2 intensity'] *
                                  options["ccs_fraction"]),
                     capital_cost=costs.at["SMR CCS", "fixed"])

        network.madd("Link",
                     nodes + " SMR",
                     bus0=nodes + " gas",
                     bus1=nodes + " H2",
                     bus2="co2 atmosphere",
                     p_nom_extendable=True,
                     carrier="SMR",
                     efficiency=costs.at["SMR", "efficiency"],
                     efficiency2=costs.at['gas', 'CO2 intensity'],
                     capital_cost=costs.at["SMR", "fixed"])


def add_transport(network):
    print("adding transport")
    nodes = pop_layout.index

    network.add("Carrier", "Li ion")

    network.madd("Bus",
                 nodes,
                 suffix=" EV battery",
                 carrier="Li ion")

    network.madd("Load", nodes, suffix=" transport", bus=nodes +
                 " EV battery", carrier="transport", p_set=(1 -
                                                            options['transport_fuel_cell_share']) *
                 (transport[nodes] +
                  shift_df(transport[nodes], 1) +
                  shift_df(transport[nodes], 2)) /
                 3.)

    # 3-phase charger with 11 kW * x% of time grid-connected
    p_nom = nodal_transport_data["number cars"] * \
        0.011 * (1 - options['transport_fuel_cell_share'])

    network.madd("Link",
                 nodes,
                 suffix=" BEV charger",
                 bus0=nodes,
                 bus1=nodes + " EV battery",
                 p_nom=p_nom,
                 carrier="BEV charger",
                 p_max_pu=avail_profile[nodes],
                 efficiency=0.9,  # [B]
                 # These were set non-zero to find LU infeasibility when availability = 0.25
                 # p_nom_extendable=True,
                 # p_nom_min=p_nom,
                 # capital_cost=1e6,  # i.e. so high it only gets built where necessary
                 )

    if options["v2g"]:

        network.madd("Link",
                     nodes,
                     suffix=" V2G",
                     bus1=nodes,
                     bus0=nodes + " EV battery",
                     p_nom=p_nom,
                     carrier="V2G",
                     p_max_pu=avail_profile[nodes],
                     efficiency=0.9)  # [B]

    if options["bev"]:

        network.madd("Store",
                     nodes,
                     suffix=" battery storage",
                     bus=nodes + " EV battery",
                     carrier="battery storage",
                     e_cyclic=True,
                     e_nom=nodal_transport_data["number cars"] * 0.05 * options["bev_availability"] * (
                         1 - options['transport_fuel_cell_share']),
                     # 50 kWh battery
                     # http://www.zeit.de/mobilitaet/2014-10/auto-fahrzeug-bestand
                     e_max_pu=1,
                     e_min_pu=dsm_profile[nodes])

    if options['transport_fuel_cell_share'] != 0:

        network.madd("Load",
                     nodes,
                     suffix=" transport fuel cell",
                     bus=nodes + " H2",
                     carrier="transport fuel cell",
                     p_set=options['transport_fuel_cell_share'] / costs.at["fuel cell",
                                                                           "efficiency"] * transport[nodes])


def add_heat(network):
    print("adding heat")
    sectors = ["residential", "services"]

    # stores the different groups of nodes
    nodes = {}

    # rural are areas with low heating density and individual heating
    # urban are areas with high heating density
    # urban can be split into district heating (central) and individual
    # heating (decentral)
    # for central nodes, residential and services are aggregated
    urban_fraction = pop_layout["urban"] / \
                     (pop_layout[["urban", "rural"]].sum(axis=1))

    for sector in sectors:
        nodes[sector + " rural"] = pop_layout.index
        nodes[sector + " urban decentral"] = pop_layout.index

    if options["central"] and not options["central_real"]:
        #        urban_decentral_ct = pd.Index(["ES", "GR", "PT", "IT", "BG"])
        central_fraction = options['central_fraction']
#        urban_ct = pd.DataFrame(urban_fraction)
#        urban_ct["country"] =  urban_ct.index.str[:2]
        decentral_nodes = dist_heat_share[dist_heat_share == 0]
        dist_fraction = central_fraction * urban_fraction
        nodes["urban central"] = dist_fraction.index

    if options["central_real"]:  # take current district heating share
        dist_fraction = dist_heat_share * \
            pop_layout["urban_ct_fraction"] / pop_layout["fraction"]
        nodes["urban central"] = dist_fraction.index
        # if district heating share larger than urban fraction -> set urban
        # fraction to district heating share
        urban_fraction = pd.concat(
            [urban_fraction, dist_fraction], axis=1).max(axis=1)
        diff = urban_fraction - dist_fraction
        dist_fraction += diff * options["dh_strength"]
        print("************************************")
        print(
            "the current DH share compared to the maximum possible is increased \
               \n by a factor of ",
            options["dh_strength"],
            "resulting DH share: ",
            dist_fraction)
        print("**********************************")

    # NB: must add costs of central heating afterwards (EUR 400 / kWpeak, 50a,
    # 1% FOM from Fraunhofer ISE)

    if options["retrofitting_exogenous"]:
        print("natural renovation rate of ", options["retro_rate"] * 100,
              "% over ", options["years"], " years is assumed.")
        for sector in sectors:
            heat_demand[sector + " space"] = heat_demand[sector + " space"].apply(lambda x: space_heat_retro(
                x, options["years"], options["retro_rate"], options["dE"], option=options["retro_opt"]))

    heat_types = ["residential rural", "services rural",
                  "residential urban decentral", "services urban decentral",
                  "urban central"]
    for name in heat_types:

        name_type = "central" if name == "urban central" else "decentral"

        network.add("Carrier", name + " heat")

        network.madd("Bus",
                     nodes[name] + " " + name + " heat",
                     carrier=name + " heat")

        #  Add heat load
        for sector in sectors:
            if "rural" in name:
                factor = 1 - urban_fraction[nodes[name]]
            elif "urban central" in name:
                factor = dist_fraction[nodes[name]]
            elif "urban decentral" in name:
                factor = urban_fraction[nodes[name]] - \
                    dist_fraction[nodes[name]]
            else:
                factor = None

            if sector in name:
                heat_load = heat_demand[[sector + " water",
                                         sector + " space"]].groupby(level=1,
                                                                     axis=1).sum()[nodes[name]].multiply(factor)
        if name == "urban central":
            heat_load = heat_demand.groupby(level=1, axis=1).sum()[nodes[name]].multiply(
                factor * (1 + options['district_heating_loss']))

        # distribute heat demand over one year
        if options["base_load"]:
            print("heat demand is assumed as base load")
            base_load = heat_load.sum() / len(heat_load)
            heat_load = heat_load.apply(
                lambda x: base_load.loc[x.index], axis=1)

        network.madd("Load",
                     nodes[name],
                     suffix=" " + name + " heat",
                     bus=nodes[name] + " " + name + " heat",
                     carrier=name + " heat",
                     p_set=heat_load)

        # #  Add heat pumps

        heat_pump_type = "air" if "urban" in name else "ground"
        costs_name = "{} {}-sourced heat pump".format(
            name_type, heat_pump_type)
        cop = {"air": ashp_cop, "ground": gshp_cop}
        efficiency = cop[heat_pump_type][nodes[name]
                                         ] if options["time_dep_hp_cop"] else costs.at[costs_name, 'efficiency']

        network.madd("Link",
                     nodes[name],
                     suffix=" {} {} heat pump".format(name,
                                                      heat_pump_type),
                     bus0=nodes[name],
                     bus1=nodes[name] + " " + name + " heat",
                     carrier="{} {} heat pump".format(name,
                                                      heat_pump_type),
                     efficiency=efficiency,
                     capital_cost=costs.at[costs_name,
                                           'efficiency'] * costs.at[costs_name,
                                                                    'fixed'],
                     p_nom_extendable=True)

        if options["tes"]:

            network.add("Carrier", name + " water tanks")

            network.madd("Bus",
                         nodes[name] + " " + name + " water tanks",
                         carrier=name + " water tanks")

            network.madd("Link",
                         nodes[name] + " " + name + " water tanks charger",
                         bus0=nodes[name] + " " + name + " heat",
                         bus1=nodes[name] + " " + name + " water tanks",
                         efficiency=costs.at['water tank charger', 'efficiency'],
                         carrier=name + " water tanks charger",
                         p_nom_extendable=True)

            network.madd("Link",
                         nodes[name] + " " + name + " water tanks discharger",
                         bus0=nodes[name] + " " + name + " water tanks",
                         bus1=nodes[name] + " " + name + " heat",
                         carrier=name + " water tanks discharger",
                         efficiency=costs.at['water tank discharger',
                                             'efficiency'],
                         p_nom_extendable=True)

            #  [HP] 180 day time constant for centralised, 3 day for decentralised
            tes_time_constant_days = options["tes_tau"] if name_type == "decentral" else 180.

            network.madd("Store",
                         nodes[name] + " " + name + " water tanks",
                         bus=nodes[name] + " " + name + " water tanks",
                         e_cyclic=True,
                         e_nom_extendable=True,
                         carrier=name + " water tanks",
                         standing_loss=1 - np.exp(-1 / (24. * tes_time_constant_days)),
                         capital_cost=costs.at[name_type + ' water tank storage',
                                               'fixed'])

        if options["boilers"]:

            network.madd("Link",
                         nodes[name] + " " + name + " resistive heater",
                         bus0=nodes[name],
                         bus1=nodes[name] + " " + name + " heat",
                         carrier=name + " resistive heater",
                         efficiency=costs.at[name_type + ' resistive heater',
                                             'efficiency'],
                         capital_cost=costs.at[name_type + ' resistive heater',
                                               'efficiency'] * costs.at[name_type + ' resistive heater',
                                                                        'fixed'],
                         p_nom_extendable=True)
#            if name == "urban central":
            network.madd("Link",
                         nodes[name] + " " + name + " gas boiler",
                         p_nom_extendable=True,
                         bus0=nodes[name] + " gas",
                         bus1=nodes[name] + " " + name + " heat",
                         bus2="co2 atmosphere",
                         carrier=name + " gas boiler",
                         efficiency=costs.at[name_type + ' gas boiler',
                                             'efficiency'],
                         efficiency2=costs.at['gas',
                                              'CO2 intensity'],
                         capital_cost=costs.at[name_type + ' gas boiler',
                                               'efficiency'] * costs.at[name_type + ' gas boiler',
                                                                        'fixed'])

        if options["solar_thermal"]:

            network.add("Carrier", name + " solar thermal")

            network.madd("Generator",
                         nodes[name],
                         suffix=" " + name + " solar thermal collector",
                         bus=nodes[name] + " " + name + " heat",
                         carrier=name + " solar thermal",
                         p_nom_extendable=True,
                         capital_cost=costs.at[name_type + ' solar thermal',
                                               'fixed'],
                         p_max_pu=solar_thermal[nodes[name]])

        if options["chp"]:

            if name == "urban central":
                # add gas CHP; biomass CHP is added in biomass section
                network.madd("Link",
                             nodes[name] + " urban central gas CHP electric",
                             bus0=nodes[name] + " gas",
                             bus1=nodes[name],
                             bus2="co2 atmosphere",
                             carrier="urban central gas CHP electric",
                             p_nom_extendable=True,
                             capital_cost=(costs.at['central gas CHP', 'fixed']
                                           * costs.at['central gas CHP', 'efficiency']),
                             marginal_cost=costs.at['central gas CHP', 'VOM'],
                             efficiency=costs.at['central gas CHP', 'efficiency'],
                             efficiency2=costs.at['gas', 'CO2 intensity'],
                             c_b=costs.at['central gas CHP', 'c_b'],
                             c_v=options["chp_parameters"]["c_v"],
                             p_nom_ratio=costs.at['central gas CHP', 'p_nom_ratio'])

                network.madd("Link",
                             nodes[name] + " urban central gas CHP heat",
                             bus0=nodes[name] + " gas",
                             bus1=nodes[name] + " urban central heat",
                             bus2="co2 atmosphere",
                             carrier="urban central gas CHP heat",
                             p_nom_extendable=True,
                             marginal_cost=costs.at['central gas CHP',
                                                    'VOM'],
                             efficiency=(costs.at['central gas CHP',
                                                  'efficiency'] / options["chp_parameters"]["c_v"]),
                             efficiency2=costs.at['gas',
                                                  'CO2 intensity'])

                network.madd("Link",
                             nodes[name] + " urban central gas CHP CCS electric",
                             bus0=nodes[name] + " gas",
                             bus1=nodes[name],
                             bus2="co2 atmosphere",
                             bus3="co2 stored",
                             carrier="urban central gas CHP CCS electric",
                             p_nom_extendable=True,
                             capital_cost=(costs.at['central gas CHP CCS', 'fixed']
                                           * costs.at['central gas CHP CCS', 'efficiency']),
                             marginal_cost=costs.at['central gas CHP CCS', 'VOM'],
                             efficiency=costs.at['central gas CHP CCS', 'efficiency'],
                             efficiency2=costs.at['gas', 'CO2 intensity'] * (1 - options["ccs_fraction"]),
                             efficiency3=costs.at['gas', 'CO2 intensity'] * options["ccs_fraction"],
                             c_b=costs.at['central gas CHP CCS', 'c_b'],
                             c_v=options["chp_parameters"]["c_v"],
                             p_nom_ratio=costs.at['central gas CHP CCS', 'p_nom_ratio'])

                network.madd("Link",
                             nodes[name] + " urban central gas CHP CCS heat",
                             bus0=nodes[name] + " gas",
                             bus1=nodes[name] + " urban central heat",
                             bus2="co2 atmosphere",
                             bus3="co2 stored",
                             carrier="urban central gas CHP CCS heat",
                             p_nom_extendable=True,
                             marginal_cost=costs.at['central gas CHP CCS',
                                                    'VOM'],
                             efficiency=(costs.at['central gas CHP CCS',
                                                  'efficiency'] / options["chp_parameters"]["c_v"]),
                             efficiency2=costs.at['gas',
                                                  'CO2 intensity'] * (1 - options["ccs_fraction"]),
                             efficiency3=costs.at['gas',
                                                  'CO2 intensity'] * options["ccs_fraction"])

            else:
                network.madd("Link",
                             nodes[name] + " " + name + " micro gas CHP",
                             p_nom_extendable=True,
                             bus0=nodes[name] + " gas",
                             bus1=nodes[name],
                             bus2=nodes[name] + " " + name + " heat",
                             bus3="co2 atmosphere",
                             carrier=name + " micro gas CHP",
                             efficiency=costs.at['micro CHP', 'efficiency'],
                             efficiency2=costs.at['micro CHP', 'efficiency-heat'],
                             efficiency3=costs.at['gas', 'CO2 intensity'],
                             capital_cost=costs.at['micro CHP', 'fixed'])

    if options['retrofitting']:

        print("adding retrofitting")
        # resample heat demand to not overestimate retrofitting
        heat_demand_r =  heat_demand.resample(opts[1]).mean()
        print("heat demand resampled")
        # get space heat demand
        space_heat_demand = pd.concat([heat_demand_r["residential space"],
                                       heat_demand_r["services space"]],
                                      axis=1)

        res = {}
        retro_cost = pd.read_csv(snakemake.input.retro_cost_energy,
                                 index_col=[0, 1], skipinitialspace=True,
                                 header=[0, 1])
        floor_area = pd.read_csv(snakemake.input.floor_area, index_col=[0, 1])

        index = pd.MultiIndex.from_product([pop_layout.index, sectors + ["tot"]])
        square_metres = pd.DataFrame(np.nan, index=index, columns=["m²"])

        # weighting for share of space heat demand
        w_space = {}
        for sector in sectors:
            w_space[sector] = heat_demand_r[sector + " space"] / \
                (heat_demand_r[sector + " space"] + heat_demand_r[sector + " water"])
        w_space["tot"] = ((heat_demand_r["services space"] +
                           heat_demand_r["residential space"]) /
                           heat_demand_r.groupby(level=[1], axis=1).sum())

        network.add("Carrier", "retrofitting")

        for node in list(heat_demand.columns.levels[1]):
            retro_nodes = pd.Index([node])
            space_heat_demand_node = space_heat_demand[retro_nodes]
            space_heat_demand_node.columns = sectors
            ct = node[:2]
            if ct in floor_area.index.levels[0]:
                square_metres = (pop_layout.loc[node].fraction
                                 * floor_area.loc[ct, "value"] * 10**6)
                for carrier in heat_types:
                    name = node + " " + carrier + " heat"
                    if (name in list(network.loads_t.p_set.columns)):

                        if "urban central" in carrier:
                            f = dist_fraction[node]
                        elif "urban decentral" in carrier:
                            f = urban_fraction[node] - dist_fraction[node]
                        else:
                            f = 1 - urban_fraction[node]

                        if f == 0:
                            continue

                        if "residential" in carrier:
                            sec = "residential"
                        elif "services" in carrier:
                            sec = "services"
                        else:
                            sec = "tot"

                        square_metres_c = (square_metres.loc[sec] * f)
                        # weighting instead of taking space heat demand to
                        # allow simulatounsly exogenous and optimised
                        # retrofitting
                        demand = (network.loads_t.p_set[name].resample(opts[1])
                                  .mean())
                        space_heat_demand_c = demand * w_space[sec][node]
                        res[node+" "+carrier+" heat"] = space_heat_demand_c
                        space_peak_c = space_heat_demand_c.max()
                        if space_peak_c == 0:
                            continue
                        space_pu_c = (
                            space_heat_demand_c /
                            space_peak_c).to_frame(
                            name=node)

                        dE = retro_cost.loc[(ct, sec), ("dE")]
                        dE_diff = abs(dE.diff()).fillna(dE.iloc[0])
                        cost_c = retro_cost.loc[(ct, sec), ("cost")]
                        capital_cost = cost_c * square_metres_c / \
                            ((1 - dE) * space_peak_c)
                        steps = retro_cost.cost.columns
                        if (capital_cost.diff() < 0).sum():
                            print(
                                "warning, costs are not linear for ", ct, " ", sec)
                            s = capital_cost[(capital_cost.diff() < 0)].index
                            steps = steps.drop(s)

                        space_pu_c = (space_pu_c.reindex(index=heat_demand.index)
                                      .fillna(method="ffill"))
                        for strength in steps:
                            network.madd(
                                'Generator',
                                retro_nodes,
                                suffix=' retrofitting ' + strength + " " + carrier,
                                bus=node + " " + carrier + " heat",
                                strength=' retrofitting ' + strength,
                                type=carrier,
                                carrier="retrofitting",
                                p_nom_extendable=True,
                                p_nom_max=(
                                    1 - dE_diff[strength]) * space_peak_c,
                                dE=dE_diff[strength],
                                p_max_pu=space_pu_c,
                                p_min_pu=space_pu_c,
                                country=ct,
                                capital_cost=capital_cost[strength])

            else:
                print("no retrofitting data for ", ct,
                      " the country is skipped.")


def add_biomass(network):

    print("adding biomass")

    nodes = pop_layout.index

    # biomass distributed at country level - i.e. transport within country
    # allowed
    cts = pop_layout.ct.value_counts().index

    biomass_potentials = pd.read_csv(snakemake.input.biomass_potentials,
                                     index_col=0)

    # costs for biomass transport
    transport_costs = pd.read_csv(snakemake.input.biomass_transport,
                                  index_col=0)

    # potential per node
    biomass_pot_node = (biomass_potentials.loc[pop_layout.ct]
                        .set_index(pop_layout.index)
                        .mul(pop_layout.fraction, axis="index"))

    network.add("Carrier", "biogas")
    network.add("Carrier", "solid biomass")

    network.madd("Bus",
                 biomass_pot_node.index + " biogas",
                 carrier="biogas")

    network.madd("Bus",
                 biomass_pot_node.index + " solid biomass",
                 carrier="solid biomass")

    network.madd("Store",
                 biomass_pot_node.index +" biogas",
                 bus=biomass_pot_node.index + " biogas",
                 carrier="biogas",
                 e_nom=biomass_pot_node["biogas"].values,
                 marginal_cost=costs.at['biogas', 'fuel'],
                 e_initial=biomass_pot_node["biogas"].values)

    network.madd("Store",
                 biomass_pot_node.index + " solid biomass",
                 bus=biomass_pot_node.index + " solid biomass",
                 carrier="solid biomass",
                 e_nom=biomass_pot_node["solid biomass"].values,
                 #                 e_nom_extendable = True,
                 marginal_cost=costs.at['solid biomass', 'fuel'],
                 e_initial=biomass_pot_node["solid biomass"].values)

    network.madd("Link",
                 nodes + " biogas to gas",
                 bus0=biomass_pot_node.index + " biogas",
                 bus1=biomass_pot_node.index + " gas",
                 bus2="co2 atmosphere",
                 carrier="biogas to gas",
                 efficiency2=-costs.at['gas', 'CO2 intensity'],
                 capital_cost=costs.loc["biogas upgrading", "fixed"],
                 marginal_cost=costs.loc["biogas upgrading", "VOM"],
                 p_nom_extendable=True)


    # add biomass transport
    biomass_transport = create_network_topology(n, "Biomass transport ")
    # make transport in both directions
    df = biomass_transport.copy()
    df["bus1"] = biomass_transport.bus0
    df["bus0"] = biomass_transport.bus1
    df.rename(index=lambda x: "Biomass transport " + df.at[x, "bus0"]
              + " -> " + df.at[x, "bus1"],
              inplace=True)
    biomass_transport = pd.concat([biomass_transport, df])

    # costs
    bus0_costs = biomass_transport.bus0.apply(
        lambda x: transport_costs.loc[x[:2]])
    bus1_costs = biomass_transport.bus1.apply(
        lambda x: transport_costs.loc[x[:2]])
    biomass_transport["costs"] = pd.concat(
        [bus0_costs, bus1_costs], axis=1).mean(axis=1)
#
    network.madd("Link",
                 biomass_transport.index,
                 bus0=biomass_transport.bus0 + " solid biomass",
                 bus1=biomass_transport.bus1 + " solid biomass",
                 #                 p_min_pu=-1,
                 p_nom_extendable=True,
                 length=biomass_transport.length.values,
                 marginal_cost=biomass_transport.costs * biomass_transport.length.values,
                 capital_cost=1,
                 carrier="solid biomass transport")

    # AC buses with district heating
    urban_central = n.buses.index[n.buses.carrier == "urban central heat"]
    if not urban_central.empty and options["chp"]:
        urban_central = urban_central.str[:-len(" urban central heat")]

        network.madd("Link",
                     urban_central + " urban central solid biomass CHP electric",
                     bus0=urban_central + " solid biomass",
                     bus1=urban_central,
                     carrier="urban central solid biomass CHP electric",
                     p_nom_extendable=True,
                     capital_cost=(costs.at['central solid biomass CHP', 'fixed']
                                   * costs.at['central solid biomass CHP', 'efficiency']),
                     marginal_cost=costs.at['central solid biomass CHP', 'VOM'],
                     efficiency=costs.at['central solid biomass CHP', 'efficiency'],
                     c_b=costs.at['central solid biomass CHP', 'c_b'],
                     c_v=options["chp_parameters"]["c_v"],
                     p_nom_ratio=costs.at['central solid biomass CHP', 'p_nom_ratio'])

        network.madd("Link",
                     urban_central + " urban central solid biomass CHP heat",
                     bus0=urban_central + " solid biomass",
                     bus1=urban_central + " urban central heat",
                     carrier="urban central solid biomass CHP heat",
                     p_nom_extendable=True,
                     marginal_cost=costs.at['central solid biomass CHP',
                                            'VOM'],
                     efficiency=(costs.at['central solid biomass CHP',
                                          'efficiency'] / options["chp_parameters"]["c_v"]))

        network.madd("Link",
                     urban_central + " urban central solid biomass CHP CCS electric",
                     bus0=urban_central + " solid biomass",
                     bus1=urban_central,
                     bus2="co2 atmosphere",
                     bus3="co2 stored",
                     carrier="urban central solid biomass CHP CCS electric",
                     p_nom_extendable=True,
                     capital_cost=(costs.at['central solid biomass CHP CCS', 'fixed']
                                   * costs.at['central solid biomass CHP CCS', 'efficiency']),
                     marginal_cost=costs.at['central solid biomass CHP CCS', 'VOM'],
                     efficiency=costs.at['central solid biomass CHP CCS', 'efficiency'],
                     efficiency2=-costs.at['solid biomass', 'CO2 intensity'] * options["ccs_fraction"],
                     efficiency3=costs.at['solid biomass', 'CO2 intensity'] * options["ccs_fraction"],
                     c_b=costs.at['central solid biomass CHP', 'c_b'],
                     c_v=options["chp_parameters"]["c_v"],
                     p_nom_ratio=costs.at['central solid biomass CHP', 'p_nom_ratio'])

        network.madd("Link",
                     urban_central + " urban central solid biomass CHP CCS heat",
                     bus0=urban_central + " solid biomass",
                     bus1=urban_central + " urban central heat",
                     bus2="co2 atmosphere",
                     bus3="co2 stored",
                     carrier="urban central solid biomass CHP CCS heat",
                     p_nom_extendable=True,
                     marginal_cost=costs.at['central solid biomass CHP CCS',
                                            'VOM'],
                     efficiency=(costs.at['central solid biomass CHP CCS',
                                          'efficiency'] / options["chp_parameters"]["c_v"]),
                     efficiency2=-costs.at['solid biomass',
                                           'CO2 intensity'] * options["ccs_fraction"],
                     efficiency3=costs.at['solid biomass',
                                          'CO2 intensity'] * options["ccs_fraction"])


def add_industry(network):

    print("adding industrial demand")

    nodes = pop_layout.index

    # 1e6 to convert TWh to MWh
    industrial_demand = 1e6 * pd.read_csv(snakemake.input.industrial_demand,
                                          index_col=0)

    solid_biomass_by_country = (industrial_demand["solid biomass"]
                                .groupby(pop_layout.ct).sum())

    network.madd("Bus",
                 ["solid biomass for industry"],
                 carrier="solid biomass for industry")

    network.madd("Load",
                 ["solid biomass for industry"],
                 bus="solid biomass for industry",
                 carrier="solid biomass for industry",
                 p_set=solid_biomass_by_country.sum() / 8760.)

    network.madd("Link",
                 nodes + " solid biomass for industry",
                 bus0=nodes + " solid biomass",
                 bus1="solid biomass for industry",
                 carrier="solid biomass for industry",
                 p_nom_extendable=True,
                 efficiency=1.)

    network.madd("Link",
                 nodes + " solid biomass for industry CCS",
                 bus0=nodes + " solid biomass",
                 bus1="solid biomass for industry",
                 bus2="co2 atmosphere",
                 bus3="co2 stored",
                 carrier="solid biomass for industry CCS",
                 p_nom_extendable=True,
                 capital_cost=(costs.at["industry CCS", "fixed"] *
                               costs.at['solid biomass', 'CO2 intensity']),
                 efficiency=0.9,
                 efficiency2=(-costs.at['solid biomass', 'CO2 intensity']
                              * options["ccs_fraction"]),
                 efficiency3=(costs.at['solid biomass', 'CO2 intensity']
                              * options["ccs_fraction"]))

    network.madd("Bus",
                 ["gas for industry"],
                 carrier="gas for industry")

    network.madd("Load",
                 ["gas for industry"],
                 bus="gas for industry",
                 carrier="gas for industry",
                 p_set=industrial_demand.loc[nodes, "methane"].sum() / 8760.)

    network.madd("Link",
                 nodes + "gas for industry",
                 bus0=nodes + " gas",
                 bus1="gas for industry",
                 bus2="co2 atmosphere",
                 carrier="gas for industry",
                 p_nom_extendable=True,
                 efficiency=1.,
                 efficiency2=costs.at['gas', 'CO2 intensity'])

    network.madd("Link",
                 nodes + "gas for industry CCS",
                 bus0=nodes + " gas",
                 bus1="gas for industry",
                 bus2="co2 atmosphere",
                 bus3="co2 stored",
                 carrier="gas for industry CCS",
                 p_nom_extendable=True,
                 capital_cost=(costs.at["industry CCS", "fixed"]
                               * costs.at['gas', 'CO2 intensity']),
                 efficiency=0.9,
                 efficiency2=(costs.at['gas', 'CO2 intensity']
                              * (1 - options["ccs_fraction"])),
                 efficiency3=(costs.at['gas', 'CO2 intensity']
                              * options["ccs_fraction"]))

    network.madd("Load",
                 nodes,
                 suffix=" H2 for industry",
                 bus=nodes + " H2",
                 carrier="H2 for industry",
                 p_set=industrial_demand.loc[nodes, "hydrogen"] / 8760.)

    navigation = nodal_energy_totals.loc[nodes, ["total international navigation",
                "total domestic navigation"]].sum(axis=1) * 1e6 / 8760.
    navigation_load = navigation * (options['shipping_average_efficiency'] /
                                    costs.at["fuel cell", "efficiency"])

    network.madd("Load",
                  nodes,
                  suffix=" H2 for shipping",
                  bus=nodes + " H2",
                  carrier="H2 for shipping",
                  p_set=navigation_load)

    network.add("Bus",
                "Fischer-Tropsch",
                carrier="Fischer-Tropsch")

    # use madd to get carrier inserted
    network.madd("Store",
                  ["Fischer-Tropsch Store"],
                  bus="Fischer-Tropsch",
                  e_nom_extendable=True,
                  e_cyclic=True,
                  carrier="Fischer-Tropsch",
                  capital_cost=0.)  # could correct to e.g. 0.001 EUR/kWh * annuity and O&M

    network.add("Generator",
                "fossil oil",
                bus="Fischer-Tropsch",
                p_nom_extendable=True,
                carrier="oil",
                capital_cost=0.,
                marginal_cost=costs.at["oil", 'fuel'])

    network.madd("Link",
                 nodes + " Fischer-Tropsch",
                 bus0=nodes + " H2",
                 bus1="Fischer-Tropsch",
                 bus2="co2 stored",
                 carrier="Fischer-Tropsch",
                 efficiency=costs.at["Fischer-Tropsch", 'efficiency'],
                 capital_cost=costs.at["Fischer-Tropsch", 'fixed'],
                 efficiency2=(-costs.at["oil", 'CO2 intensity']
                              * costs.at["Fischer-Tropsch", 'efficiency']),
                 p_nom_extendable=True)

    network.madd("Load",
                 ["naphtha for industry"],
                 bus="Fischer-Tropsch",
                 carrier="naphtha for industry",
                 p_set=industrial_demand.loc[nodes, "naphtha"].sum() / 8760.)

    network.madd("Load",
                 ["kerosene for aviation"],
                 bus="Fischer-Tropsch",
                 carrier="kerosene for aviation",
                 p_set=navigation.sum())

    # NB: CO2 gets released again to atmosphere when plastics decay or
    # kerosene is burned except for the process emissions when naphtha is used
    # for petrochemicals, which can be captured with other industry process
    # emissions tco2 per hour

    co2 = (network.loads.loc[["naphtha for industry", "kerosene for aviation"],
                              "p_set"].sum()
            * costs.at["oil", 'CO2 intensity']
            - industrial_demand.loc[nodes, "process emission from feedstock"].sum()
            / 8760.)

    network.madd("Load",
                 ["Fischer-Tropsch emissions"],
                 bus="co2 atmosphere",
                 carrier="Fischer-Tropsch emissions",
                 p_set=-co2)

    network.madd("Load",
                 nodes,
                 suffix=" low-temperature heat for industry",
                 bus=[node + " urban central heat" if node +
                      " urban central heat" in network.buses.index else node +
                      " services urban decentral heat" for node in nodes],
                 carrier="low-temperature heat for industry",
                 p_set=(industrial_demand.loc[nodes, "low-temperature heat"]
                        / 8760.))

    network.madd("Load",
                 nodes,
                 suffix=" industry new electricity",
                 bus=nodes,
                 carrier="industry new electricity",
                 p_set=((industrial_demand.loc[nodes, "electricity"]
                         - industrial_demand.loc[nodes, "current electricity"])
                        / 8760.))

    network.madd("Bus",
                  ["process emissions"],
                  carrier="process emissions")

    # this should be process emissions fossil+feedstock
    # then need load on atmosphere for feedstock emissions that are currently
    # going to atmosphere via Link Fischer-Tropsch demand

    network.madd("Load",
                 ["process emissions"],
                 bus="process emissions",
                 carrier="process emissions",
                 p_set=(-industrial_demand.loc[nodes,
                                               ["process emission",
                                                "process emission from feedstock"]]
                        .sum(axis=1).sum() / 8760.))

    network.madd("Link",
                 ["process emissions"],
                 bus0="process emissions",
                 bus1="co2 atmosphere",
                 carrier="process emissions",
                 p_nom_extendable=True,
                 efficiency=1.)

    # # assume enough local waste heat for CCS
    network.madd("Link",
                 ["process emissions CCS"],
                 bus0="process emissions",
                 bus1="co2 atmosphere",
                 bus2="co2 stored",
                 carrier="process emissions CCS",
                 p_nom_extendable=True,
                 capital_cost=costs.at["industry CCS", "fixed"],
                 efficiency=(1 - options["ccs_fraction"]),
                 efficiency2=options["ccs_fraction"])


def add_waste_heat(network):

    print("adding possibility to use industrial and fuel cell waste heat in \
              district heating")

    # AC buses with district heating
    urban_central = n.buses.index[n.buses.carrier == "urban central heat"]
    if not urban_central.empty:
        urban_central = urban_central.str[:-len(" urban central heat")]
        if "I" in opts:
            print("use industry waste heat")
            if options['use_fischer_tropsch_waste_heat']:
                n.links.loc[urban_central + " Fischer-Tropsch",
                            "bus3"] = urban_central + " urban central heat"
                n.links.loc[urban_central + " Fischer-Tropsch", "efficiency3"] = 0.95 - \
                    n.links.loc[urban_central + " Fischer-Tropsch", "efficiency"]

        if options['use_fuel_cell_waste_heat']:
            print("use fuel cell waste heat")
            n.links.loc[urban_central + " H2 Fuel Cell",
                        "bus2"] = urban_central + " urban central heat"
            n.links.loc[urban_central + " H2 Fuel Cell", "efficiency2"] = 0.95 - \
                n.links.loc[urban_central + " H2 Fuel Cell", "efficiency"]


def restrict_technology_potential(n, tech, limit):
    print(
        "restricting potentials (p_nom_max) for {} to {} of technical potential".format(
            tech,
            limit))
    gens = n.generators.index[n.generators.carrier.str.contains(tech)]
    # beware if limit is 0 and p_nom_max is np.inf, 0*np.inf is nan
    n.generators.loc[gens, "p_nom_max"] *= limit


def decentral(n):
    n.lines.drop(n.lines.index, inplace=True)
    n.links.drop(n.links.index[n.links.carrier.isin(
        ["DC", "B2B"])], inplace=True)


def remove_h2_network(n):

    nodes = pop_layout.index

    n.links.drop(n.links.index[n.links.carrier.isin(
        ["H2 pipeline"])], inplace=True)

    n.stores.drop(["EU H2 Store"], inplace=True)

    if options['hydrogen_underground_storage']:
        h2_capital_cost = costs.at["hydrogen storage underground", "fixed"]
    else:
        h2_capital_cost = costs.at["hydrogen storage tank", "fixed"]

    # put back nodal H2 storage
    n.madd("Store",
           nodes + " H2 Store",
           bus=nodes + " H2",
           e_nom_extendable=True,
           e_cyclic=True,
           carrier="H2 Store",
           capital_cost=h2_capital_cost)


# %%
if __name__ == "__main__":
    #  Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils.snakemake import MockSnakemake
        os.chdir("/home/ws/bw0928/Dokumente/pypsa-eur-sec/scripts")
        snakemake = MockSnakemake(
            wildcards=dict(
                network='elec',
                simpl='',
                clusters='38',
                lv='2',
                opts='Co2L-3H',
                sector_opts="[Co2L0p0-24H-T-H-B-I]"),
            input=dict(
                network='../pypsa-eur/networks/{network}_s{simpl}_{clusters}.nc',
                energy_totals_name='data/energy_totals.csv',
                co2_totals_name='data/co2_totals.csv',
                transport_name='data/transport_data.csv',
                biomass_potentials='data/biomass/biomass_potentials.csv',
                biomass_transport='data/biomass/biomass_transport_costs.csv',
                heat_profile="data/heat_load_profile_BDEW.csv",
                costs="data/costs/",
                costs_old="data/costs_old.csv",
                retro_cost_energy="resources/retro_cost_{network}_s{simpl}_{clusters}.csv",
                floor_area="resources/floor_area_{network}_s{simpl}_{clusters}.csv",
                retro_tax_w="data/eu_elec_taxes_weighting.csv",
                h2_cavern="data/hydrogen_salt_cavern_potentials.csv",
                traffic_data="data/emobility/",
                clustered_pop_layout="resources/pop_layout_{network}_s{simpl}_{clusters}.csv",
                industrial_demand="resources/industrial_demand_{network}_s{simpl}_{clusters}.csv",
                heat_demand_urban="resources/heat_demand_urban_{network}_s{simpl}_{clusters}.nc",
                heat_demand_rural="resources/heat_demand_rural_{network}_s{simpl}_{clusters}.nc",
                heat_demand_total="resources/heat_demand_total_{network}_s{simpl}_{clusters}.nc",
                temp_soil_total="resources/temp_soil_total_{network}_s{simpl}_{clusters}.nc",
                temp_soil_rural="resources/temp_soil_rural_{network}_s{simpl}_{clusters}.nc",
                temp_soil_urban="resources/temp_soil_urban_{network}_s{simpl}_{clusters}.nc",
                temp_air_total="resources/temp_air_total_{network}_s{simpl}_{clusters}.nc",
                temp_air_rural="resources/temp_air_rural_{network}_s{simpl}_{clusters}.nc",
                temp_air_urban="resources/temp_air_urban_{network}_s{simpl}_{clusters}.nc",
                cop_soil_total="resources/cop_soil_total_{network}_s{simpl}_{clusters}.nc",
                cop_soil_rural="resources/cop_soil_rural_{network}_s{simpl}_{clusters}.nc",
                cop_soil_urban="resources/cop_soil_urban_{network}_s{simpl}_{clusters}.nc",
                cop_air_total="resources/cop_air_total_{network}_s{simpl}_{clusters}.nc",
                cop_air_rural="resources/cop_air_rural_{network}_s{simpl}_{clusters}.nc",
                cop_air_urban="resources/cop_air_urban_{network}_s{simpl}_{clusters}.nc",
                solar_thermal_total="resources/solar_thermal_total_{network}_s{simpl}_{clusters}.nc",
                solar_thermal_urban="resources/solar_thermal_urban_{network}_s{simpl}_{clusters}.nc",
                solar_thermal_rural="resources/solar_thermal_rural_{network}_s{simpl}_{clusters}.nc",
                timezone_mappings='data/timezone_mappings.csv'),
            output=dict(
                network='/results/prenetworks/{network}_s{simpl}_{clusters}_lv{lv}_{opts}.nc',
                costs='/costs/assumed_costs_{network}_s{simpl}_{clusters}_lv{lv}_{opts}_{sector_opts}.csv'
            )
        )
        with open('/home/ws/bw0928/Dokumente/pypsa-eur-sec/config.yaml', encoding='utf8') as f:
            snakemake.config = yaml.safe_load(f)

    logging.basicConfig(level=snakemake.config['logging_level'])

    timezone_mappings = pd.read_csv(
        snakemake.input.timezone_mappings,
        index_col=0,
        squeeze=True,
        header=None)

    options = snakemake.config["sector"]

    opts = snakemake.wildcards.sector_opts.split('-')

    n = pypsa.Network(snakemake.input.network,
                      override_component_attrs=override_component_attrs)

    Nyears = n.snapshot_weightings.sum() / 8760.

    pop_layout = pd.read_csv(snakemake.input.clustered_pop_layout, index_col=0)
    pop_layout["ct"] = pop_layout.index.str[:2]
    ct_total = pop_layout.total.groupby(pop_layout["ct"]).sum()
    ct_urban = pop_layout.urban.groupby(pop_layout["ct"]).sum()
    pop_layout["ct_total"] = pop_layout["ct"].map(ct_total.get)
    pop_layout["fraction"] = pop_layout["total"] / pop_layout["ct_total"]
    pop_layout["urban_ct_fraction"] = pop_layout["urban"] / \
        pop_layout["ct"].map(ct_urban.get)

    costs = prepare_costs()

    remove_elec_base_techs(n)

    n.loads["carrier"] = "electricity"

    if not options["costs_old"]:
        # update old pypsa-eur costs with new costs
        update_elec_costs(n, costs)

    # TODO
    costs.at['solid biomass', 'CO2 intensity'] = 0.3

    add_co2_tracking(n)

    # add_generation(n)

    add_storage(n)

    for o in opts:
        if "space" in o:
            limit = o[o.find("space") + 5:]
            limit = float(limit.replace("p", ".").replace("m", "-"))
            print(o, limit)
            options['space_heating_fraction'] = limit
        if o[:4] == "wave":
            wave_cost_factor = float(o[4:].replace("p", ".").replace("m", "-"))
            print(
                "Including wave generators with cost factor of",
                wave_cost_factor)
            add_wave(n, wave_cost_factor)
        if o[:4] == "dist":
            options['electricity_distribution_grid'] = True
            options['electricity_distribution_grid_cost_factor'] = float(o[4:].replace("p",".").replace("m","-"))

    (nodal_energy_totals, heat_demand, ashp_cop, gshp_cop, solar_thermal,
     transport, avail_profile, dsm_profile, co2_totals, nodal_transport_data,
     dist_heat_share) = prepare_data(n)

    if "nodistrict" in opts:
        options["central"] = False

    if "T" in opts:
        add_transport(n)

    if "H" in opts:
        add_heat(n)

    if "B" in opts:
        add_biomass(n)

    if "I" in opts:
        add_industry(n)

    if "H" in opts:
        add_waste_heat(n)

    if "decentral" in opts:
        decentral(n)

    if "noH2network" in opts:
        remove_h2_network(n)

    for o in opts:
        m = re.match(r'^\d+h$', o, re.IGNORECASE)
        if m is not None:
            n = average_every_nhours(n, m.group(0))
            break
    else:
        logger.info("No resampling")

    for o in opts:
        if "Co2L" in o:

            limit = o[o.find("Co2L") + 4:]
            print(o, limit)
            if limit == "":
                limit = snakemake.config['co2_reduction']
            else:
                limit = float(limit.replace("p", ".").replace("m", "-"))
            add_co2limit(n, Nyears, limit)
        #  add_emission_prices(n, exclude_co2=True)

    #  if 'Ep' in opts:
    #      add_emission_prices(n)

        for tech in ["solar", "onwind", "offwind"]:
            if tech in o:
                limit = o[o.find(tech) + len(tech):]
                limit = float(limit.replace("p", ".").replace("m", "-"))
                restrict_technology_potential(n, tech, limit)

    if options['electricity_distribution_grid']:
        insert_electricity_distribution_grid(n)

    if not options["ccs"]:
        print("no CCS")
        n.links = n.links[~n.links.carrier.str.contains("CCS")]

    if not options["fossil_gas"]:
        print("no imported fossil gas")
        n.generators = n.generators[n.generators.carrier != "gas"]
# %%
    n.export_to_netcdf(snakemake.output.network)
