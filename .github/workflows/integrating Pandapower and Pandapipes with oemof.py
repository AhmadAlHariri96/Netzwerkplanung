import pandapipes as ppipes
import pandapower as ppower
from oemof.tools import economics
import pandas as pd
import matplotlib.pyplot as plt

from pandapipes import networks as g_nw
from pandapower import networks as e_nw
from pandapipes.multinet.create_multinet import create_empty_multinet, add_net_to_multinet
from pandapipes.multinet.control.controller.multinet_control import P2GControlMultiEnergy, G2PControlMultiEnergy
from pandapipes.multinet.control.run_control_multinet import run_control

from pandas import DataFrame
from numpy.random import random
from pandapower.timeseries import DFData

from os.path import join, dirname
from pandapower.timeseries import OutputWriter

from pandapipes.multinet.control.controller.multinet_control import coupled_p2g_const_control, \
    coupled_g2p_const_control

from pandapipes.multinet.timeseries.run_time_series_multinet import run_timeseries

from oemof.solph import components as cmp
from oemof import solph


#Distinguishs between Power and Gas Nets, logs corresponding data with the OutputWriter() function
def create_output_writers(multinet, time_steps=None):
    nets = multinet["nets"]
    ows = dict()
    for key_net in nets.keys():
        ows[key_net] = {}
        if isinstance(nets[key_net], ppower.pandapowerNet):
            log_variables = [('res_bus', 'vm_pu'),
                             ('res_line', 'loading_percent'),
                             ('res_line', 'i_ka'),
                             ('res_bus', 'p_mw'),
                             ('res_bus', 'q_mvar'),
                             ('res_load', 'p_mw'),
                             ('res_load', 'q_mvar'),
                             ('res_ext_grid', 'p_mw'),]
            ow = OutputWriter(nets[key_net], time_steps=time_steps,
                              log_variables=log_variables,
                              output_path=join(dirname('__file__'),'timeseries', 'results', 'power'),
                              output_file_type=".csv")
            ows[key_net] = ow
        elif isinstance(nets[key_net], ppipes.pandapipesNet):
            log_variables = [('res_sink', 'mdot_kg_per_s'),
                             ('res_source', 'mdot_kg_per_s'),
                             ('res_ext_grid', 'mdot_kg_per_s'),
                             ('res_pipe', 'v_mean_m_per_s'),
                             ('res_junction', 'p_bar'),
                             ('res_junction', 't_k')]
            ow = OutputWriter(nets[key_net], time_steps=time_steps,
                              log_variables=log_variables,
                              output_path=join(dirname('__file__'), 'timeseries', 'results', 'gas'),
                              output_file_type=".csv")
            ows[key_net] = ow
        else:
            raise AttributeError("Could not create an output writer for nets of kind " + str(key_net))
    return ows

#Number of Timesteps
n_ts=10

###############################################################################
#Electrical Net
###############################################################################

#Voltage Limits
min_vm_pu = 0.9
max_vm_pu = 1.1

#Line load limits in percent
max_line_load = 100

#Leistung PtG elektrisch in MW
p_PtG_mw = 0.2

#Leistung Grid/Gen elektrisch in MW
max_p_mw_grid = 1
max_p_mw_gen = 1


price_gas = 0.04
epc_wind = economics.annuity(capex=1000, n=20, wacc=0.05)
epc_pv = economics.annuity(capex=1000, n=20, wacc=0.05)
epc_storage = economics.annuity(capex=1000, n=20, wacc=0.05)
#create empty power net
net_power = ppower.create_empty_network()

#create buses
bus1 = ppower.create_bus(net_power, vn_kv=0.4, name="Bus 1")
bus2 = ppower.create_bus(net_power, vn_kv=0.4, name="Bus 2")
bus3 = ppower.create_bus(net_power, vn_kv=0.4, name="Bus 3")
bus4 = ppower.create_bus(net_power, vn_kv=0.4, name="Bus 4")
bus5 = ppower.create_bus(net_power, vn_kv=0.4, name="Bus 5")
bus6 = ppower.create_bus(net_power, vn_kv=0.4, name="Bus 6")
bus7 = ppower.create_bus(net_power, vn_kv=0.4, name="Bus 7")
bus8 = ppower.create_bus(net_power, vn_kv=0.4, name="Bus 8")
bus9 = ppower.create_bus(net_power, vn_kv=0.4, name="Bus 9")
bus10 = ppower.create_bus(net_power, vn_kv=0.4, name="Bus 10")
bus11 = ppower.create_bus(net_power, vn_kv=0.4, name="Bus 11")
bus12 = ppower.create_bus(net_power, vn_kv=0.4, name="Bus 12")
bus13 = ppower.create_bus(net_power, vn_kv=0.4, name="Bus 13")
bus14 = ppower.create_bus(net_power, vn_kv=0.4, name="Bus 14")                    

#create external grid
grid1 = ppower.create_ext_grid(net_power, bus=bus1, min_p_mw=-1, max_p_mw=max_p_mw_grid, name="Ext Grid1")
#grid2 = ppower.create_ext_grid(net_power, bus=bus7, min_p_mw=-1, max_p_mw=1, name="Ext Grid1")


#create Bus switch
#switch1 = ppower.create_switch(net_power, bus = 7, element = 8, et = "", closed=True)

#create loads
load1 = ppower.create_load(net_power, bus=bus1, p_mw=0.03, min_p_mw=0, max_p_mw=0.2, min_q_mvar=0, max_q_mvar=0.1, controllable=False,  name="Load 1")
load2 = ppower.create_load(net_power, bus=bus2, p_mw=0.03, min_p_mw=0, max_p_mw=0.2, min_q_mvar=0, max_q_mvar=0.1, controllable=False, name="Load 2")
load3 = ppower.create_load(net_power, bus=bus3, p_mw=0.03 ,min_p_mw=0, max_p_mw=0.2, min_q_mvar=0, max_q_mvar=0.1, controllable=False, name="Load 3")
load4 = ppower.create_load(net_power, bus=bus4, p_mw=0.03, min_p_mw=0, max_p_mw=0.2, min_q_mvar=0, max_q_mvar=0.1, controllable=False, name="Load 4")
load5 = ppower.create_load(net_power, bus=bus5, p_mw=0.03, min_p_mw=0, max_p_mw=0.2, min_q_mvar=0, max_q_mvar=0.1, controllable=False,  name="Load 5")
load6 = ppower.create_load(net_power, bus=bus6, p_mw=0.03, min_p_mw=0, max_p_mw=0.2, min_q_mvar=0, max_q_mvar=0.1, controllable=False, name="Load 6")
load7 = ppower.create_load(net_power, bus=bus7, p_mw=0.03, min_p_mw=0, max_p_mw=0.2, min_q_mvar=0, max_q_mvar=0.1, controllable=False, name="Load 7")
load8 = ppower.create_load(net_power, bus=bus8, p_mw=0.03, min_p_mw=0, max_p_mw=0.2, min_q_mvar=0, max_q_mvar=0.1, controllable=False, name="Load 8")
load9 = ppower.create_load(net_power, bus=bus9, p_mw=0.03, min_p_mw=0, max_p_mw=0.2, min_q_mvar=0, max_q_mvar=0.1, controllable=False, name="Load 9")
load10 = ppower.create_load(net_power, bus=bus10, p_mw=0.03, min_p_mw=0, max_p_mw=0.2, min_q_mvar=0, max_q_mvar=0.1, controllable=False, name="Load 10")
load11 = ppower.create_load(net_power, bus=bus11, p_mw=0.03, min_p_mw=0, max_p_mw=0.2, min_q_mvar=0, max_q_mvar=0.1, controllable=False, name="Load 11")
load12 = ppower.create_load(net_power, bus=bus12, p_mw=0.03, min_p_mw=0, max_p_mw=0.2, min_q_mvar=0, max_q_mvar=0.1, controllable=False, name="Load 12")
load13 = ppower.create_load(net_power, bus=bus13, p_mw=0.03, min_p_mw=0, max_p_mw=0.2, min_q_mvar=0, max_q_mvar=0.1, controllable=False, name="Load 13")
load14 = ppower.create_load(net_power, bus=bus14, p_mw=0.03, min_p_mw=0, max_p_mw=0.2, min_q_mvar=0, max_q_mvar=0.1, controllable=False, name="Load 14")


#create PV 
#PV1 = ppower.create_gen(net_power, bus=bus4, p_mw=0.01, q_mvar=0, min_p_mw=0, max_p_mw=0.01, min_q_mvar=-1, max_q_mvar=1, controllable=True, name="PV1")
#PV2 = ppower.create_sgen(net_power, bus=bus6, p_mw=0.01, min_p_mw=0, max_p_mw=0.01, min_q_mvar=-1, max_q_mvar=0.1, name="PV2")
#PV3 = ppower.create_sgen(net_power, bus=bus10, p_mw=0.01, min_p_mw=0, max_p_mw=0.01, min_q_mvar=-1, max_q_mvar=0.1, name="PV3")

#create variable generator
Gen1 = ppower.create_gen(net_power, bus=bus7, p_mw=0.4, min_p_mw=0, max_p_mw=max_p_mw_gen, controllable=True,  name="Gen1")


#create lines
line1 = ppower.create_line(net_power, from_bus=bus1, to_bus=bus2, length_km=0.05, std_type="NAYY 4x150 SE", name="Line 1_2", max_loading_percent=max_line_load)
line2 = ppower.create_line(net_power, from_bus=bus2, to_bus=bus3, length_km=0.05, std_type="NAYY 4x150 SE", name="Line 2_3", max_loading_percent=max_line_load)
line3 = ppower.create_line(net_power, from_bus=bus3, to_bus=bus4, length_km=0.05, std_type="NAYY 4x150 SE", name="Line 3_4", max_loading_percent=max_line_load)
line4 = ppower.create_line(net_power, from_bus=bus4, to_bus=bus5, length_km=0.05, std_type="NAYY 4x150 SE", name="Line 4_5", max_loading_percent=max_line_load)
line5 = ppower.create_line(net_power, from_bus=bus5, to_bus=bus6, length_km=0.05, std_type="NAYY 4x150 SE", name="Line 5_6", max_loading_percent=max_line_load)
line6 = ppower.create_line(net_power, from_bus=bus6, to_bus=bus7, length_km=0.05, std_type="NAYY 4x150 SE", name="Line 6_7", max_loading_percent=max_line_load)
line7 = ppower.create_line(net_power, from_bus=bus7, to_bus=bus8, length_km=0.05, std_type="NAYY 4x150 SE", name="Line 7_8", max_loading_percent=max_line_load)
line8 = ppower.create_line(net_power, from_bus=bus8, to_bus=bus9, length_km=0.05, std_type="NAYY 4x150 SE", name="Line 8_9", max_loading_percent=max_line_load)
line9 = ppower.create_line(net_power, from_bus=bus9, to_bus=bus10, length_km=0.05, std_type="NAYY 4x150 SE", name="Line 7_8", max_loading_percent=max_line_load)
line10 = ppower.create_line(net_power, from_bus=bus10, to_bus=bus11, length_km=0.05, std_type="NAYY 4x150 SE", name="Line 8_9", max_loading_percent=max_line_load)
line11 = ppower.create_line(net_power, from_bus=bus11, to_bus=bus2, length_km=0.05, std_type="NAYY 4x150 SE", name="Line 9_2", max_loading_percent=max_line_load)
line12 = ppower.create_line(net_power, from_bus=bus1, to_bus=bus12, length_km=0.05, std_type="NAYY 4x150 SE", name="Line 1_10", max_loading_percent=max_line_load)
line13 = ppower.create_line(net_power, from_bus=bus12, to_bus=bus13, length_km=0.05, std_type="NAYY 4x150 SE", name="Line 10_11", max_loading_percent=max_line_load)
line14 = ppower.create_line(net_power, from_bus=bus13, to_bus=bus14, length_km=0.05, std_type="NAYY 4x150 SE", name="Line 11_12", max_loading_percent=max_line_load)

#create cost functions

ppower.create_poly_cost(net_power, element=grid1, et="ext_grid", cp1_eur_per_mw = 20)
ppower.create_poly_cost(net_power, element=Gen1, et="gen", cp1_eur_per_mw = 10)


#run net
ppower.runopp(net_power, verbose=False, delta=1e-16, suppress_warnings=True)

#print net 
#simple_plot(net_power, ext_grid_size=4.0, plot_loads=True, plot_gens=True,  gen_size=4.0)
#pf_res_plotly(net_power, auto_open=False, filename="Plot-ohne-PtG.html")


print("\n","Line load in percent:")
print(net_power.res_line.loading_percent, "\n")
print("Bus voltage per unit:")
print(net_power.res_bus.vm_pu, "\n")
print("Grid output")
print(net_power.res_ext_grid, "\n")
#print("PV output")
#print(net_power.res_sgen, "\n")
print("Variable generator output")
print(net_power.res_gen, "\n")
print("Sum load in mv")
print(sum(net_power.load.p_mw), " \n")

print("power grid in percent ")
print(net_power.res_ext_grid.p_mw/max_p_mw_grid*100, "\n")

print("generator in percent ")
print(net_power.res_gen.p_mw/max_p_mw_gen*100, "\n")

print("grid / generator")
print(net_power.res_ext_grid.p_mw/net_power.res_gen.p_mw, "\n")

print(f"The optimal cost are: {net_power.res_cost} $/hr")


###############################################################################
#Hydrogen Net
###############################################################################

# create an empty gas network
net_gas = ppipes.create_empty_network(fluid="hydrogen")

# create network elements, such as junctions, external grid, pipes, valves, sinks and sources
junction1 = ppipes.create_junction(net_gas, pn_bar=1.05, tfluid_k=293.15, name="Connection to External Grid", geodata=(0, 0))
junction2 = ppipes.create_junction(net_gas, pn_bar=1.05, tfluid_k=293.15, name="Junction 2", geodata=(2, 0))
junction3 = ppipes.create_junction(net_gas, pn_bar=1.05, tfluid_k=293.15, name="Junction 3", geodata=(7, 4))
junction4 = ppipes.create_junction(net_gas, pn_bar=1.05, tfluid_k=293.15, name="Junction 4", geodata=(7, -4))
junction5 = ppipes.create_junction(net_gas, pn_bar=1.05, tfluid_k=293.15, name="Junction 5", geodata=(5, 3))
junction6 = ppipes.create_junction(net_gas, pn_bar=1.05, tfluid_k=293.15, name="Junction 6", geodata=(5, -3))

ext_grid = ppipes.create_ext_grid(net_gas, junction=junction1, p_bar=1.1, t_k=293.15, name="Grid Connection")

pipe1 = ppipes.create_pipe_from_parameters(net_gas, from_junction=junction1, to_junction=junction2, length_km=10, diameter_m=0.3, name="Pipe 1", geodata=[(0, 0), (2, 0)])
pipe2 = ppipes.create_pipe_from_parameters(net_gas, from_junction=junction2, to_junction=junction3, length_km=2, diameter_m=0.3, name="Pipe 2", geodata=[(2, 0), (2, 4), (7, 4)])
pipe3 = ppipes.create_pipe_from_parameters(net_gas, from_junction=junction2, to_junction=junction4, length_km=2.5, diameter_m=0.3, name="Pipe 3", geodata=[(2, 0), (2, -4), (7, -4)])
pipe4 = ppipes.create_pipe_from_parameters(net_gas, from_junction=junction3, to_junction=junction5, length_km=1, diameter_m=0.3, name="Pipe 4", geodata=[(7, 4), (7, 3), (5, 3)])
pipe5 = ppipes.create_pipe_from_parameters(net_gas, from_junction=junction4, to_junction=junction6, length_km=1, diameter_m=0.3, name="Pipe 5", geodata=[(7, -4), (7, -3), (5, -3)])

valve = ppipes.create_valve(net_gas, from_junction=junction5, to_junction=junction6, diameter_m=0.05, opened=True)

sink = ppipes.create_sink(net_gas, junction=junction4, mdot_kg_per_s=0.002, name="Sink 1")

source = ppipes.create_source(net_gas, junction=junction3, mdot_kg_per_s=0.001)
#ppipes.pipeflow(net_gas)

#plot.simple_plot(net_gas, plot_sinks=True, plot_sources=True, sink_size=4.0, source_size=4.0)


###############################################################################
#Multinet
###############################################################################

# create multinet and add networks:
multinet = create_empty_multinet("PtG_multinet")
add_net_to_multinet(multinet, net_power, 'power')
add_net_to_multinet(multinet, net_gas, 'gas')

# create elements corresponding to conversion units:
p2g_el = ppower.create_load(net_power, bus=bus7, p_mw=p_PtG_mw, name="power to gas consumption")
p2g_gas = ppipes.create_source(net_gas, junction=junction4, mdot_kg_per_s=0, name="power to gas feed in")


#Create Datasource ds for Timeseries
profiles = DataFrame()
profiles['power to gas consumption'] = random(n_ts)*p_PtG_mw
ds = DFData(profiles)

print(profiles)

#Write data
ows = create_output_writers(multinet, n_ts)

#Controller for Multi-energy flow and time series
coupled_p2g_const_control(multinet, p2g_el, p2g_gas,
                          name_power_net="power", name_gas_net="gas",
                          profile_name='power to gas consumption', data_source=ds,
                          p2g_efficiency=0.7)


#solver options
#Creates a dict in the required form with the specified solver options for each net
run_options = {
    'power': ppower.runopp,
    'gas': ppipes.pipeflow
}

# run simulation:
run_timeseries(multinet, time_steps=range(n_ts), output_writers=ows, run = run_options)

#print net 

#pf_res_plotly(net_power, auto_open=False, filename="Plot-mit-PtG.html")
# Erstellen eines leeren Energiesystems
demand_el = [
    0,
    0,
    0,
    1,
    1,
    1,
    0,
    0,
    1,
    1,
    1,
    0,
    0,
    1,
    1,
    1,
    0,
    0,
    1,
    1,
    1,
    1,
    0,
    0,
]
idx = pd.date_range('2023-01-01', periods=24, freq='H')
es = solph.EnergySystem(timeindex=idx, infer_last_interval=False)

oemof_buses = {}
oemof_sinks = {}
oemof_sources = {}
oemof_storages = []

# Konvertierung der Pandapower Busse zu Oemof Bussen
for bus in net_power.bus.itertuples():
    power_bus_label = f"Power_Bus_{bus.Index}"
    if power_bus_label not in oemof_buses:
        power_bus = solph.Bus(label=power_bus_label)
        es.add(power_bus)
        oemof_buses[power_bus_label] = power_bus
        # Hinzufügen einer Überschusskomponente für diesen Bus
        # Hinzufügen einer Überschuss-Senke für diesen Bus, um überschüssige Energie aufzunehmen und zu verhindern,
        #dass das Modell durch Überproduktion unlösbar wird. 
        #Dies ermöglicht eine flexiblere Handhabung von Energieüberschüssen im System.
        excess_label = f"Excess_{power_bus_label}"
        excess_sink = solph.components.Sink(label=excess_label, inputs={power_bus: solph.Flow()})
        es.add(excess_sink)
        # Hinzufügen einer Mangel-Quelle für diesen Bus
        # Hinzufügen einer Mangel-Quelle für diesen Bus,
        #um mögliche Energiemängel durch das Bereitstellen von Energie zu hohen Kosten auszugleichen.
        #Dies dient als letzte Option zur Sicherstellung der Versorgung, wenn das vorhandene Angebot die Nachfrage nicht decken kann.
        shortage_label = f"Shortage_{power_bus_label}"
        shortage_source = solph.components.Source(label=shortage_label, outputs={power_bus: solph.Flow(variable_costs=1000)})
        es.add(shortage_source)
        
   # Konvertierung der Pandapipes Junctions zu Oemof Bussen
for junction in net_gas.junction.itertuples():
    gas_bus_label = f"Gas_Junction_{junction.Index}"
    if gas_bus_label not in oemof_buses:
        gas_bus = solph.Bus(label=gas_bus_label)
        es.add(gas_bus)
        oemof_buses[gas_bus_label] = gas_bus
   
# power bus and components
#bel = solph.Bus(label="bel")

if hasattr(net_power, 'load') and not net_power.load.empty:
    for pandapower_load in net_power.load.itertuples(index=False):
        p_mw = pandapower_load.p_mw
        # Umrechnung von MW in Watt
        #P_W_oemof = p_mw * 1e6
        nominal_value=10
        oemof_sink = solph.components.Sink(label=pandapower_load.name,
                                inputs={oemof_buses[f"Power_Bus_{pandapower_load.bus}"]: 
                                        solph.Flow(fix=demand_el, nominal_value=nominal_value)})
        es.add(oemof_sink)
        oemof_sinks[pandapower_load.name] = oemof_sink

# Konvertierung von Pandapower-Sources in Oemof-Sources
if hasattr(net_power, 'sgen') and not net_power.sgen.empty:
    for pandapower_source in net_power.sgen.itertuples(index=False):
        source_label = pandapower_source.name  # Name der Quelle
        bus_index = pandapower_source.bus     # Bus-Index
        p_mw = pandapower_source.p_mw         # Leistung in MW
        variable_costs=pandapower_source.variable_costs
        min_p_mw_oemof=pandapower_source.min_p_mw * 1e6
        max_p_mw_oemof=pandapower_source.max_p_mw* 1e6
        nominal_value=pandapower_source.nominal_value
        startup_costs=pandapower_source.startup_costs
        shutdown_costs=pandapower_source.shutdown_costs
        oemof_source = solph.components.Source(label=source_label,
                                    outputs={oemof_buses[f"Power_Bus_{bus_index}"]: 
                                             solph.Flow(nominal_value=nominal_value,
                                                        variable_costs=variable_costs,
                                                        nonconvex=solph.NonConvex(startup_costs=startup_costs, shutdown_costs=shutdown_costs),
                                                        )})
        es.add(oemof_source)
        oemof_sources[source_label] = oemof_source

# Integration der pandapower Storage in oemof.solph
if hasattr(net_power, 'storage') and not net_power.storage.empty:
    for pandapower_storage in net_power.storage.itertuples(index=False):
        connected_bus = oemof_buses[f"Power_Bus_{pandapower_storage.bus}"]
        #storage_capacity = pandapower_storage.max_e_mwh  # Maximale Energiekapazität in MWh
        #charge_power = pandapower_storage.max_p_mw  # Maximale Ladeleistung in MW
        #discharge_power = pandapower_storage.max_p_mw  # Maximale Entladeleistung in MW
        #charge_efficiency = pandapower_storage.efficiency_percent / 100  # Ladeeffizienz
        #discharge_efficiency = pandapower_storage.efficiency_percent / 100  # Entladeeffizienz
        #loss_rate = pandapower_storage.self_discharge_percent / 100  # Selbstentladungsrate pro Stunde

        oemof_storage = solph.components.GenericStorage(
            label=pandapower_storage.name,
            inputs={connected_bus: solph.Flow(variable_costs=0.0)}, 
            outputs={connected_bus: solph.Flow()},  # Annahme von Null variablen Kosten fürs Entladen
            nominal_storage_capacity=solph.Investment(ep_costs=epc_storage),
            loss_rate=0.00,
            initial_storage_level=0,
            invest_relation_input_capacity=1 / 6,
            invest_relation_output_capacity=1 / 6,
            inflow_conversion_factor=1,
            outflow_conversion_factor=0.8,
            #initial_storage_level=pandapower_storage.soc_percent / 100  # Anfängliches Speicherniveau als Anteil
        )

        # Hinzufügen des Storage-Objekts zur Liste und zum Energiesystem
        es.add(oemof_storage)
        oemof_storages.append(oemof_storage)

# Konvertierung von Pandapipes-Sources in Oemof-Sources
if hasattr(net_gas, 'source') and not net_gas.source.empty:
    for pandapipes_source in net_gas.source.itertuples(index=False):
        source_label = pandapipes_source[0]  # Name der Quelle
        junction_index = pandapipes_source[1]  # Junction-Index
        mdot_kg_per_s = pandapipes_source[2]  # Massenstrom in kg/s
    
        #P_W_oemof = mdot_kg_per_s * c_p * temperature_difference_K  # Umrechnung in Energiestrom in Watt
    
        oemof_source = solph.components.Source(label=source_label,
                                    outputs={oemof_buses[f"Gas_Junction_{junction_index}"]: 
                                             solph.Flow(variable_costs=price_gas)})
        es.add(oemof_source)
        oemof_sources[source_label] = oemof_source


# Konvertierung von Pandapipes-Sinks in Oemof-Sinks
if hasattr(net_gas, 'sink') and not net_gas.sink.empty:
    for pandapipes_sink in net_gas.sink.itertuples(index=False):
        mdot_kg_per_s = pandapipes_sink.mdot_kg_per_s
        #P_W_oemof = mdot_kg_per_s * c_p * temperature_difference_K  # Umrechnung in Energiestrom in Watt
    
        oemof_sink = cmp.Sink(label=pandapipes_sink.name,
                                inputs={oemof_buses[f"Gas_Junction_{pandapipes_sink.junction}"]: 
                                        solph.Flow(fix=100, nominal_value=100)})
        es.add(oemof_sink)
        oemof_sinks[pandapipes_sink.name] = oemof_sink





oemof_converter = solph.components.Converter(
label="P2G_Converter",
inputs={oemof_buses["Gas_Junction_0"]: solph.Flow()},
outputs={oemof_buses["Power_Bus_0"]: solph.Flow(nominal_value=10, variable_costs=0)},
conversion_factors={oemof_buses["Power_Bus_0"]: 0.7}
)
es.add(oemof_converter)


# create an optimization problem and solve it
om = solph.Model(es)

# debugging
# om.write('problem.lp', io_options={'symbolic_solver_labels': True})

# solve model
om.solve(solver="cbc", solve_kwargs={"tee": True})

# create result object
results = solph.processing.results(om)

# Energieflüsse extrahieren
flow_dataframes = []
for (component, attr), values in results.items():
    if 'sequences' in values:
        df = values['sequences']
        df.columns = [f"{component.label}_{attr}_{col}" for col in df.columns]
        flow_dataframes.append(df)

# Speicherebenen extrahieren
storage_dataframes = []
for component in es.groups:
    if isinstance(component, solph.components.GenericStorage):
        df = solph.views.node(results, component.label)['sequences']
        df.columns = [f"{component.label}_storage_content"]
        storage_dataframes.append(df)

# Zusammenführen aller DataFrames zu einem einzigen DataFrame
all_dataframes = flow_dataframes + storage_dataframes
result_df = pd.concat(all_dataframes, axis=1)

# Speichern der Ergebnisse in eine Excel-Datei
result_df.to_excel("optimization_results.xlsx", index=True)

print("Ergebnisse wurden in 'optimization_results.xlsx' gespeichert.")

# Plotten der Energieflüsse
for df in flow_dataframes:
    df.plot(title="Energy Flows")
    plt.xlabel("Time")
    plt.ylabel("Flow")
    plt.show()

# Plotten der Speicherebenen
for df in storage_dataframes:
    df.plot(title="Storage Levels")
    plt.xlabel("Time")
    plt.ylabel("Storage Content")
    plt.show()
