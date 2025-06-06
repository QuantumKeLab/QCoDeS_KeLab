from qcodes_contrib_drivers.drivers.QuTech.IVVI import IVVI
# from qcodes.instrument_drivers.QuTech.IVVI import IVVI
from qcodes.instrument_drivers.tektronix.Keithley_2000 import Keithley_2000
from qcodes.instrument_drivers.stanford_research.SR830 import SR830
from qcodes.instrument_drivers.Lakeshore.Model_325  import Model_325 
from Model_332 import Model_332

keithley_1 = Keithley_2000('keithley_1', 'GPIB0::16::INSTR')
station.add_component(keithley_1)

keithley_2 = Keithley_2000('keithley_2', 'GPIB0::17::INSTR')
station.add_component(keithley_2)

lockin_1 = SR830('lockin_1', 'GPIB0::13::INSTR')
station.add_component(lockin_1)

lockin_2 = SR830('lockin_2', 'GPIB0::9::INSTR')
station.add_component(lockin_2)

lakeshore = Model_332 ("lakeshore", "GPIB0::12::INSTR")
station.add_component(lakeshore)

ivvi = IVVI('ivvi', 'ASRL1::INSTR')
station.add_component(ivvi)
