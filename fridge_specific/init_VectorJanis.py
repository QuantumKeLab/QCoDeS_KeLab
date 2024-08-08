from qcodes_contrib_drivers.drivers.QuTech.IVVI import IVVI
# from qcodes.instrument_drivers.QuTech.IVVI import IVVI
from qcodes.instrument_drivers.tektronix.Keithley_6500 import Keithley_6500
from qcodes.instrument_drivers.stanford_research.SR830 import SR830
from Model_340 import Model_340
from Model_4G import Model_4G
from Model_LM500 import Model_LM500
from qcodes.instrument_drivers.yokogawa.GS200 import GS200

level_meter = Model_LM500("level_meter", "GPIB0::3::INSTR")
station.add_component(level_meter)

keithley_1 = Keithley_6500('keithley_1', 'USB0::0x05E6::0x6500::04438025::INSTR')
station.add_component(keithley_1)

keithley_2 = Keithley_6500('keithley_2', 'USB0::0x05E6::0x6500::04441805::INSTR')
station.add_component(keithley_2)

lockin_1 = SR830('lockin_1', 'GPIB0::8::INSTR')
station.add_component(lockin_1)

lockin_2 = SR830('lockin_2', 'GPIB0::9::INSTR')
station.add_component(lockin_2)

lakeshore = Model_340("lakeshore", "GPIB0::12::INSTR")
station.add_component(lakeshore)

cryomag = Model_4G("cryomag", 'GPIB0::11::INSTR')
station.add_component(cryomag)

ivvi = IVVI('ivvi', 'ASRL4::INSTR')
station.add_component(ivvi)

# yoko = GS200('yoko', 'USB0::0x0B21::0x0039::91T926460::INSTR')
# station.add_component(yoko)
