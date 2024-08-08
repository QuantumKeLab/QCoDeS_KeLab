from qcodes_contrib_drivers.drivers.QuTech.IVVI import IVVI
# from qcodes.instrument_drivers.QuTech.IVVI import IVVI
from qcodes.instrument_drivers.tektronix.Keithley_2000 import Keithley_2000
from qcodes.instrument_drivers.stanford_research.SR830 import SR830
from Model_340 import Model_340
from Model_CS4 import Model_CS4
from Model_LM510_USB import Model_LM510_USB
from qcodes.instrument_drivers.yokogawa.GS200 import GS200

level_meter = Model_LM510_USB("level_meter", "ASRL3::INSTR")
station.add_component(level_meter)

keithley_1 = Keithley_2000('keithley_1', 'GPIB0::13::INSTR')
station.add_component(keithley_1)

keithley_2 = Keithley_2000('keithley_2', 'GPIB0::14::INSTR')
station.add_component(keithley_2)

lockin_1 = SR830('lockin_1', 'GPIB0::8::INSTR')
station.add_component(lockin_1)

lockin_2 = SR830('lockin_2', 'GPIB0::9::INSTR')
station.add_component(lockin_2)

lakeshore = Model_340("lakeshore", "GPIB0::12::INSTR")
station.add_component(lakeshore)

cryomag = Model_CS4("cryomag", 'GPIB0::5::INSTR')
station.add_component(cryomag)

ivvi = IVVI('ivvi', 'ASRL1::INSTR')
station.add_component(ivvi)

yoko = GS200('yoko', 'GPIB0::4::INSTR')
station.add_component(yoko)
