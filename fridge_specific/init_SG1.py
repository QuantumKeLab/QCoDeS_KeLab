from qcodes_contrib_drivers.drivers.QuTech.IVVI import IVVI
from qcodes.instrument_drivers.tektronix.Keithley_6500 import Keithley_6500
from qcodes.instrument_drivers.stanford_research.SR830 import SR830
from qcodes.instrument_drivers.oxford.triton import Triton
# from Oxford_triton import Oxford_triton
from MercuryiPS_RS232 import MercuryiPS
from qcodes.instrument_drivers.yokogawa.GS200 import GS200
from GS610 import GS610

keithley_1 = Keithley_6500('keithley_1', 'USB0::0x05E6::0x6500::04389947::INSTR')
station.add_component(keithley_1)

keithley_2 = Keithley_6500('keithley_2', 'USB0::0x05E6::0x6500::04387950::INSTR')
station.add_component(keithley_2)

# lockin_3 = SR830('lockin_3', 'GPIB0::10::INSTR')
# station.add_component(lockin_3)

# lockin_4 = SR830('lockin_4', 'GPIB0::6::INSTR')
# station.add_component(lockin_4)

#### non-locoal, another two lockins####
lockin_1 = SR830('lockin_1', 'GPIB0::8::INSTR')
station.add_component(lockin_1)

lockin_2 = SR830('lockin_2', 'GPIB0::9::INSTR')
station.add_component(lockin_2)

ivvi = IVVI('ivvi', 'ASRL4::INSTR')
station.add_component(ivvi)

#Damaz's server
# triton = Oxford_triton('triton', address='145.94.39.251', port=5611)

#Native server
# triton = Triton('triton', address='145.94.39.251', port=33576)
# station.add_component(triton)

# Direct magnet connection
# magnet = MercuryiPS('magnet', 'ASRL1::INSTR')
# station.add_component(magnet)

# yoko = GS200('yoko', 'TCPIP0::169.254.0.20::7655::SOCKET')
yoko = GS200('yoko', 'GPIB0::4::INSTR')
station.add_component(yoko)

# yoko = GS610('gs610', 'USB0::0x0B21::0x001E::91W813376C::INSTR')
# station.add_component(yoko)