from qcodes.instrument_drivers.QuTech.IVVI import IVVI
from qcodes.instrument_drivers.tektronix.Keithley_2000 import Keithley_2000
from qcodes.instrument_drivers.stanford_research.SR830 import SR830
from qcodes.instrument_drivers.american_magnetics.AMI430 import AMI430, AMI430_3D
from qcodes.math.field_vector import FieldVector

keithley_1 = Keithley_2000('keithley_1', 'GPIB0::15::INSTR')
station.add_component(keithley_1)

keithley_2 = Keithley_2000('keithley_2', 'GPIB0::17::INSTR')
station.add_component(keithley_2)

keithley_3 = Keithley_2000('keithley_3', 'GPIB0::16::INSTR')
station.add_component(keithley_3)

lockin_1 = SR830('lockin_1', 'GPIB0::8::INSTR')
station.add_component(lockin_1)

lockin_2 = SR830('lockin_2', 'GPIB0::9::INSTR')
station.add_component(lockin_2)

ivvi = IVVI('ivvi', 'ASRL3::INSTR')
station.add_component(ivvi)

ivvi_2 = IVVI('ivvi_2', 'ASRL4::INSTR')
station.add_component(ivvi_2)

magnet_x = AMI430("x", address='192.168.2.3', port=7180)
magnet_y = AMI430("y", address='192.168.2.2', port=7180)
magnet_z = AMI430("z", address='192.168.2.1', port=7180)

station.add_component(magnet_x)
station.add_component(magnet_y)
station.add_component(magnet_z)

field_limit = [  # If any of the field limit functions are satisfied we are in the safe zone.
    lambda x, y, z: x < 1 and y < 1 and z < 1
]

i3d = AMI430_3D(
    "AMI430-3D",
    magnet_x,
    magnet_y,
    magnet_z,
    field_limit=field_limit
)

station.add_component(i3d)