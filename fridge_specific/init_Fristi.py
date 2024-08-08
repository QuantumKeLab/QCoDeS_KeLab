from qcodes_contrib_drivers.drivers.QuTech.IVVI import IVVI
#from qcodes.instrument_drivers.QuTech.IVVI import IVVI
from qcodes_contrib_drivers.drivers.Tektronix.Keithley_2700 import Keithley_2700
from qcodes.instrument_drivers.stanford_research.SR830 import SR830
from qcodes.instrument_drivers.rigol.DG4000 import Rigol_DG4000
from qcodes.instrument_drivers.rigol.DS1074Z import DS1074Z
from qcodes.instrument_drivers.rigol.DS4000 import DS4000
from qcodes.instrument_drivers.american_magnetics.AMI430 import AMI430, AMI430_3D
from qcodes.math_utils.field_vector import FieldVector
from GS610 import GS610

keithley_1 = Keithley_2700('keithley_1', 'GPIB0::17::INSTR')
station.add_component(keithley_1)

keithley_2 = Keithley_2700('keithley_2', 'GPIB0::14::INSTR')
station.add_component(keithley_2)

# lockin_1 = SR830('lockin_1', 'GPIB0::10::INSTR')
# station.add_component(lockin_1)

# lockin_2 = SR830('lockin_2', 'GPIB0::8::INSTR')
# station.add_component(lockin_2)

rigol_awg = Rigol_DG4000('rigol_awg', 'USB0::0x1AB1::0x0641::DG4E153100321::INSTR')
station.add_component(rigol_awg)

#rigol_scope = DS1074Z('rigol_scope', 'USB0::0x1AB1::0x04CE::DS1ZB161650342::INSTR')
#station.add_component(rigol_scope)

scope = DS4000('DS4034_2', 'USB0::0x1AB1::0x04B1::DS4A191000057::INSTR')
station.add_component(scope)

ivvi = IVVI('ivvi', 'ASRL1::INSTR')
station.add_component(ivvi)

magnet_x = AMI430("x", address='192.168.0.3', port=7180)
magnet_y = AMI430("y", address='192.168.0.2', port=7180)
magnet_z = AMI430("z", address='192.168.0.1', port=7180)

field_limit = [  # If any of the field limit functions are satisfied we are in the safe zone.
    lambda x, y, z: x < 1 and y < 3 and z < 8
]

i3d = AMI430_3D(
    "AMI430-3D",
    magnet_x,
    magnet_y,
    magnet_z,
    field_limit=field_limit
)

station.add_component(magnet_x)
station.add_component(magnet_y)
station.add_component(magnet_z)
station.add_component(i3d)

#yoko = GS610('gs610', 'USB0::0x0B21::0x001E::91W813376C::INSTR')
#station.add_component(yoko)