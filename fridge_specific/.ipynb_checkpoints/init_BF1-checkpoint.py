# from qcodes_contrib_drivers.drivers.QuTech.IVVI import IVVI
# from qcodes.instrument_drivers.QuTech.IVVI import IVVI
# from qcodes.instrument_drivers.tektronix.Keithley_6500 import Keithley_6500
# from qcodes.instrument_drivers.tektronix.Keithley_2000 import Keithley_2000
# from qcodes.instrument_drivers.rohde_schwarz.SGS100A import RohdeSchwarzSGS100A
# from Kei213_2 import K213
# from SMB100A import SMB100A
from qcodes.instrument_drivers.stanford_research.SR860 import SR860
# from qcodes.instrument_drivers.american_magnetics.AMI430 import AMI430, AMI430_3D
# from qcodes.math_utils.field_vector import FieldVector
# from qcodes.instrument_drivers.yokogawa.GS200 import GS200

# level_meter = Model_LM500("level_meter", "GPIB0::3::INSTR")
# station.add_component(level_meter)

# keithley_1 = Keithley_2000('keithley_1', 'GPIB0::22::INSTR')
# station.add_component(keithley_1)

# keithley_2 = Keithley_2000('keithley_2', 'GPIB0::23::INSTR')
# station.add_component(keithley_2)

# K213 = K213('k213', 'GPIB0::9::INSTR')
# station.add_component(K213)

# SGS = RohdeSchwarzSGS100A('SGS', 'GPIB0::28::INSTR')
# station.add_component(SGS)

# SMB = SMB100A('SMB', 'GPIB0::28::INSTR')
# station.add_component(SMB)

lockin_1 = SR860('lockin_1', 'GPIB0::5::INSTR')
station.add_component(lockin_1)

# lakeshore = Model_340("lakeshore", "GPIB0::12::INSTR")
# station.add_component(lakeshore)

# magnet_x = AMI430("x", address='192.168.0.3', port=7180)
# magnet_y = AMI430("y", address='192.168.0.2', port=7180)
# magnet_z = AMI430("z", address='192.168.0.1', port=7180)

# field_limit = [  # If any of the field limit functions are satisfied we are in the safe zone.
#     lambda x, y, z: x < 1 and y < 1 and z < 9
# ]

# i3d = AMI430_3D(
#     "AMI430-3D",
#     magnet_x,
#     magnet_y,
#     magnet_z,
#     field_limit=field_limit
# )

# station.add_component(magnet_x)
# station.add_component(magnet_y)
# station.add_component(magnet_z)
# station.add_component(i3d)

# ivvi = IVVI('ivvi', 'ASRL4::INSTR')
# station.add_component(ivvi)

# yoko = GS200('yoko', 'USB0::0x0B21::0x0039::91T926460::INSTR')
# station.add_component(yoko)
