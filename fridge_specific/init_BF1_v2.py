from qcodes.math_utils.field_vector import FieldVector
from qcodes.instrument_drivers.stanford_research.SR860 import SR860
from qcodes.instrument_drivers.Keithley.Keithley_2400 import Keithley2400
from qcodes.instrument_drivers.tektronix.Keithley_6500 import Keithley_6500
from qcodes.instrument_drivers.rohde_schwarz.SGS100A import RohdeSchwarzSGS100A
from qcodes.instrument_drivers.american_magnetics.AMI430 import AMI430, AMI430_3D

keithley_K1 = Keithley_6500('keithley_K1', 'GPIB0::6::INSTR')
station.add_component(keithley_K1)

# keithley_2 = Keithley_6500('keithley_2', 'GPIB0::7::INSTR')
# station.add_component(keithley_2)

# keithley_19 = Keithley2400('keithley_19', 'GPIB0::19::INSTR')
# station.add_component(keithley_19)

# keithley_20 = Keithley2000('keithley_20', 'GPIB0::22::INSTR')
# station.add_component(keithley_20)

keithley_24 = Keithley2400('keithley_24', 'GPIB0::22::INSTR')
station.add_component(keithley_24)

lockin_1 = SR860('lockin_1', 'GPIB0::5::INSTR')
station.add_component(lockin_1)

# SGS = RohdeSchwarzSGS100A('SGS', 'GPIB0::28::INSTR')
# station.add_component(SGS)

# folder_path = 'C:\\Users\\admin\\SynologyDrive\\09 Data\\Fridge log'
# bf = BlueFors('bf_fridge',
#               folder_path=folder_path,
#               channel_vacuum_can=1,
#               channel_pumping_line=2,
#               channel_compressor_outlet=3,
#               channel_compressor_inlet=4,
#               channel_mixture_tank=5,
#               channel_venting_line=6,
#               channel_50k_plate=1,
#               channel_4k_plate=2,
#               channel_magnet=3,
#               channel_still=6,
#               channel_mixing_chamber=5)


# magnet_z = AMI430("z", address='169.254.113.167', port=7180)
# magnet_y = AMI430("y", address='169.254.229.77', port=7180)
# magnet_x = AMI430("x", address='169.254.67.202', port=7180)

# field_limit = [  # If any of the field limit functions are satisfied we are in the safe zone.
#     lambda x, y, z: x < 1 and y < 1 and z < 9
# ]

# i3d = AMI430_3D(
#    "AMI430-3D",
#     magnet_x,
#     magnet_y,
#     magnet_z,
#     field_limit=field_limit
# )

# station.add_component(magnet_x)
# station.add_component(magnet_y)
# station.add_component(magnet_z)
# station.add_component(i3d)

