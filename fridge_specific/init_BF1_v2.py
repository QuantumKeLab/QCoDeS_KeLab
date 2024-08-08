import pyvisa
import qcodes as qc
station = qc.Station()
from qcodes.instrument_drivers.stanford_research.SR860 import SR860
from qcodes.instrument_drivers.Keithley.Keithley_2400 import Keithley2400
from qcodes.instrument_drivers.Keithley.Keithley_2440 import Keithley2440
from qcodes.instrument_drivers.tektronix.Keithley_6500 import Keithley_6500
from qcodes.instrument_drivers.rohde_schwarz.SGS100A import RohdeSchwarzSGS100A
from qcodes.instrument_drivers.american_magnetics.AMI430 import AMI430, AMI430_3D


rm = pyvisa.ResourceManager()
resources = rm.list_resources()

for resource in resources:
    try:
        my_device = rm.open_resource(resource)
        idn_string = my_device.query('*IDN?')
        print(f"Device: {resource}")
        print(f"IDN: {idn_string}")
        
        if "KEITHLEY" in idn_string and "MODEL DMM6500" in idn_string:
            DMM6500 = Keithley_6500('DMM6500', resource)
            station.add_component(DMM6500)
            print(f"Added Keithley DMM6500 at {resource} to the station.")
            
        elif "KEITHLEY" in idn_string and "MODEL 2400" in idn_string:
            K2400 = Keithley2400('keithley_24', resource)
            station.add_component(K2400)
        
        elif "KEITHLEY" in idn_string and "MODEL 2440" in idn_string:
            K2440 = Keithley2440('K2440', resource)
            station.add_component(K2440)
            
        elif "SR860" in idn_string:
            SR860 = SR860('SR860', resource)
            station.add_component(SR860)
            print(f"Added SR860 at {resource} to the station.")
            
        elif "SGS100A" in idn_string:
            SGS100A = RohdeSchwarzSGS100A('SGS100A', resource)
            station.add_component(SGS100A)
            print(f"Added SGS100A at {resource} to the station.")
        
        elif "AMI" in idn_string:
            AMI = AMI430('AMI430', resource)
            station.add_component(AMI)
            print(f"Added AMI430 at {resource} to the station.")
            
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

        
            
    except Exception as e:
        print(f"Error connecting to {resource}: {e}")