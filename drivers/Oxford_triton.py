from qcodes.instrument.ip import IPInstrument
from qcodes.utils.validators import Bool, Numbers, Ints, Anything
from qcodes import validators as vals
from typing import Dict, Optional

import numpy as np

from qcodes.math_utils.field_vector import FieldVector
from functools import partial

class Oxford_triton(IPInstrument):

    #overwrite base class function
    def get_idn(self) -> Dict[str, Optional[str]]:
        return {'value': 'No IDN for you'}

    def write(self, cmd: str) -> None:
        resp = self.ask(cmd)
        if 'INVALID' in resp:
            raise ValueError('Invalid command. Got response: {}'.format(resp))


    def __init__(self, name, address, port,
                 **kwargs):

        super().__init__(name, address, port, terminator='\r\n',
                         write_confirmation=False, persistent=False, timeout = 10, **kwargs)

        self._parent_instrument = None

        self.add_parameter('MCTemp',
                           get_cmd='READ:DEV:T5:TEMP:SIG:TEMP',
                           get_parser = lambda val: float(val[26:-2]),
                           units = 'K',
                           set_cmd=None)

        self.add_parameter('OVCpressure',
                           get_cmd='READ:DEV:P6:PRES:SIG:PRES',
                           get_parser=lambda val: (val[26:-1]),
                           units='',
                           set_cmd=None)

        self.add_parameter(name='status',
                           label='Status',
                           get_cmd='READ:SYS:DR:STATUS',
                           get_parser=lambda val: (val[19:-1]),
                           set_cmd=None)

        self.add_parameter(name='PRST_aft_sweep',
                           label='Persistent on Completion',
                           get_cmd='READ:SYS:VRM:POC',
                           get_parser=lambda val: (val[17:-1]),
                           vals=vals.Enum('ON', 'OFF'),
                           set_cmd='SET:SYS:VRM:POC:{}')

        self.add_parameter(name='PRST_heater',
                           label='persistent heater',
                           get_cmd='READ:SYS:VRM:SWHT',
                           get_parser=lambda val: (val[18:-1]),
                           set_cmd=None)

        self.add_parameter(name='sweep_time',
                           label='time remaining before the actual value is the set value',
                           get_cmd='READ:SYS:VRM:RVST:TIME',
                           get_parser=lambda val: 60*float(val[23:-2]),
                           units='s',
                           set_cmd=None)

        self.add_parameter(name='MGN_status',
                           label='Magnet status',
                           get_cmd='READ:SYS:VRM:ACTN',
                           get_parser=lambda val: (val[18:22]),
                           set_cmd=None)

        self.add_parameter(name='MGN_coord_mode',
                           label='Magnet coordinate system',
                           get_cmd='READ:SYS:VRM:COO',
                           get_parser=lambda val: (val[17:-1]),
                           vals=vals.Enum('CART', 'CYL', 'SPH'),
                           set_cmd='SET:SYS:VRM:COO:{}')

        self.add_parameter(name='X',
                           label='Magnet field X',
                           get_cmd=partial(self._get_measured, 'x'),
                           set_cmd=None,
                           unit = 'T',
                           vals = Numbers())

        self.add_parameter(name='Y',
                           label='Magnet field Y',
                           get_cmd=partial(self._get_measured, 'y'),
                           set_cmd=None,
                           unit='T',
                           vals=Numbers())

        self.add_parameter(name='Z',
                           label='Magnet field Z',
                           get_cmd=partial(self._get_measured, 'z'),
                           set_cmd=None,
                           unit='T',
                           vals=Numbers())




    def _get_measured(self, *names):
        field = self.ask('READ:SYS:VRM:VECT')
        field = ''.join([x for x in field if x in '0123456789.- '])  # Remove all non numerics from string
         # Find position of spaces between the three values
        ind = [field.find(' '), field.find(' ', field.find(' ') + 1)]
        B1 = float(field[0:ind[0]])
        B2 = float(field[ind[0] + 1:ind[1]])
        B3 = float(field[ind[1] + 1:])

        # [COO:CART] = [X Y Z]
        # [COO:CYL] = [rho theta Z] ???? theta or phi????
        # [COO:SPH] = [r theta phi]

        original_coord_mode = self.MGN_coord_mode()

        if original_coord_mode == 'CART':
            measured_field = FieldVector(x=B1, y=B2, z=B3)
        elif original_coord_mode == 'SPH':
            measured_field = FieldVector(r=B1, theta=np.degrees(B2), phi=np.degrees(B3))
        elif original_coord_mode == 'CYL':
            measured_field = FieldVector(rho=B1, phi=np.degrees(B2), z=B3)
        else:
            raise ValueError('Unknown coordinate mode!')

        measured_values = measured_field.get_components(*names)

        # Convert angles from radians to degrees
        d = dict(zip(names, measured_values))

        # Do not do "return list(d.values())", because then there is
        # no guaranty that the order in which the values are returned
        # is the same as the original intention
        return_value = [d[name] for name in names]

        if len(names) == 1:
            return_value = return_value[0]

        return return_value


    def set_zero(self):
        msg = self.write('SET:SYS:VRM:ACTN:RTOZ')

    ## Magnet sweeps to zero in safe mode
    def set_zerosafe(self):
        # Initiates the vector-rotate magnet controller system to sweep zero as fast as possible
        # this command cannot be interrupted by [HOLD]
        msg = self.write('SET:SYS:VRM:ACTN:SAFE')