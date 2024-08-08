from time import sleep, time
import numpy as np
import qcodes as qc
from qcodes import (Instrument, VisaInstrument,
                    ManualParameter, MultiParameter,
                    validators as vals)


class Model_CS4(VisaInstrument):
    """
    QCoDeS driver for the Cryomagnetics CS4 magnet power supply
    """

    def __init__(self, name, address, **kwargs):
        # supplying the terminator means you don't need to remove it from every response
        super().__init__(name, address, terminator='\n', **kwargs)

        self.add_parameter('units',
                            get_cmd='UNITS?',
                            set_cmd='UNITS {}',
                            vals=vals.Enum('A', 'T', 'G', 'kG'))

        self.add_parameter('heater',
                           get_cmd='PSHTR?',
                           set_cmd='PSHTR {}',
                           get_parser = lambda val : 'on' if val == '1' else 'off',
                           set_parser = lambda val : 'ON' if val == 'on' else 'OFF',
                           vals=vals.Enum('on', 'off'))

        self.add_parameter('rate',
                            get_cmd='RATE? 0',
                            set_cmd= 'RATE 0 {:.4f}',
                            unit='A/s',
                            get_parser=float,
                            vals=vals.Numbers(min_value = 0.0, max_value = 0.012))

        self.add_parameter('tolerance',
                            get_cmd=None,
                            set_cmd=None,
                            unit='mT',
                            initial_value=5,
                            vals=vals.Numbers(min_value = 0.1, max_value = 100.0))

        self.add_parameter('field',
                            get_cmd='IMAG?',
                            set_cmd=self._set_field,
                            get_parser = lambda val: float(val.split(" ", 2)[0]))

        self.add_parameter('uplim',
                            get_cmd='ULIM?',
                            set_cmd='ULIM {}',
                            unit='T',
                            get_parser = lambda val: float(val.split(" ", 2)[0]),
                            vals=vals.Numbers(min_value = -9.0, max_value = 9.0))

        self.add_parameter('lowlim',
                            get_cmd='LLIM?',
                            set_cmd='LLIM {}',
                            unit='T',
                            get_parser = lambda val: float(val.split(" ", 2)[0]),
                            vals=vals.Numbers(min_value = -9.0, max_value = 9.0))

        self.connect_message()

    def _set_field(self, new_val):
        validator=vals.Numbers(min_value = -9.0, max_value = 9.0)
        validator.validate(new_val)
        current_field = self.field()
        self._check_heater()

        if current_field < new_val:
            self.uplim(new_val)
            self.write('SWEEP UP')
        else:
            self.lowlim(new_val)
            self.write('SWEEP DOWN')

        while np.abs((self.field() - new_val)) * 1000 > self.tolerance():
            sleep(1)

    def ramp_field(self, new_val):
        validator=vals.Numbers(min_value = -9.0, max_value = 9.0)
        validator.validate(new_val)
        current_field = self.field()
        self._check_heater()

        if current_field < new_val:
            self.uplim(new_val)
            self.write('SWEEP UP')
        else:
            self.lowlim(new_val)
            self.write('SWEEP DOWN')

    def _check_heater(self):
        if self.heater() != 'on':
            self.heater('on')
            sleep(10)
