from time import sleep, time
import numpy as np
import qcodes as qc
from qcodes import (Instrument, VisaInstrument,
                    ManualParameter, MultiParameter,
                    validators as vals)
from qcodes import InstrumentChannel

class MagnetSingleChannel(InstrumentChannel):

    def __init__(self, parent, name, channel, field_lim):
        super().__init__(parent, name)
        self._channel = channel  # Channel on the temperature controller

        self.add_parameter('heater',
                    get_cmd='PSHTR?',
                    set_cmd='PSHTR {}',
                    get_parser = lambda val : 'on' if val == '1' else 'off',
                    set_parser = lambda val : 'ON' if val == 'on' else 'OFF',
                    vals=vals.Enum('on', 'off'))

        self.add_parameter('internal_units',
                    get_cmd='UNITS?',
                    set_cmd='UNITS {}',
                    vals=vals.Enum('A', 'G'))

        self.add_parameter('units',
                    get_cmd=None,
                    set_cmd=None,
                    initial_value='T',
                    vals=vals.Enum('T'))

        self.add_parameter('rate',
                            get_cmd='RATE? 0',
                            set_cmd= 'RATE 0 {:.4f}',
                            unit='A/s',
                            get_parser=float,
                            vals=vals.Numbers(min_value = 0.0, max_value = 0.05))

        self.add_parameter('tolerance',
                            get_cmd=None,
                            set_cmd=None,
                            units='mT',
                            initial_value=5,
                            vals=vals.Numbers(min_value = 0.1, max_value = 100.0))

        self.add_parameter('field_lim',
                            get_cmd=None,
                            set_cmd=None,
                            units='T',
                            initial_value=field_lim,
                            vals=vals.Numbers(min_value = 0.0, max_value = 9.0))

        self.add_parameter('field',
                            get_cmd='IMAG?',
                            set_cmd=self._set_field,
                            get_parser = lambda val: 1e-1 * float( val[:-2] ) )

        self.add_parameter('uplim',
                            get_cmd='ULIM?',
                            set_cmd='ULIM {}',
                            units='T',
                            get_parser = lambda val: 1e-1 * float( val[:-2] ),
                            set_parser = lambda val : 1e1 * val,
                            vals=vals.Numbers(min_value = -self.field_lim(), max_value = self.field_lim()))

        self.add_parameter('lowlim',
                            get_cmd='LLIM?',
                            set_cmd='LLIM {}',
                            units='T',
                            get_parser = lambda val: 1e-1 * float( val[:-2] ),
                            set_parser = lambda val : 1e1 * val,
                            vals=vals.Numbers(min_value = -self.field_lim(), max_value = self.field_lim()))

    def write(self, cmd: str) -> None:
        self._parent.write('CHAN {}'.format(self._channel))
        return self._parent.write(cmd)

    def ask(self, cmd: str) -> str:
        self._parent.write('CHAN {}'.format(self._channel))
        return self._parent.ask(cmd)

    def _set_field(self, new_val):
        validator=vals.Numbers(min_value = -self.field_lim(), max_value = self.field_lim())
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

    def _check_heater(self):
        if self.heater() != 'on':
            self.heater('on')
            sleep(10)


class Model_4G(VisaInstrument):
    """
    Lakeshore Model 332 Temperature Controller Driver
    """

    def __init__(self, name, address, **kwargs):
        # supplying the terminator means you don't need to remove it from every response
        super().__init__(name, address, terminator='\n', **kwargs)

        self.channel_A = MagnetSingleChannel(self, 'A', 1, 4)
        self.channel_B = MagnetSingleChannel(self, 'B', 2, 9)

        self.add_submodule('A', self.channel_A)
        self.add_submodule('B', self.channel_B)

        self.connect_message()
