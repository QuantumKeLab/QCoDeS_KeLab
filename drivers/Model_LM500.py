from time import sleep, time
from datetime import datetime
from typing import ClassVar, Dict
import numpy as np
import qcodes as qc
from qcodes import (Instrument, VisaInstrument,
                    ManualParameter, MultiParameter,
                    validators as vals)


class Model_LM500(VisaInstrument):
    """
    QCoDeS driver for the Cryomagnetics LM510 level meter
    """

    MODES: ClassVar[Dict[str, str]] = {
        'Disabled' : '0.0',
        'Sample/Hold' : 'S',
        'Continuous' : 'C'}

    def __init__(self, name, address, **kwargs):
        # supplying the terminator means you don't need to remove it from every response
        super().__init__(name, address, terminator='\n', **kwargs)

        self.add_parameter('lastval',
                            get_cmd='MEAS?',
                            set_cmd=False,
                            units='in',
                            get_parser = lambda val: float(val.split(" ", 2)[0]))

        self.add_parameter('interval',
                            get_cmd='INTVL?',
                            set_cmd='INTVL {}',
                            set_parser = lambda val: datetime.strptime(val, "%H:%M:%S").strftime("%H:%M:%S"),
                            units='')

        self.add_parameter('sample_mode',
                            get_cmd='MODE?',
                            set_cmd='MODE {}',
                            set_parser = lambda val : self.MODES[val],
                            vals=vals.Enum('Sample/Hold', 'Continuous','Disabled'),
                            units='')

        self.connect_message()

