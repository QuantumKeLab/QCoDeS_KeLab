from time import sleep, time
import numpy as np
import qcodes as qc
from qcodes import (Instrument, VisaInstrument,
                    ManualParameter, MultiParameter,
                    validators as vals)


class Model_LM510_USB(VisaInstrument):
    """
    QCoDeS driver for the Cryomagnetics LM510 level meter
    """

    def __init__(self, name, address, **kwargs):
        # supplying the terminator means you don't need to remove it from every response
        super().__init__(name, address, terminator='\n', **kwargs)
        self.add_parameter('lastval',
                            get_cmd='MEAS?',
                            set_cmd=False,
                            unit='in',
                            get_parser = lambda val: float(val.split(" ", 2)[0]))

    def write_raw(self, cmd: str) -> None:
        """
        Low-level interface to ``visa_handle.write``.
        Args:
            cmd: The command to send to the instrument.
        """
        self.visa_log.debug(f"Writing: {cmd}")

        nr_bytes_written, ret_code = self.visa_handle.write(cmd)
        echoed_value = self.visa_handle.read_raw().decode('ASCII').rstrip()
        #print(f"Echoed command: {echoed_value}")
        self.visa_log.debug(f"Echoed command: {echoed_value}")
        self.check_error(ret_code)

    def ask_raw(self, cmd: str) -> str:
        """
        Low-level interface to ``visa_handle.ask``.
        Args:
            cmd: The command to send to the instrument.
        Returns:
            str: The instrument's response.
        """
        self.visa_log.debug(f"Querying: {cmd}")
        echoed_value = self.visa_handle.query(cmd)
        response = self.visa_handle.read_raw().decode('ASCII').rstrip()
        #print(f"Response: {response}; Echoed command: {echoed_value}")
        self.visa_log.debug(f"Response: {response}; Echoed command: {echoed_value}")
        return response
