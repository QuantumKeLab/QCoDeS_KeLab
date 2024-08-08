from typing import ClassVar, Dict, Any
import time
from qcodes.instrument.group_parameter import GroupParameter, Group
from qcodes.instrument_drivers.Lakeshore.lakeshore_base import LakeshoreBase, BaseOutput, BaseSensorChannel
import qcodes.utils.validators as vals
from qcodes import InstrumentChannel


# There are 4 sensors channels (a.k.a. measurement inputs) in Model 332.

_channel_name_to_command_map: Dict[str, str] = {'A': 'A',
                                                'B': 'B'}

class Output_332(InstrumentChannel):
    """
    Class for control output of model 340
    """

    MODES: ClassVar[Dict[str, int]] = {
        'Manual PID': 1,
        'Zone': 2,
        'Open Loop': 3,
        'AutoTune PID': 4,
        'AutoTune PI': 5,
        'AutoTune P': 6}

    RANGES: ClassVar[Dict[str, int]] = {
        'off': 0,
        '0.5W': 1,
        '5W': 2,
        '50W': 3} 

    def __init__(self, parent) \
            -> None:
        super().__init__(parent, 'heater_output')

        self.INVERSE_RANGES: Dict[int, str] = {
            v: k for k, v in self.RANGES.items()}

        output_index = 1

        self._has_pid = True
        self._output_index = output_index

        self.add_parameter('mode',
                           label='Control mode',
                           docstring='Specifies the control mode',
                           val_mapping=self.MODES,
                           get_cmd=f'CMODE? {output_index}',
                           set_cmd=f'CMODE {output_index} {{mode}}')

        self.add_parameter('input_channel',
                           label='Input channel',
                           docstring='Specifies which measurement input to '
                                     'control from (note that only '
                                     'measurement inputs are available)',
                           parameter_class=GroupParameter)
        #parameter for setpoints units always 1 (Kelvin)
        #control loop is always on 1
        self.add_parameter('powerup_enable',
                           label='Power-up enable on/off',
                           docstring='Specifies whether the output remains on '
                                     'or shuts off after power cycle.',
                           val_mapping={True: 1, False: 0},
                           parameter_class=GroupParameter)

        self.output_group = Group([self.input_channel, self.powerup_enable],
                                  set_cmd=f'CSET {output_index},{{input_channel}},'
                                          f'1,1,'
                                          f'{{powerup_enable}}',
                                  get_cmd=f'CSET? {output_index}')

        # Parameters for Closed Loop PID Parameter Command
        self.add_parameter('P',
                           label='Proportional (closed-loop)',
                           docstring='The value for closed control loop '
                                     'Proportional (gain)',
                           vals=vals.Numbers(0, 1000),
                           get_parser=float,
                           parameter_class=GroupParameter)
        self.add_parameter('I',
                           label='Integral (closed-loop)',
                           docstring='The value for closed control loop '
                                     'Integral (reset)',
                           vals=vals.Numbers(0, 1000),
                           get_parser=float,
                           parameter_class=GroupParameter)
        self.add_parameter('D',
                           label='Derivative (closed-loop)',
                           docstring='The value for closed control loop '
                                     'Derivative (rate)',
                           vals=vals.Numbers(0, 1000),
                           get_parser=float,
                           parameter_class=GroupParameter)
        self.pid_group = Group([self.P, self.I, self.D],
                               set_cmd=f'PID {output_index}, '
                                       f'{{P}}, {{I}}, {{D}}',
                               get_cmd=f'PID? {output_index}')

        self.add_parameter('output_range',
                           label='Heater range',
                           docstring='Specifies heater output range. The range '
                                     'setting has no effect if an output is in '
                                     'the `Off` mode, and does not apply to '
                                     'an output in `Monitor Out` mode. '
                                     'An output in `Monitor Out` mode is '
                                     'always on.',
                           val_mapping=self.RANGES,
                           set_cmd=f'RANGE {output_index}, {{}}',
                           get_cmd=f'RANGE? {output_index}')

        self.add_parameter('output',
                           label='Output',
                           unit='% of heater range',
                           docstring='Specifies heater output in percent of '
                                     'the current heater output range.\n'
                                     'Note that when the heater is off, '
                                     'this parameter will return the value of 0.005.',
                           get_parser=float,
                           get_cmd=f'HTR? {output_index}',
                           set_cmd=False)

        self.add_parameter('setpoint',
                           label='Setpoint value (in sensor units)',
                           docstring='The value of the setpoint in the '
                                     'preferred units of the control loop '
                                     'sensor (which is set via '
                                     '`input_channel` parameter)',
                           vals=vals.Numbers(0, 400),
                           get_parser=float,
                           set_cmd=f'SETP {output_index}, {{}}',
                           get_cmd=f'SETP? {output_index}')
    def wait_until_set_point_reached(self,
                                     wait_cycle_time,
                                     wait_tolerance,
                                     wait_equilibration_time):
        """
        This function runs a loop that monitors the value of the heater's
        input channel until the read values is close to the setpoint value
        that has been set before.

        Note that if the setpoint value is in a different range,
        this function may wait forever because that setpoint cannot be
        reached within the current range.

        Args:
            wait_cycle_time
                this time is being waited between the readings
            wait_tolerance
                this value is used to determine if the reading value is
                close enough to the setpoint value according to the
                following formula:
                `abs(t_reading - t_setpoint)/t_reading < wait_tolerance`
            wait_equilibration_time:
                within this time, the reading value has to stay within the
                defined tolerance in order for this function to return (same as
                `wait_equilibration_time` parameter);
        """

        active_channel_id = self.input_channel()
        active_channel = getattr(self.root_instrument, active_channel_id)

        t_setpoint = self.setpoint()

        time_now = time.perf_counter()
        time_enter_tolerance_zone = time_now

        while time_now - time_enter_tolerance_zone < equilibration_time:
            time_now = time.perf_counter()

            t_reading = active_channel.temperature()

            if abs(t_reading - t_setpoint) / t_reading > tolerance:
                # Reset time_enter_tolerance_zone to time_now because we left
                # the tolerance zone here (if we even were inside one)
                time_enter_tolerance_zone = time_now

            time.sleep(wait_cycle_time)

class Model_332_Channel(InstrumentChannel):

    def __init__(self, parent, name, channel):
        super().__init__(parent, name)

        self._channel = channel  # Channel on the temperature controller

        # Add the various channel parameters

        self.add_parameter('temperature',
                           get_cmd='KRDG? {}'.format(self._channel),
                           get_parser=float,
                           label='Temperature',
                           unit='K')

        self.add_parameter('sensor_raw',
                           get_cmd=f'SRDG? {self._channel}',
                           get_parser=float,
                           label='Raw reading',
                           unit='Ohms')


class Model_332(LakeshoreBase):
    """
    Lakeshore Model 332 Temperature Controller Driver
    """
    channel_name_command: Dict[str, str] = _channel_name_to_command_map

    CHANNEL_CLASS = Model_332_Channel

    input_channel_parameter_values_to_channel_name_on_instrument = \
        _channel_name_to_command_map

    def __init__(self, name: str, address: str, **kwargs) -> None:
        super().__init__(name, address, **kwargs)

        self.output_1 = Output_332(self)
        self.add_submodule('heater_output', self.output_1)
