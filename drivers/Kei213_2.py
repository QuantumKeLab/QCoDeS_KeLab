from qcodes import VisaInstrument, validators as vals
import numpy as np
from qcodes.utils.validators import Numbers



class K213(VisaInstrument):
    """
    This is the code for Keithley 213 Quad Voltage Source
    """

    def __init__(self, name, address, reset=False,  **kwargs):
        super().__init__(name, address,  terminator='\n', **kwargs)
# general commands
#        self.add_parameter(name='voltAutoRange',
#                           label='Voltage Auto Range setting',
#                           unit='V',
#                           get_cmd='P2 A? X',
#                           set_cmd='P2 A1 X',
#                           get_parser=float,
#                           vals=Numbers(min_value=0,
#                                        max_value=1))
#        self.add_parameter('rf_switch',
#                   label='Rf_switch',
#                           unit='ns',
#                   set_cmd='{}',
#                           get_cmd=!'PULM:INT:PWID?',
#                   val_mapping ={ 'ON' : 'RF1', 'OFF' :'RF0'})  # ON/OFF RF signal
    
        self.add_parameter(name='voltage1',
                           label='Voltage1',
                           unit='V',
                           get_cmd='P1 V? X',
                           set_cmd='P1 V{} X',                         
                           vals=Numbers(min_value=-10,
                                        max_value=10))    
        self.add_parameter(name='autorange1',
                           label='Autorange1',
                           unit='',
                           get_cmd='P1 A? X',
                           set_cmd='P1 A{} X',                         
                           vals=Numbers(min_value=0,
                                        max_value=1)) 
        self.add_parameter(name='vrange1',
                           label='VRange1',
                           unit='V',
                           get_cmd='P1 R? X',
                           set_cmd='P1 R{} X',                         
                           vals=Numbers(min_value=0,
                                        max_value=3)) 
        
        
        self.add_parameter(name='voltage2',
                           label='Voltage2',
                           unit='V',
                           get_cmd='P2 V? X',
                           set_cmd='P2 V{} X',
#                            get_parser = lambda s: float(s[1:]),
#                           set_parser= lambda x: x/1e9,
#                           get_parser= lambda x: float(x)*1e6 ,                           
                           vals=Numbers(min_value=-10,
                                        max_value=10))
        self.add_parameter(name='autorange2',
                           label='Autorange1',
                           unit='',
                           get_cmd='P2 A? X',
                           set_cmd='P2 A{} X',                         
                           vals=Numbers(min_value=0,
                                        max_value=1)) 
        self.add_parameter(name='vrange2',
                           label='VRange2',
                           unit='V',
                           get_cmd='P2 R? X',
                           set_cmd='P2 R{} X',                         
                           vals=Numbers(min_value=0,
                                        max_value=3)) 
        
        self.add_parameter(name='voltage3',
                           label='Voltage3',
                           unit='V',
                           get_cmd='P3 V? X',
                           set_cmd='P3 V{} X',                         
                           vals=Numbers(min_value=-10,
                                        max_value=10))
        self.add_parameter(name='autorange3',
                           label='Autorange3',
                           unit='',
                           get_cmd='P3 A? X',
                           set_cmd='P3 A{} X',                         
                           vals=Numbers(min_value=0,
                                        max_value=1)) 
        self.add_parameter(name='vrange3',
                           label='VRange3',
                           unit='V',
                           get_cmd='P3 R? X',
                           set_cmd='P3 R{} X',                         
                           vals=Numbers(min_value=0,
                                        max_value=3)) 
        
        self.add_parameter(name='voltage4',
                           label='Voltage4',
                           unit='V',
                           get_cmd='P4 V? X',
                           set_cmd='P4 V{} X',                         
                           vals=Numbers(min_value=-10,
                                        max_value=10))
        self.add_parameter(name='autorange4',
                           label='Autorange4',
                           unit='',
                           get_cmd='P4 A? X',
                           set_cmd='P4 A{} X',                         
                           vals=Numbers(min_value=0,
                                        max_value=1)) 
        self.add_parameter(name='vrange4',
                           label='VRange4',
                           unit='V',
                           get_cmd='P4 R? X',
                           set_cmd='P4 R{} X',                         
                           vals=Numbers(min_value=0,
                                        max_value=3)) 



# reset values after each reconnect
#        self.power(0)
#        self.power_offset(0)
#        self.connect_message()
#        self.add_function('reset', call_cmd='*RST')

 
if __name__ == "__main__":
 
    try:
#            ats_inst.close()
#            acquisition_controller.close()
            Instrument.close_all()
    except KeyError:
        pass    
    except NameError:
        pass    
    
#     Vgate =  K213(name = 'Vgate', address = "GPIB::9::INSTR")
#     Vgate.voltage.set(0.04 + 1*0.62)
#     print ("voltage =", Vgate.voltage.get(), "V")
#    Instrument.close_all()