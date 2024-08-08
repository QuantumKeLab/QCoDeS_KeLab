from qcodes.dataset.measurements import Measurement
import numpy as np
from qcodes.instrument.specialized_parameters import ElapsedTimeParameter
from time import sleep
from tqdm.notebook import tqdm


#param_set1 is the virtual parameter, param_set2 and param_set3 are the two real parameters,

#param_set2=param_set1*cos(alpha), param_set3=param_set1*sin(alpha), thus alpha=0 degrees corresponds to 
#a full sweep of param_set2 and alpha=90 degrees to a full sweep of param_set3 

#alpha has to be given in degrees!

def do1d_vir_gate(param_set, start, stop, num_points, delay, param_set2, param_set3, alpha, *param_meas):
    meas = Measurement()
    meas.register_parameter(param_set)  # register the first independent parameter
    output = []
    param_set2.post_delay = delay
    param_set3.post_delay = delay
    # do1D enforces a simple relationship between measured parameters
    # and set parameters. For anything more complicated this should be reimplemented from scratch
    for parameter in param_meas:
        meas.register_parameter(parameter, setpoints=(param_set,))
        output.append([parameter, None])

    with meas.run() as datasaver:
        for set_point in tqdm(np.linspace(start, stop, num_points)):
            x = set_point*np.cos(np.deg2rad(alpha))
            y = set_point*np.sin(np.deg2rad(alpha))
            param_set2.set(x)
            param_set3.set(y)
            for i, parameter in enumerate(param_meas):
                output[i][1] = parameter.get()
            datasaver.add_result((param_set, set_point),
                                 *output)
    dataid = datasaver.run_id  # convenient to have for plotting
    return dataid

def do1d_vir_gate_off(param_set, start, stop, num_points, delay, param_set2, param_set3, offset1, offset2, alpha, *param_meas):
    meas = Measurement()
    meas.register_parameter(param_set)  # register the first independent parameter
    output = []
    param_set2.post_delay = delay
    param_set3.post_delay = delay
    # do1D enforces a simple relationship between measured parameters
    # and set parameters. For anything more complicated this should be reimplemented from scratch
    for parameter in param_meas:
        meas.register_parameter(parameter, setpoints=(param_set,))
        output.append([parameter, None])

    with meas.run() as datasaver:
        for set_point in tqdm(np.linspace(start, stop, num_points)):
            x = offset1 + set_point*np.cos(np.deg2rad(alpha))
            y = offset2 + set_point*np.sin(np.deg2rad(alpha))
            param_set2.set(x)
            param_set3.set(y)
            for i, parameter in enumerate(param_meas):
                output[i][1] = parameter.get()
            datasaver.add_result((param_set, set_point),
                                 *output)
    dataid = datasaver.run_id  # convenient to have for plotting
    return dataid


#param_set1 is the virtual parameter, param_set3 and param_set4 are the two real parameters,

#param_set3=param_set1*cos(alpha), param_set4=param_set1*sin(alpha), thus alpha=0 degrees corresponds to 
#a full sweep of param_set3 and alpha=90 degrees to a full sweep of param_set4 

#alpha has to be given in degrees!

def do2d_vir_gate(param_set1, start1, stop1, num_points1, delay1,
         param_set3, param_set4, alpha, param_set2, start2, stop2, num_points2, delay2,
         *param_meas):
    # And then run an experiment

    meas = Measurement()
    meas.register_parameter(param_set1)
    param_set3.post_delay = delay1
    param_set4.post_delay = delay1
    meas.register_parameter(param_set2)
    param_set2.post_delay = delay2
    output = []
    for parameter in param_meas:
        meas.register_parameter(parameter, setpoints=(param_set1,param_set2))
        output.append([parameter, None])

    with meas.run() as datasaver:
        for set_point1 in tqdm(np.linspace(start1, stop1, num_points1), desc='first parameter'):
            x = set_point1*np.cos(np.deg2rad(alpha))
            y = set_point1*np.sin(np.deg2rad(alpha))
            param_set3.set(x)
            param_set4.set(y)
            for set_point2 in  tqdm(np.linspace(start2, stop2, num_points2), desc='nested  parameter', leave=False):
                param_set2.set(set_point2)
                for i, parameter in enumerate(param_meas):
                    output[i][1] = parameter.get()
                datasaver.add_result((param_set1, set_point1),
                                     (param_set2, set_point2),
                                     *output)
            param_set2.set(start2)
    dataid = datasaver.run_id  # convenient to have for plotting
    return dataid



def do2d_vir_gate_off(param_set1, start1, stop1, num_points1, delay1,
         param_set3, param_set4, offset_1, offset_2, alpha, param_set2, start2, stop2, num_points2, delay2,
         *param_meas):
    # And then run an experiment

    meas = Measurement()
    meas.register_parameter(param_set1)
    param_set3.post_delay = delay1
    param_set4.post_delay = delay1
    meas.register_parameter(param_set2)
    param_set2.post_delay = delay2
    output = []
    for parameter in param_meas:
        meas.register_parameter(parameter, setpoints=(param_set1,param_set2))
        output.append([parameter, None])

    with meas.run() as datasaver:
        for set_point1 in tqdm(np.linspace(start1, stop1, num_points1), desc='first parameter'):
            x = offset_1 + (set_point1*np.cos(np.deg2rad(alpha)))
            y = offset_2 + (set_point1*np.sin(np.deg2rad(alpha)))
            param_set3.set(x)
            param_set4.set(y)
            for set_point2 in  tqdm(np.linspace(start2, stop2, num_points2), desc='nested  parameter', leave=False):
                param_set2.set(set_point2)
                for i, parameter in enumerate(param_meas):
                    output[i][1] = parameter.get()
                datasaver.add_result((param_set1, set_point1),
                                     (param_set2, set_point2),
                                     *output)
            param_set2.set(start2)
    dataid = datasaver.run_id  # convenient to have for plotting
    return dataid


def do1d_repeat(param_set1, num_points1, delay1,
         param_set2, start2, stop2, num_points2, delay2,
         *param_meas):
    # And then run an experiment

    meas = Measurement()
    meas.register_parameter(param_set1)
    param_set1.post_delay = delay1
    meas.register_parameter(param_set2)
    param_set2.post_delay = delay2
    output = []
    for parameter in param_meas:
        meas.register_parameter(parameter, setpoints=(param_set1,param_set2))
        output.append([parameter, None])

    with meas.run() as datasaver:
        for set_point1 in tqdm(range(num_points1), desc='first parameter'):
            param_set1.set(set_point1)
            for set_point2 in  tqdm(np.linspace(start2, stop2, num_points2), desc='nested  parameter', leave=False):
                param_set2.set(set_point2)
                for i, parameter in enumerate(param_meas):
                    output[i][1] = parameter.get()
                datasaver.add_result((param_set1, (set_point1+1)),
                                     (param_set2, set_point2),
                                     *output)
            param_set2.set(start2)
    dataid = datasaver.run_id  # convenient to have for plotting
    return dataid


def do1d_twoways_repeat(param_set1, num_points1, delay1,
         param_set2, start2, stop2, num_points2, delay2,
         *param_meas):
    # And then run an experiment

    meas = Measurement()
    meas.register_parameter(param_set1)
    param_set1.post_delay = delay1
    meas.register_parameter(param_set2)
    param_set2.post_delay = delay2
    output = []
    set_point_count=1
    
    for parameter in param_meas:
        meas.register_parameter(parameter, setpoints=(param_set1,param_set2))
        output.append([parameter, None])

    with meas.run() as datasaver:
        for set_point1 in tqdm(range(num_points1), desc='first parameter'):
            param_set1.set(set_point_count)
            for set_point2 in  tqdm(np.linspace(start2, stop2, num_points2), desc='nested  parameter', leave=False):
                param_set2.set(set_point2)
                for i, parameter in enumerate(param_meas):
                    output[i][1] = parameter.get()
                datasaver.add_result((param_set1, set_point_count),
                                     (param_set2, set_point2),
                                     *output)
            set_point_count = set_point_count+1
            for set_point2 in  tqdm(np.linspace(stop2, start2, num_points2), desc='nested  parameter', leave=False):
                param_set2.set(set_point2)
                for i, parameter in enumerate(param_meas):
                    output[i][1] = parameter.get()
                datasaver.add_result((param_set1, set_point_count),
                                     (param_set2, set_point2),
                                     *output)
            set_point_count = set_point_count+1
    dataid = datasaver.run_id  # convenient to have for plotting
    return dataid

def do2d(param_set1, start1, stop1, num_points1, delay1,
         param_set2, start2, stop2, num_points2, delay2,
         *param_meas):
    # And then run an experiment

    meas = Measurement()
    meas.register_parameter(param_set1)
    param_set1.post_delay = delay1
    meas.register_parameter(param_set2)
    param_set2.post_delay = delay2
    output = []
    for parameter in param_meas:
        meas.register_parameter(parameter, setpoints=(param_set1,param_set2))
        output.append([parameter, None])

    with meas.run() as datasaver:
        for set_point1 in tqdm(np.linspace(start1, stop1, num_points1), desc='first parameter'):
            param_set1.set(set_point1)
            for set_point2 in  tqdm(np.linspace(start2, stop2, num_points2), desc='nested  parameter', leave=False):
                param_set2.set(set_point2)
                for i, parameter in enumerate(param_meas):
                    output[i][1] = parameter.get()
                datasaver.add_result((param_set1, set_point1),
                                     (param_set2, set_point2),
                                     *output)
            param_set2.set(start2)
    dataid = datasaver.run_id  # convenient to have for plotting
    return dataid

def three_terminal_c(param_set1, start1, stop1, num_points1, delay1,
         param_set2, start2, stop2, num_points2, delay2,
         param_set3, start3, stop3, num_points3, delay3,
         *param_meas):
    # And then run an experiment

    meas1 = Measurement()
    meas1.register_parameter(param_set1)
    param_set1.post_delay = delay1
    meas1.register_parameter(param_set2)
    param_set2.post_delay = delay2
    output1 = []
    for parameter in param_meas:
        meas1.register_parameter(parameter, setpoints=(param_set1,param_set2))
        output1.append([parameter, None])
    
    meas2 = Measurement()
    meas2.register_parameter(param_set1)
    #param_set1.post_delay = delay1
    meas2.register_parameter(param_set3)
    param_set3.post_delay = delay3
    output2 = []
    for parameter in param_meas:
        meas2.register_parameter(parameter, setpoints=(param_set1,param_set3))
        output2.append([parameter, None])

    with meas1.run() as datasaver1, meas2.run() as datasaver2:
        for set_point1 in tqdm(np.linspace(start1, stop1, num_points1), desc='first parameter'):
            param_set1.set(set_point1)
            
            # Lockin 1 master, lockin 2 slave
            lockin_1.reference_source('internal')
            lockin_2.reference_source('external')
            
            #Lockin1 initialization
            appl_voltage_ac_L1(5e-6) # 4e-8 is zero
            appl_voltage_ac_L2(4e-8) # 4e-8 is zero
            lockin_1.frequency(13.711)
            lockin_1.sensitivity(1e-2)
            lockin_2.sensitivity(1e-3)
            
            #Setting dc bias on bottom side to 0
            param_set3.set(0.0)
            param_set2.set(start2)
            
            sleep(3)
                
            for set_point2 in  tqdm(np.linspace(start2, stop2, num_points2), desc='nested  parameter', leave=False):
                param_set2.set(set_point2)
                for i, parameter in enumerate(param_meas):
                    output1[i][1] = parameter.get()
                datasaver1.add_result((param_set1, set_point1),
                                     (param_set2, set_point2),
                                     *output1)
            
            # Lockin 1 slave, lockin 2 master
            lockin_1.reference_source('external')
            lockin_2.reference_source('internal')
            
            #Lockin2 initialization
            appl_voltage_ac_L1(4e-8) # 4e-8 is zero
            appl_voltage_ac_L2(5e-6) # 4e-8 is zero
            lockin_2.frequency(13.711)
            lockin_1.sensitivity(1e-3)
            lockin_2.sensitivity(1e-2)
            
            #Setting dc bias on top side to 0
            param_set2.set(0.0)
            param_set3.set(start3)
            
            sleep(3)
            
            # Need to set ac excitation to 0 as well
            
            for set_point3 in  tqdm(np.linspace(start3, stop3, num_points3), desc='nested  parameter', leave=False):
                param_set3.set(set_point3)
                for i, parameter in enumerate(param_meas):
                    output2[i][1] = parameter.get()
                datasaver2.add_result((param_set1, set_point1),
                                     (param_set3, set_point3),
                                     *output2)
            
            
    dataid1 = datasaver1.run_id  # convenient to have for plotting
    dataid2 = datasaver2.run_id
    return dataid1, dataid2