from qcodes.dataset.measurements import Measurement
import numpy as np
from qcodes.instrument.specialized_parameters import ElapsedTimeParameter
from time import sleep
from tqdm.notebook import tqdm


def do2ddyn(param_set1, start1, stop1, num_points1, dyn_param, dyn_value1,dyn_value2, delay1,
          param_set2, start2, stop2, num_points2, delay2,
         *param_meas):
    # And then run an experiment

    meas = Measurement()
    meas.register_parameter(param_set1)
    meas.register_parameter(dyn_param)
    param_set1.post_delay = delay1
    dyn_param.post_delay = delay1
    meas.register_parameter(param_set2)
    param_set2.post_delay = delay2
    output = []
    for parameter in param_meas:
        meas.register_parameter(parameter, setpoints=(param_set1,param_set2))
        output.append([parameter, None])

    with meas.run() as datasaver:
        for set_point1 in tqdm(np.linspace(start1, stop1, num_points1), desc='first parameter'):
            param_set1.set(set_point1)
            dyn_set = set_point1*dyn_value1+dyn_value2
            dyn_param.set(dyn_set)
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


def do2ddyn_bias(param_set1, start1, stop1, num_points1, delay1,
          param_set2, start2, stop2, num_points2, delay2,
                 dyn_param, dyn_value1,delay3,
         *param_meas):
    # And then run an experiment

    meas = Measurement()
    meas.register_parameter(param_set1)
    meas.register_parameter(dyn_param)
    param_set1.post_delay = delay1
    dyn_param.post_delay = delay3
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
                dyn_set = set_point2*dyn_value1
                dyn_param.set(dyn_set)
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
         *param_meas,sensitivity_local = 3e-5,sensitivity_nonlocal = 1e-4,):
    
    '''
    this function is for measuring two probes concurrently;
    param_set1 will be the field/gate/..., param_set2 will be bias1, param_set3 will be bias2
    - add another param_set as another gate
    '''
    
    
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
            appl_voltage_AC_L1(5e-6) # Turning ac excitation for L1 on
            appl_voltage_AC_L2(4e-8) # Turning ac excitation for L2 off
            lockin_1.frequency(19.99)
            lockin_1.sensitivity(sensitivity_local)
            lockin_2.sensitivity(sensitivity_nonlocal)
            
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
            appl_voltage_AC_L1(4e-8) # Turning ac excitation for L1 off
            appl_voltage_AC_L2(5e-6) # Turning ac excitation for L2 on
            lockin_2.frequency(19.99) 
            lockin_1.sensitivity(sensitivity_nonlocal) 
            lockin_2.sensitivity(sensitivity_local)
            
            #Setting dc bias on top side to 0
            param_set2.set(0.0)
            param_set3.set(start3)
            
            sleep(3)
            
            for set_point3 in  tqdm(np.linspace(start3, stop3, num_points3), desc='nested  parameter', leave=False):
                param_set3.set(set_point3)
                for i, parameter in enumerate(param_meas):
                    output2[i][1] = parameter.get()
                datasaver2.add_result((param_set1, set_point1),
                                     (param_set3, set_point3),
                                     *output2)
            
            
    dataid1 = datasaver1.run_id  
    dataid2 = datasaver2.run_id
    return dataid1, dataid2


def three_terminal_dynamic_gates(param_set1, start1, stop1, num_points1, delay1,
         param_set2, start2, stop2, num_points2, delay2,
         param_set3, start3, stop3, num_points3, delay3,
         param_set4, start4, stop4, num_points4, 
         *param_meas,sensitivity_local = 2e-3,sensitivity_nonlocal = 5e-4,):
    
    '''
    this function is for measuring two probes concurrently;
    param_set1 will be the field/gate/..., param_set2 will be bias1, param_set3 will be bias2
    - add another param_set as another gate
    '''
    
    
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
    meas2.register_parameter(param_set4)
    param_set4.post_delay = delay1
    meas2.register_parameter(param_set3)
    param_set3.post_delay = delay3
    output2 = []
    for parameter in param_meas:
        meas2.register_parameter(parameter, setpoints=(param_set4,param_set3))
        output2.append([parameter, None])

    with meas1.run() as datasaver1, meas2.run() as datasaver2:

        for idx,set_point1 in enumerate(tqdm(np.linspace(start1, stop1, num_points1), desc='first parameter')):
            param_set1.set(set_point1)
            
            # Lockin 1 master, lockin 2 slave
            lockin_1.reference_source('internal')
            lockin_2.reference_source('external')
            
            #Lockin1 initialization
            appl_voltage_AC_L1(8e-6) # Turning ac excitation for L1 on
            appl_voltage_AC_L2(4e-8) # Turning ac excitation for L2 off
            lockin_1.frequency(19.99)
            lockin_1.sensitivity(sensitivity_local)
            lockin_2.sensitivity(sensitivity_nonlocal)
            
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
            appl_voltage_AC_L1(4e-8) # Turning ac excitation for L1 off
            appl_voltage_AC_L2(8e-6) # Turning ac excitation for L2 on
            lockin_2.frequency(19.99) 
            lockin_1.sensitivity(sensitivity_nonlocal) 
            lockin_2.sensitivity(sensitivity_local)
            
            #Setting dc bias on top side to 0
            param_set2.set(0.0)
            param_set3.set(start3)
            lst4 = np.linspace(start4,stop4,num_points4)
            param_set4.set(lst4[idx])
            sleep(3)
            
            for set_point3 in  tqdm(np.linspace(start3, stop3, num_points3), desc='nested  parameter', leave=False):
                param_set3.set(set_point3)
                for i, parameter in enumerate(param_meas):
                    output2[i][1] = parameter.get()
                datasaver2.add_result((param_set4, lst4[idx]),
                                     (param_set3, set_point3),
                                     *output2)
            
            
    dataid1 = datasaver1.run_id  
    dataid2 = datasaver2.run_id
    return dataid1, dataid2



def concurrent_do2ddyn(func,param_set1, start1, stop1, num_points1,delay1,param_set2, start2, stop2, num_points2, delay2,dyn_param, delay3,*param_meas):
    
    '''
    The first one is used to measure the current components and thus generate the setpoints for the second msmt. 
    with that being said, the first one can be run very fast because only DC components are needed (0.05s), that's the first
    two components from param_meas

    - add if statement inside the loop, determine then whether to record left or right bias as the param_set_2
    - add delay3 for dyn_parameter (the corrected bias parameter) accounting for lock-in readout... so delay2 can be supershort say 0.05s
    - call the function at first 
    '''

    meas1 = Measurement()
    meas2 = Measurement()
    
    
    meas1.register_parameter(param_set1)
    meas1.register_parameter(param_set2)
    meas2.register_parameter(param_set1)
    meas2.register_parameter(dyn_param)
    meas2.register_parameter(param_set2)
    
    param_set1.post_delay = delay1
    param_set2.post_delay = delay2
    dyn_param.post_delay = delay3 
    
    #  ''
    
    if 'left' in param_set2.label:
        terminal = 'L'
    else:
        terminal = 'R'
    
    output1 = []
    output2 = []
    
    for parameter in param_meas[:2]:
        meas1.register_parameter(parameter, setpoints=(param_set1,param_set2))
        output1.append([parameter, None])
    for parameter in param_meas:
        meas2.register_parameter(parameter, setpoints=(param_set1,param_set2))
        output2.append([parameter, None])

    with meas1.run() as datasaver1, meas2.run() as datasaver2:
        for set_point1 in tqdm(np.linspace(start1, stop1, num_points1), desc='first parameter'):
            param_set1.set(set_point1)
            current_l = []
            current_r = []
            for set_point2 in  tqdm(np.linspace(start2, stop2, num_points2), desc='nested  parameter', leave=False):
                param_set2.set(set_point2)
                for i, parameter in enumerate(param_meas[:2]):
                    output1[i][1] = parameter.get()
                datasaver1.add_result((param_set1, set_point1),
                                     (param_set2, set_point2),
                                     *output1)
                current_l.append(output1[0][1])
                current_r.append(output1[1][1])
                
                # now , give new setpoints as set_point1 and dyn_set,
                # run the first line of the second msmt ###
            
                # now the question is: how to get current_l*c_r from *output or from datasaver1
                # solution: create new array current_l and current_r. Refresh them every line!
                # cite the voltage compensation function, generate new setpoints array #

            voltage_appl_l, voltage_appl_r = func(terminal,param_set1,set_point1,num_points2,current_l,current_r)
            
            if 'left' in param_set2.label:                  # to determine which one is used for the setpoints of param_set2
                for i,set_point2 in enumerate(tqdm(np.linspace(start2, stop2, num_points2), desc='nested  parameter', leave=False)):
                    param_set2.set(set_point2)
                    dyn_param.set(voltage_appl_r[i])
                    for i, parameter in enumerate(param_meas):
                        output2[i][1] = parameter.get()
                    datasaver2.add_result((param_set1, set_point1),
                                         (param_set2, set_point2),
                                         *output2)
            else:
                for i,set_point2 in enumerate(tqdm(np.linspace(start2, stop2, num_points2), desc='nested  parameter', leave=False)):
                    param_set2.set(set_point2)
                    dyn_param.set(voltage_appl_l[i])
                    for i, parameter in enumerate(param_meas):
                        output2[i][1] = parameter.get()
                    datasaver2.add_result((param_set1, set_point1),
                                         (param_set2, set_point2),
                                         *output2)

            param_set2.set(start2)
    # not sure whether this is correct...
    dataid1 = datasaver1.run_id  # convenient to have for plotting
    dataid2 = datasaver2.run_id  # convenient to have for plotting
    return dataid1,dataid2


def concurrent_do1ddyn(func,param_set1, start1, stop1, num_points1,delay1,delay2,dyn_param,
         *param_meas):
    
    '''
    The first one is used to measure the current components and thus generate the setpoints for the second msmt. 
    with that being said, the first one can be run very fast because only DC components are needed (0.05s), that's the first
    two components from param_meas

    - add if statement inside the loop, determine then whether to record left or right bias as the param_set_2
    - add delay3 for dyn_parameter (the corrected bias parameter) accounting for lock-in readout... so delay2 can be supershort say 0.05s
    - call the function at first 
    '''

    meas1 = Measurement()
    meas2 = Measurement()
    
    
    meas1.register_parameter(param_set1)
    meas2.register_parameter(param_set1)
    meas2.register_parameter(dyn_param)
    if 'left' in param_set1.label:
        terminal = 'L'
    else:
        terminal = 'R'
    param_set1.post_delay = delay1
    dyn_param.post_delay = delay2  
    
    output1 = []
    output2 = []

    for parameter in param_meas[:2]:
        meas1.register_parameter(parameter, setpoints=(param_set1,))
        output1.append([parameter, None])
    for parameter in param_meas:
        meas2.register_parameter(parameter, setpoints=(param_set1,))
        output2.append([parameter, None])

    with meas1.run() as datasaver1, meas2.run() as datasaver2:
        current_l = []
        current_r = []
        for set_point1 in tqdm(np.linspace(start1, stop1, num_points1)):
            param_set1.set(set_point1)
            for i, parameter in enumerate(param_meas[:2]):
                output1[i][1] = parameter.get()
            datasaver1.add_result((param_set1, set_point1),
                                 *output1)
            current_l.append(output1[0][1])
            current_r.append(output1[1][1])
                
                # now , give new setpoints as set_point1 and dyn_set,
                # run the first line of the second msmt ###

        voltage_appl_l, voltage_appl_r = func(terminal, num_points1,current_l,current_r)
        if 'left' in param_set1.label:                  # to determine which one is used for the setpoints of param_set2
            for i,set_point1 in enumerate(tqdm(np.linspace(start1, stop1, num_points1))):
                param_set1.set(set_point1)
                dyn_param.set(voltage_appl_r[i])
                for i, parameter in enumerate(param_meas):
                    output2[i][1] = parameter.get()
                datasaver2.add_result((param_set1, set_point1),
                                     *output2)
        else:
            for i,set_point1 in enumerate(tqdm(np.linspace(start1, stop1, num_points1))):
                param_set1.set(set_point1)
                dyn_param.set(voltage_appl_l[i])
                for i, parameter in enumerate(param_meas):
                    output2[i][1] = parameter.get()
                datasaver2.add_result((param_set1, set_point1),
                                     *output2)

    dataid1 = datasaver1.run_id  # convenient to have for plotting
    dataid2 = datasaver2.run_id  # convenient to have for plotting
    return dataid1, dataid2




def do2d_simul(param_set1, start1, stop1, num_points1, delay1,
         param_set2, start2, stop2, num_points2, delay2,
             param_set3,  
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
                param_set3.set(set_point2)
                for i, parameter in enumerate(param_meas):
                    output[i][1] = parameter.get()
                datasaver.add_result((param_set1, set_point1),
                                     (param_set2, set_point2),
                                     *output)
            param_set2.set(start2)
            param_set3.set(start2)
    dataid = datasaver.run_id  # convenient to have for plotting
    return dataid


def three_terminal_multiprobe_simultaneous(param_set1, start1, stop1, num_points1, delay1,
         param_set2, start2, stop2, num_points2, delay2,
         param_set3,
        *param_meas,sensitivity_local = 2e-3,AC_exc = 5e-6,dc_filter = True):
    


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
                param_set3.set(set_point2)
                param_set2.set(set_point2) #v_bias_R by default

                for i, parameter in enumerate(param_meas):
                    output[i][1] = parameter.get()
                datasaver.add_result((param_set1, set_point1),
                                     (param_set2, set_point2),
                                     *output)
            param_set2.set(start2)
            param_set3.set(start2)
    dataid = datasaver.run_id  # convenient to have for plotting
    return dataid
