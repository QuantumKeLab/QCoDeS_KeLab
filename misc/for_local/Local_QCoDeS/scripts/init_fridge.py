import sys
import runpy
import os

import qcodes as qc

station = qc.Station()

fridge_name = sys.argv[1]

#DO NOT MODIFY DRIVERS!!!
sys.path.append("M:/tnw/ns/qt/2D Topo/code/drivers")



if fridge_name == 'Gecko':
    exec(open('C:/Users/TUD206951/Documents/Local_QCoDeS/scripts/init_Gecko.py').read())        
else:
    print('Local file only accepts Gecko!')
