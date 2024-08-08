import sys
import runpy
import os

import qcodes as qc

station = qc.Station()

fridge_name = sys.argv[1]

#DO NOT MODIFY DRIVERS!!!
sys.path.append("M:/tnw/ns/qt/2D Topo/code/drivers")


if fridge_name == 'VectorJanis':
    exec(open('C:/Users/TUD210595/Documents/LOCAL_Qcodes/scripts/init_VectorJanis.py').read())   
else:
    print('Local file only accepts VectorJanis!')
