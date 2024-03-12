# -*- coding: utf-8 -*-
from PyFlow_ACDC import*
from PyFlow_ACDC_Results import*
 
from PyFlow_ACDC_PF import*
from PyFlow_ACDC_TS import*
import os
import subprocess


try:
    import pyomo
    from PyFlow_ACDC_OPF import *
except ImportError:
    print("Pyomo is not installed. Optimal power flow can not be done.")
    
    
def Run_example(example):
    try:
        os.chdir("examples")
        
        subprocess.run(["python", example], check=True)
    except FileNotFoundError:
        print('Example not found')
