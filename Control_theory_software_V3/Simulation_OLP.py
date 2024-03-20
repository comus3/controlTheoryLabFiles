import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib import colors as mcolors



def plot_data(FOPDT,MVm,PVm,tm,TSim,):
    """
    Plots the step responses
    of all different models
    """
    FOPDT_P_B1 = FOPDT(MVm, Kp, T_B1, theta_B1, Ts)
    FOPDT_P_B2 = FOPDT(MVm, Kp, T_B2, theta_B2, Ts)
    FOPDT_P_T = FOPDT(MVm, Kp, T_T, theta_T, Ts)
    #FOPDT_P_VDR = FOPDT(MVm, Kp, T1_VDG, T2_VDG, theta_VDG)

    plt.figure(figsize = (15,9))

    plt.subplot(2,1,2)
    plt.step(tm,PVm,'green',label='Real',where='post')
    plt.step(tm,FOPDT_P_B1,'purple',label='FOPDT Broida 1',where='post')
    plt.step(tm,FOPDT_P_B2,'red',label='FOPDT Broida 2',where='post')
    plt.step(tm,FOPDT_P_T,'blue',label='FOPDT Strejc',where='post')
    plt.ylabel('Value of PV')
    plt.xlabel('Time [s]')
    plt.legend(loc='best')
    plt.xlim([0, TSim])    

def initialize_constants():
    """
    Initializes and returns the constants Tu, Tg, Kp, a, t1, t2 used in the control theory laboratory.
    Usage : Tu, Tg, Kp, a, t1, t2 = initialize_constants()

    Returns:
    Tu (float): Ultimate period.
    Tg (float): Ultimate gain.
    Kp (float): Process gain.
    a (float): Constant a.
    t1 (float): Constant t1.
    t2 (float): Constant t2.

    which are obtained from the open loop process transfer function grphcally.
    """
    Tu = 14.0  
    Tg = 154.0  
    Kp = 39.1  # Constant 
    a = 13.0   # Constant a
    t1 = 58.0 # Constant t1
    t2 = 85.0  # Constant t2
    
    return Tu, Tg, Kp, a, t1, t2

def Broida1():
    """
    create step response of the system defined by the transfer function Pb(s) = (Kp*e^(-theta*s))/(Ts+1)
    with T=Tg and theta = Tu.

    Parameters:
    """
    global T_B1
    global theta_B1
    # Get constants from initialize_constants method
    T_B1 = Tg
    theta_B1 = Tu

    return (Kp*math.exp(-1*theta_B1)/(Ts + 1))

def Broida2():
    """
    create step response of the system defined by the transfer function Pb(s) = (Kp*e^(-theta*s))/(Ts+1)
    with T=Tg and theta = Tu.

    Parameters:
    """
    global theta_B2
    global T_B2
    # Get constants from initialize_constants method
    T_B2 = 5.5*(t2 - t1) 
    theta_B2 = (2.8*t1) - (1.8*t2)

    return (Kp*math.exp(-1*theta_B2)/(Ts + 1))
def Strejc():
    global T_T
    global T_u_th_T
    global a_n
    global b_n
    global n
    global base_T
    global theta_T

    a_n_test = Tu / Tg # = 0.0655 => a_n = 0 =< 0.0655 < a_n+1 = 1  => n = 1
    print(a_n_test)

    n = 1
    a_n = 0
    b_n = 1

    T_T = Tg / b_n
    T_u_th_T = a_n * Tg
    theta_T = Tu - T_u_th_T

    base_T = Ts + 1
    return T_T
def VanDerGrinden():
    global T1_VDG
    global T2_VDG
    global theta_VDG

    e = 2.71828
    T1_VDG = Tg * (3*a*e - 1) / (1 + a*e)
    T2_VDG = Tg * (1 - a*e) / (1 + a*e)
    theta_VDG = Tu - (T1_VDG * T2_VDG) / (T1_VDG + 3*T2_VDG)

    return (Kp * math.exp(-theta_VDG)) / ((T1_VDG + 1)+(T2_VDG + 1))
Tu, Tg, Kp, a, t1, t2 = initialize_constants()
Ts = 1






