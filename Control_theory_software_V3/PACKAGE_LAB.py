import numpy as np

import matplotlib.pyplot as plt
from package_DBR import *
from IPython.display import display, clear_output

# -----------------------------------


def stability_margin(process_obj, controller_obj, omega_freq):
    """
    Calculate stability margins using the Bode method.

    :process_obj: Process object.
    :controller_obj: Controller object.
    :omega_freq: Frequency vector (rad/s).

    Returns stability margins.
    """
    # Generate the loop gain using the Bode method
    Ls = Bode(process_obj, omega_freq, Show=False)

    # Controller Transfer Function Components
    s = 1j * omega_freq
    integration_action = 1 / (controller_obj.parameters['Ti'] * s)
    Tfd = controller_obj.parameters['Td'] * controller_obj.parameters['alpha']
    derivative_action = controller_obj.parameters['Td'] * s / (1 + Tfd * s)
    Cs = controller_obj.parameters['Kc'] * (1 + integration_action + derivative_action)

    # Calculate overall loop gain
    Ls *= Cs

    print("Loop Gain Ls:", Ls)  

    # Gain and Phase Calculations
    gain_values = 20 * np.log10(np.abs(Ls))
    phase_values = (180 / np.pi) * np.unwrap(np.angle(Ls))

    # Finding Gain and Phase Margins
    x_gain_margin, y_gain_margin, crossover_freq, phase_margin = None, None, None, None
    x_phase_margin, y_phase_margin, unity_gain_freq, gain_margin = None, None, None, None

    for i in range(len(gain_values) - 1):
        if gain_values[i] > 0 and gain_values[i + 1] < 0:
            x_gain_margin = i
            y_gain_margin = gain_values[i]
            crossover_freq = round(omega_freq[i], 2)
            phase_margin = round(abs(phase_values[i] + 180), 2)
            break

    for i in range(len(phase_values) - 1):
        if phase_values[i] > -180 and phase_values[i + 1] < -180:
            x_phase_margin = i
            y_phase_margin = phase_values[i]
            unity_gain_freq = round(omega_freq[i], 2)
            gain_margin = round(abs(gain_values[i]), 2)
            break

    return x_gain_margin, y_gain_margin, crossover_freq, phase_margin, x_phase_margin, y_phase_margin, unity_gain_freq, gain_margin



# -----------------------------------
def plot_bode_diagrams(process_obj, controller_obj, omega_freq, x_gain_margin, y_gain_margin, crossover_freq, phase_margin, x_phase_margin, y_phase_margin, unity_gain_freq, gain_margin):
    """
    Plot Bode diagrams using the stability margin results.

    :process_obj: Process object.
    :controller_obj: Controller object.
    :omega_freq: Frequency vector (rad/s).
    :x_gain_margin: Index of gain margin.
    :y_gain_margin: Gain margin value.
    :crossover_freq: Crossover frequency.
    :phase_margin: Phase margin value.
    :x_phase_margin: Index of phase margin.
    :y_phase_margin: Phase margin value.
    :unity_gain_freq: Unity gain frequency.
    :gain_margin: Gain margin value.
    """
    # Generate the loop gain using the Bode method
    Ls = Bode(process_obj, omega_freq, Show=False)

    # Controller Transfer Function Components
    s = 1j * omega_freq
    integration_action = 1 / (controller_obj.parameters['Ti'] * s)
    Tfd = controller_obj.parameters['Td'] * controller_obj.parameters['alpha']
    derivative_action = controller_obj.parameters['Td'] * s / (1 + Tfd * s)
    Cs = controller_obj.parameters['Kc'] * (1 + integration_action + derivative_action)

    # Calculate overall loop gain
    Ls *= Cs

    # Generate Bode plots
    fig, (ax_gain, ax_phase) = plt.subplots(2, 1)
    fig.set_figheight(12)
    fig.set_figwidth(22)

    # Gain part
    ax_gain.semilogx(omega_freq, 20 * np.log10(np.abs(Ls)), label=r'$L(s)$')
    gain_min = np.min(20 * np.log10(np.abs(Ls) / 5))
    gain_max = np.max(20 * np.log10(np.abs(Ls) * 5))
    ax_gain.vlines(unity_gain_freq, gain_min, 0, color='r', linestyle='--', label='Unity Gain Frequency')
    ax_gain.vlines(crossover_freq, gain_min, gain_values[x_phase_margin], color='b', linestyle='--', label='Crossover Frequency')
    ax_gain.set_xlim([np.min(omega_freq), np.max(omega_freq)])
    ax_gain.set_ylim([gain_min, gain_max])
    ax_gain.set_ylabel('Amplitude $|L(s)|$ [dB]')
    ax_gain.set_title('Bode plot of $L(s)$ with Stability Margins')
    ax_gain.legend(loc='best')

    # Phase part
    ax_phase.semilogx(omega_freq, (180 / np.pi) * np.unwrap(np.angle(Ls)), label=r'$L(s)$')
    ax_phase.vlines(unity_gain_freq, -180, phase_values[x_gain_margin], color='r', linestyle='--', label='Unity Gain Frequency')
    ax_phase.vlines(crossover_freq, -180, phase_values[x_gain_margin], color='b', linestyle='--', label='Crossover Frequency')
    ax_phase.set_xlim([np.min(omega_freq), np.max(omega_freq)])
    ph_min = np.min((180 / np.pi) * np.unwrap(np.angle(Ls))) - 10
    ph_max = np.max((180 / np.pi) * np.unwrap(np.angle(Ls))) + 10
    ax_phase.set_ylim([np.max([ph_min, -200]), ph_max])
    ax_phase.set_xlabel(r'Frequency $\omega$ [rad/s]')
    ax_phase.set_ylabel('Phase $\angle L(s)$ [°]')
    ax_phase.legend(loc='best')

    plt.show()


# -----------------------------------

def IMC_TUNING(Kp, T1, T2, theta, gamma, method='FOPDT'):
    """
    The function IMC tuning returns the IMC controller parameters Kc, Ti and Td for a FOPDT or SOPDT process model.
    
    :Kp: process gain
    :T1: first (or main) lag time constant [s]
    :T2: second lag time constant [s]
    :theta: delay [s]
    :gamma: IMC tuning parameter
    :method: process model (optional: default value is 'FOPDT')
    """
    Tc = gamma * T1
    
    if method == 'FOPDT':
        Kc = (((T1 + (theta / 2)) / (Tc + (theta / 2)))) / Kp
        Ti = T1 + (theta / 2)
        Td = (T1 * theta) / ((2 * T1) + theta)
    elif method == 'SOPDT':
        Kc = ((T1 + T2) / (Tc + theta)) / Kp
        Ti = T1 + T2
        Td = T1 * T2 / (T1 + T2)
    else:
        # Par défaut, on utilisez FOPDT
        Kc = (((T1 + (theta / 2)) / (Tc + (theta / 2)))) / Kp
        Ti = T1 + (theta / 2)
        Td = (T1 * theta) / ((2 * T1) + theta)
    return Kc, Ti, Td



# -----------------------------------
def LL_RT(MV, Kp, TLead, TLag, Ts, MVFF, PVInit=0, method='EBD'):
    """
    The function "LL_RT" needs to be included in a "for or while loop".

    :MV: input vector
    :Kp: process gain
    :T1: lead time constant [s]
    :TLead: lead time constant [s]
    :TLag: lag time constant [s] 
    :Ts: sampling period [s]
    :PV: output vector
    :PVInit: (optional: default value is 0)
    :method: discretisation method (optional: default value is 'EBD')
        EBD: Euler Backward difference
        EFD: Euler Forward difference
        TRAP: Trapezoïdal method

    The function "LL_RT" appends a value to the output vector "PV".
    The appended value is obtained from a recurrent equation that depends on the discretisation method.
    """
    if (TLead!= 0 and TLag != 0):
        K = Ts/TLag
        if len(MVFF) == 0:
            MVFF.append(PVInit)
        else: # MV[k+1] is MV[-1] and MV[k] is MV[-2]
            if method == 'EBD':
                MVFF.append(((1/(1+K))*MVFF[-1]) + ((K*Kp/(1+K))*(((1+(TLead/Ts))*MV[-1])-((TLead/Ts)*MV[-2]))))  #slide 130
            elif method == 'EFD':
                MVFF.append((1-K)*MVFF[-1] + (K*Kp*((((TLead/Ts))*MV[-1])+((1-(TLead/Ts))*MV[-2]))))  #slide 131      
            else:
                MVFF.append((1/(1+K))*MVFF[-1] + (K*Kp/(1+K))*MV[-1])
    else:
        MVFF.append(Kp*MV[-1])

# -----------------------------------

# class Controller:
#     def __init__(self, parameters):
#         self.parameters = parameters

#     def set_parameters(self, parameters):
#         self.parameters = parameters

#     def get_parameters(self):
#         return self.parameters

#     def calculate_output(self, error, pre_error, integral):
#         Kp = self.parameters.get('Kp', 0)
#         Ki = self.parameters.get('Ki', 0)
#         Kd = self.parameters.get('Kd', 0)
#         alpha = self.parameters.get('alpha', 0)

#         # intégral et dérivé
#         proportional_term = Kp * error
#         integral_term = Ki * integral
#         derivative_term = Kd * (error - pre_error)

#         # Calcul de la sortie du contrôleur
#         output = proportional_term + integral_term + derivative_term

#         return output

class Controller:

    def __init__(self, parameters):

        self.parameters = parameters
        self.parameters['Kp'] = parameters['Kp'] if 'Kp' in parameters else 1.0
        self.parameters['alpha'] = parameters['alpha'] if 'alpha' in parameters else 0.0
        self.parameters['Td'] = parameters['Td'] if 'Td' in parameters else 0.0
        self.parameters['Ti'] = parameters['Ti'] if 'Ti' in parameters else 0.0

# -----------------------------------

def PID_RT(SP, PV, Man, MVMan, MVFF, Kc, Ti, Td, alpha, Ts, MVMin, MVMax, MV, MVP, MVI, MVD, E, ManFF=False, PVInit=0, method='EBD-EBD'):
    """
    Applies Proportional Integral Derivative (PID) control to regulate a process variable.

    :param SP: Setpoint vector
    :param PV: Process variable vector
    :param Man: Manual mode vector
    :param MVMan: Manual mode input vector
    :param MVFF: Feedforward input vector
    :param Kc: Proportional gain
    :param Ti: Integral time constant
    :param Td: Derivative time constant
    :param alpha: Derivative filter coefficient
    :param Ts: Sampling period
    :param MVMin: Minimum manipulated variable (MV) value
    :param MVMax: Maximum MV value
    :param MV: MV vector
    :param MVP: Proportional action vector
    :param MVI: Integral action vector
    :param MVD: Derivative action vector
    :param E: Error vector
    :param ManFF: Boolean indicating whether feedforward is activated in manual mode (default: False)
    :param PVInit: Initial value of the process variable (default: 0)
    :param method: Discretization method for integral and derivative actions (default: 'EBD-EBD')
        - First part: Discretization method for integral action {'EBD', 'EFD', 'TRAP'}
        - Second part: Discretization method for derivative action {'EBD', 'EFD', 'TRAP'}

    The function "PID_RT" computes the next manipulated variable (MV) value based on the given inputs.
    It implements a PID controller with configurable discretization methods for integral and derivative actions.

    """
    # Séparation de la chaîne de caractères method en EBD et EBD
    method_parts = method.split('-')
    # Accès aux deux parties séparées
    method_part1 = method_parts[0]
    method_part2 = method_parts[1]

    # Error
    if len(PV) == 0:
        E.append(SP[-1]-PVInit)
    else:
        E.append(SP[-1]-PV[-1])

    # Proportionnal action
    MVP.append(Kc*E[-1])

    # Integral action
    if len(MVI) == 0:
        MVI.append((Kc*Ts/Ti)*E[-1])
    else:
        if method_part1 == 'EBD':
            MVI.append(MVI[-1]+(Kc*Ts/Ti)*E[-1])

    # Derivative action
    Tfd = alpha*Td
    if len(MVD) == 0:
        MVD.append(0)
    else:
        if method_part2 == 'EBD':
            MVD.append((Tfd/(Tfd+Ts))*MVD[-1]+((Kc*Td)/(Tfd+Ts))*(E[-1]-E[-2]))

    # Feedforward Activation
    if ManFF:
        MVFFI = MVFF[-1]
    else:
        MVFFI = 0

    # MVMan.append(0)

    # Manual Mode
    if Man[-1] == True:
        if ManFF == False:
            MVI[-1] = MVMan[-1]-MVP[-1]-MVD[-1]
        else:
            MVI[-1] = MVMan[-1]-MVP[-1]-MVD[-1]-MVFFI

    # Saturation of MV
    MV_SUM = MVP[-1]+MVI[-1]+MVD[-1]+MVFFI
    if MV_SUM > MVMax:
        MVI[-1] = MVMax-MVP[-1]-MVD[-1]-MVFFI
        MV_SUM = MVMax
    if MV_SUM < MVMin:
        MVI[-1] = MVMin-MVP[-1]-MVD[-1]-MVFFI
        MV_SUM = MVMin

    MV.append(MV_SUM)


# -----------------------------------



class Process:

    def __init__(self, parameters):

        self.parameters = parameters
        self.parameters['Kp'] = parameters['Kp'] if 'Kp' in parameters else 1.0
        self.parameters['theta'] = parameters['theta'] if 'theta' in parameters else 0.0
        self.parameters['Tlead1'] = parameters['Tlead1'] if 'Tlead1' in parameters else 0.0
        self.parameters['Tlead2'] = parameters['Tlead2'] if 'Tlead2' in parameters else 0.0
        self.parameters['Tlag1'] = parameters['Tlag1'] if 'Tlag1' in parameters else 0.0
        self.parameters['Tlag2'] = parameters['Tlag2'] if 'Tlag2' in parameters else 0.0
        self.parameters['nInt'] = parameters['nInt'] if 'nInt' in parameters else 0

# -----------------------------------


def Bode(P, omega, Show=True):
    """
    :P: Process as defined by the class "Process".
        Use the following command to define the default process which is simply a unit gain process:
            P = Process({})

        A delay, two lead time constants and 2 lag constants can be added.

        Use the following commands for a SOPDT process:
            P.parameters['Kp'] = 1.1
            P.parameters['Tlag1'] = 10.0
            P.parameters['Tlag2'] = 2.0
            P.parameters['theta'] = 2.0

        Use the following commands for a unit gain Lead-lag process:
            P.parameters['Tlag1'] = 10.0        
            P.parameters['Tlead1'] = 15.0        

    :omega: frequency vector (rad/s); generated by a command of the type "omega = np.logspace(-2, 2, 10000)".
    :Show: boolean value (optional: default value = True).
        If Show == True, the Bode diagram is show.
        If Show == False, the Bode diagram is NOT show and the complex vector Ps is returned.

    The function "Bode" generates the Bode diagram of the process P
    """

    s = 1j*omega

    Ptheta = np.exp(-P.parameters['theta']*s)
    PGain = P.parameters['Kp']*np.ones_like(Ptheta)
    PLag1 = 1/(P.parameters['Tlag1']*s + 1)
    PLag2 = 1/(P.parameters['Tlag2']*s + 1)
    PLead1 = P.parameters['Tlead1']*s + 1
    PLead2 = P.parameters['Tlead2']*s + 1

    Ps = np.multiply(Ptheta, PGain)
    Ps = np.multiply(Ps, PLag1)
    Ps = np.multiply(Ps, PLag2)
    Ps = np.multiply(Ps, PLead1)
    Ps = np.multiply(Ps, PLead2)

    if Show == True:

        fig, (ax_gain, ax_phase) = plt.subplots(2, 1)
        fig.set_figheight(12)
        fig.set_figwidth(22)

        # Gain part
        ax_gain.semilogx(omega, 20*np.log10(np.abs(Ps)), label=r'$P(s)$')
        gain_min = np.min(20*np.log10(np.abs(Ps)/5))
        gain_max = np.max(20*np.log10(np.abs(Ps)*5))
        ax_gain.set_xlim([np.min(omega), np.max(omega)])
        ax_gain.set_ylim([gain_min, gain_max])
        ax_gain.set_ylabel('Amplitude' + '\n $|P(j\omega)|$ [dB]')
        ax_gain.set_title('Bode plot of P')
        ax_gain.legend(loc='best')

        # Phase part
        ax_phase.semilogx(omega, (180/np.pi) *
                          np.unwrap(np.angle(Ps)), label=r'$P(s)$')
        ax_phase.set_xlim([np.min(omega), np.max(omega)])
        ph_min = np.min((180/np.pi)*np.unwrap(np.angle(Ps))) - 10
        ph_max = np.max((180/np.pi)*np.unwrap(np.angle(Ps))) + 10
        ax_phase.set_ylim([np.max([ph_min, -200]), ph_max])
        ax_phase.set_xlabel(r'Frequency $\omega$ [rad/s]')
        ax_phase.set_ylabel('Phase' + '\n $\,$' + r'$\angle P(j\omega)$ [°]')
        ax_phase.legend(loc='best')

    else:
        return Ps
