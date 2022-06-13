"""GigaAnalysis - Superconductors - :mod:`gigaanalysis.htsc`
--------------------------------------------------------------

Here are a few functions for equations that are useful for high temperature 
superconducting science. These are useful for getting doping values from 
transition temperatures and vice-versa. The default values for these are 
given for YBCO using the values from DOI: 10.1103/PhysRevB.73.180505 Also 
extracting the transition temperature from stepped data.
"""

from .data import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def trans_res(data, res_onset, under_nan=False, over_nan=False, 
        as_ratio=False):
    """Returns the dependent variable value at the resistive transition.

    This assumes the data is sorted and returns the last value that is below 
    the onset resistance specified.
    
    Parameters
    ----------
    data : Data
        The sorted resistivity data to look for the transition in.
    res_onset : float 
        The value of resistivity that if measured is then considered that 
        the sample is now not superconducting. This is in the same units as 
        given in the data.
    under_nan : bool, optional
        If the default of 'False' zero is returned if the all the data lays 
        above the onset value. If 'True' NaN is returned.
    over_nan : bool, optional
        If the default of 'False' the last value is returned if all the data 
        lays below the onset value. If 'True' NaN is returned.
    as_ratio : bool, optional
        If `True` then the value of res_onset is multiplied by the maximum 
        value of the data. For most examples this means that if 
        ``res_onset = 0.01`` then the transition would be at 1% of the 
        high temperature value. The default is `False`.

    Returns
    -------
    transition_onset : float
        The last value of the dependent variable where the independent 
        variable is below the given onset value.
    """
    if not isinstance(data, Data):
        raise TypeError(
            f"data need to be a Data object but instead was {type(data)}.")

    if as_ratio:
        res_onset = data.y.max()*res_onset

    r_data = data[::-1].append([[0., -np.inf]])
    low_x = r_data.x[r_data.y <= res_onset]
    if under_nan and low_x.size == 1:
        return np.nan
    elif over_nan and low_x.size == len(r_data):
        return np.nan
    else:
        return low_x[0]


def _dome_output(orignal, out_x, out_y, as_Data):
    """Formats the output of the dome set of functions. Takes the input and 
    the outputs and returns the result in the correct format.
    """
    if as_Data:
        if out_y.size == 1:
            return Data(np.asarray([[out_x, out_y]]))
        else:
            return Data(out_x, out_y)
    elif out_y.size == 1 and not isinstance(orignal, np.ndarray):
        return float(out_y)
    else:
        return out_y


def dome_p2tc(doping, t_max=94.3, p_max=16., p_w=11., as_Data=False):
    """This converts values of doping to transition temperature on a SC dome.

    The default parameters from the dome are taken from YBCO, but are 
    changeable.
    
    Parameters
    -----------
    doping : float or numpy.ndarray
        The value or values of the doping to calculate the critical 
        temperature for. The units are percent of holes per unit cell per 
        plane.
    t_max : float : optional
        The maximum critical temperature of the dome in Kelvin. The default 
        is 94.3 K.
    p_max : float, optional
        The doping at the maximum critical temperature in percent. The 
        default is 16 %.
    p_w : float, optional
        The half width of the dome in percent doping. The default value is
        11 %.
    as_Data : bool, optional
        If 'True' a :class:`.Data` object is returned with the dopings as 
        the dependent variable and the critical temperatures as the 
        independent variable. The default is 'False' which returns a 
        :class:`numpy.ndarray`.

    Returns
    -------
    critical_temperature : float, numpy.ndarray, or Data
        The values of the critical temperature of the superconductor at the 
        given doping values in Kelvin.
    """
    p = np.asarray(doping)
    tc = np.array(t_max*(1. - (p - p_max)*(p - p_max)/p_w/p_w))
    tc[tc<0] = 0
    return _dome_output(doping, p, tc, as_Data)


def dome_tc2p(critical_temperature, side, 
        t_max=94.3, p_max=16., p_w=11., as_Data=False):
    """This converts values of the critical temperature to doping.

    The default parameters from the dome are taken from YBCO, but are 
    changeable.
    
    Parameters
    -----------
    critical_temperature : float or numpy.ndarray
        The value or values of the critical temperature to calculate the 
        doping for. The units are percent of holes per unit cell per 
        plane.
    side : str or numpy.ndarray of {'UD' or 'OD'}
        The side of the dome to calculate the doping of. 'UD' for the under 
        doped size, and 'OD' for the over doped side. This is either a 
        string or an array the length of the given critical temperatures.
    t_max : float : optional
        The maximum critical temperature of the dome in Kelvin. The default 
        is 94.3 K.
    p_max : float, optional
        The doping at the maximum critical temperature in percent. The 
        default is 16 %.
    p_w : float, optional
        The half width of the dome in percent doping. The default value is
        11 %.
    as_Data : bool, optional
        If 'True' a :class:`.Data` object is returned with the critical 
        temperatures as the dependent variable and the dopings as the 
        independent variable. The default is 'False' which returns a 
        :class:`numpy.ndarray`.

    Returns
    -------
    doping : float, numpy.ndarray, or Data
        The values of the doping of the superconductor with the given 
        critical temperature and side of the dome.
    """
    side = np.array(side)
    if np.any(~((side == 'UD') | (side == 'OD'))):
        raise ValueError(
            f"side needs to be 'UD' or 'OD'.")
    
    sign = np.ones(side.shape)
    sign[side == 'UD'] *= -1
    
    tc = np.array(critical_temperature)
    if np.any(tc < 0) or np.any(tc > t_max):
        raise ValueError(
            f"Critical temperature is outside the possible "
            f"range 0 K to {t_max:.4} K.")

    p = p_max + sign*p_w*np.sqrt((1. - tc/t_max))
    
    return _dome_output(critical_temperature, tc, p, as_Data)


def cdw_factor(doping, a_cdw=-0.204, p_cdw=11.874, w_cdw=3.746, 
        as_Data=False):
    """Calculates the faction the critical temperature is reduced by CDW.

    This is for calculating the ratio of critical temperature suppression 
    from the Charge Density Wave. The values are for YBCO.
    
    Parameters
    -----------
    doping : float or numpy.ndarray
        The value or values of the doping to calculate the critical 
        temperature suppression for. The units are percent of holes per unit 
        cell per plane.
    a_cdw : float : optional
        The maximum critical temperature suppression of the dome as a ratio. 
        The value is absolute so negative values are a suppression and the 
        default is -0.204.
    p_cdw : float, optional
        The doping at the maximum amount of CDW in percent. The default 
        value is 11.874 %.
    w_cdw : float, optional
        The full width half maximum of the CDW dome in percent doping. The 
        default value is 3.764 %.
    as_Data : bool, optional
        If 'True' a :class:`.Data` object is returned with the dopings as 
        the dependent variable and the critical temperatures suppression as 
        the independent variable. The default is 'False' which returns a 
        :class:`numpy.ndarray`.

    Returns
    -------
    critical_temperature : float, numpy.ndarray, or Data
        The values of the critical temperature suppression of the 
        superconductor due to the CDW at the given doping values as a ratio.
    """
    p = np.asarray(doping)
    cdw = 1. + a_cdw*np.exp(-4*np.log(2)*np.power((p - p_cdw)/w_cdw, 2))
    return _dome_output(doping, p, cdw, as_Data)


def ybco_p2tc(doping, t_max=94.3, p_max=16., p_w=11.,
        a_cdw=-0.204, p_cdw=11.874, w_cdw=3.746, as_Data=False):
    """This converts values of doping to transition temperature of YBCO.

    This calculates the transition temperature from the doping while taking 
    into consideration the effect of CDW. The default parameters from the 
    dome are taken from YBCO, but are changeable.
    
    Parameters
    -----------
    doping : float or numpy.ndarray
        The value or values of the doping to calculate the critical 
        temperature for. The units are percent of holes per unit cell per 
        plane.
    t_max : float : optional
        The maximum critical temperature of the dome in Kelvin. The default 
        is 94.3 K.
    p_max : float, optional
        The doping at the maximum critical temperature in percent. The 
        default is 16 %.
    p_w : float, optional
        The half width of the dome in percent doping. The default value is
        11 %.
    a_cdw : float : optional
        The maximum critical temperature suppression of the dome as a ratio. 
        The value is absolute so negative values are a suppression and the 
        default is -0.204.
    p_cdw : float, optional
        The doping at the maximum amount of CDW in percent. The default 
        value is 11.874 %.
    w_cdw : float, optional
        The full width half maximum of the CDW dome in percent doping. The 
        default value is 3.764 %.
    as_Data : bool, optional
        If 'True' a :class:`.Data` object is returned with the dopings as 
        the dependent variable and the critical temperatures as the 
        independent variable. The default is 'False' which returns a 
        :class:`numpy.ndarray`.

    Returns
    -------
    critical_temperature : float, numpy.ndarray, or Data
        The values of the critical temperature of the superconductor at the 
        given doping values in Kelvin.
    """
    p = np.asarray(doping)
    tc = dome_p2tc(p, t_max=t_max, p_max=p_max, p_w=p_w)
    tc *= cdw_factor(p, a_cdw=a_cdw, p_cdw=p_cdw, w_cdw=w_cdw)
    tc[tc<0] = 0
    return _dome_output(doping, p, tc, as_Data)


def ybco_tc2p(critical_temperature, side,
        t_max=94.3, p_max=16., p_w=11.,
        a_cdw=-0.204, p_cdw=11.874, w_cdw=3.746,
        gen_points=500, as_Data=False):
    """This converts values of the critical temperature to doping.

    This takes into consideration the Charge Density Wave (CDW) found in 
    YBCO. The default parameters from the dome are taken from YBCO, but are 
    changeable.
    
    Parameters
    -----------
    critical_temperature : float or numpy.ndarray
        The value or values of the critical temperature to calculate the 
        doping for. The units are percent of holes per unit cell per 
        plane.
    side : str or numpy.ndarray of {'UD' or 'OD'}
        The side of the dome to calculate the doping of. 'UD' for the under 
        doped size, and 'OD' for the over doped side. This is either a 
        string or an array the length of the given critical temperatures.
    t_max : float : optional
        The maximum critical temperature of the dome in Kelvin. The default 
        is 94.3 K.
    p_max : float, optional
        The doping at the maximum critical temperature in percent. The 
        default is 16 %.
    p_w : float, optional
        The half width of the dome in percent doping. The default value is
        11 %.
    a_cdw : float : optional
        The maximum critical temperature suppression of the dome as a ratio. 
        The value is absolute so negative values are a suppression and the 
        default is -0.204.
    p_cdw : float, optional
        The doping at the maximum amount of CDW in percent. The default 
        value is 11.874 %.
    w_cdw : float, optional
        The full width half maximum of the CDW dome in percent doping. The 
        default value is 3.764 %.
    gen_points : int, optional
        The doping values are calculated by interpolating along a curve. 
        This parameters specifies how many points to generate for the 
        interpolation. THe default is 500.
    as_Data : bool, optional
        If 'True' a :class:`.Data` object is returned with the critical 
        temperatures as the dependent variable and the dopings as the 
        independent variable. The default is 'False' which returns a 
        :class:`numpy.ndarray`.

    Returns
    -------
    doping : float, numpy.ndarray, or Data
        The values of the doping of the superconductor with the given 
        critical temperature and side of the dome.
    """
    side = np.array(side)
    if np.any(~((side == 'UD') | (side == 'OD'))):
        raise ValueError(
            f"side needs to be 'UD' or 'OD'.")
    
    gen_dome = ybco_p2tc(
        np.linspace(p_max - p_w*1.001, p_max + p_w*1.001,
            gen_points),  # 1.001 is to make sure get tc=0
        t_max=t_max, p_max=p_max, p_w=p_w,
        a_cdw=a_cdw, p_cdw=p_cdw, w_cdw=w_cdw, as_Data=True)
    
    max_arg = gen_dome.y.argmax()
    
    tc = np.array(critical_temperature)
    tc = tc.reshape(tc.size)
    if np.any(tc < 0) or np.any(tc > gen_dome.y.max()):
        raise ValueError(
            f"Critical temperature is outside the possible "
            f"range 0 K to {gen_dome.y.max():.4} K.")

    p_UD = swap_xy(gen_dome[:max_arg+1]).interp_values(tc).y
    p_OD = swap_xy(gen_dome[-1:max_arg-1:-1]).interp_values(tc).y
    
    if side.size == 1 and side == 'UD':
        p = p_UD
    elif side.size == 1 and side == 'OD':
        p = p_OD
    else:
        p = p_UD
        p[side=='OD'] = p_OD[side=='OD']
    
    return _dome_output(critical_temperature, tc, p, as_Data)

