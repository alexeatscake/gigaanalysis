"""GigaAnalysis - Constants - :mod:`gigaanalysis.const`
----------------------------------------------------------

Here is contained a collection of functions with when called return values 
of physical constants. They always return floats and all have one optional 
parameter 'unit' which default is 'SI' for the International System of Units 
values for these parameters.
The module :mod:`scipy.constants` contains many more than what is listed 
here, but I included these for the different units.
"""

from .data import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def __pick_unit(unit, units_dict):
    """Takes value from dictionary and returns value

    Parameters
    ----------
    unit: str
        The unit chosen
    units_dcit: dict
        The dictionary with the units and the values

    Returns
    -------
    const_value: float
        The value that is requested
    """
    if unit in units_dict.keys():
        return units_dict[unit]
    else:
        if len(units_dict) == 1:
            raise ValueError("unit must be '{}'".format(list(units_dict)[0]))
        else:
            unit_list = ["'{}',".format(x) for x in units_dict.keys()]
            unit_list[-1] = "or {}.".format(unit_list[-1][:-1])
            raise ValueError("unit must be {}".format(" ".join(unit_list)))


def amu(unit='SI'):
    """`Unified Atomic mass unit 
    <https://en.wikipedia.org/wiki/Dalton_(unit)>`_ or Dalton

    =====  =====
    Unit   Value
    =====  =====
    'SI'   1.66053906660e-27 kg
    'CGS'  1.66053906660e-24 
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Atomic mass unit : float
    """
    return __pick_unit(unit, {
        'SI': 1.66053906660e-27,
        'CGS': 1.66053906660e-24,
        })


def Na(unit='SI'):
    """`Avogadro constant
    <https://en.wikipedia.org/wiki/Avogadro_constant>`_

    =====  =====
    Unit   Value
    =====  =====
    'SI'   6.02214076e+23 1/mol
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Avogadro constant : float
    """
    return __pick_unit(unit, {
        'SI': 6.02214076e+23,
        })


def kb(unit='SI'):
    """`Boltzmann constant 
    <http://en.wikipedia.org/wiki/Boltzmann_constant>`_

    =====  =====
    Unit   Value
    =====  =====
    'SI'   1.380649e-23 J/K
    'eV'   8.617333262145e-5 eV/K
    'CGS'  1.380649e-16 erg/K/
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Boltzmann constant : float
    """
    return __pick_unit(unit, {
        'SI': 1.380649e-23,
        'eV': 8.617333262145e-5,
        'CGS': 1.380649e-16,
        })


def muB(unit='SI'):
    """`Bohr magneton 
    <https://en.wikipedia.org/wiki/Bohr_magneton>`_

    =====  =====
    Unit   Value
    =====  =====
    'SI'   9.274009994e-24 J/T
    'eV'   5.7883818012e-5 eV/T
    'CGS'  9.274009994e-21 erg/T
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Bohr magneton : float
    """
    return __pick_unit(unit, {
        'SI': 9.274009994e-24,
        'eV': 8.617333262145e-5,
        'CGS': 1.380649e-16,
        })


def a0(unit='SI'):
    """`Bohr radius 
    <https://en.wikipedia.org/wiki/Bohr_radius>`_

    ======  ======
    Unit    Value
    ======  ======
    'SI'    5.29177210903e-11 m
    'CGS'   5.29177210903e-9 cm
    ======  ======

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Bohr radius : float
    """
    return __pick_unit(unit, {
        'SI':  5.29177210903e-11,
        'CGS': 5.29177210903e-9,
        })


def me(unit='SI'):
    """`Electron rest mass 
    <https://en.wikipedia.org/wiki/Electron_rest_mass>`_

    ======  =====
    Unit    Value
    ======  =====
    'SI'    9.1093837015e-31 kg
    'CGS'   9.1093837015e-29 g
    'MeVc'  5.1099895000e-1 MeV/c^2
    'uamu'  5.48579909065e-4 Da
    ======  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Bohr magneton : float
    """
    return __pick_unit(unit, {
        'SI':  9.1093837015e-31,
        'CGS': 9.1093837015e-29,
        'MeVc':5.1099895000e-1,
        'uamu':5.48579909065e-4,
        })


def qe(unit='SI'):
    """`Elementary charge
    <https://en.wikipedia.org/wiki/Elementary_charge>`_
    
    =====  =====
    Unit   Value
    =====  =====
    'SI'   1.602176634e-19 C
    'CGS'  1.602176634e-20 statC
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Elementary charge : float
    """
    return __pick_unit(unit, {
        'SI':  1.602176634e-19,
        'CGS': 1.602176634e-20,
        })


def alpha(unit='SI'):
    """`Fine-structure constant 
    <https://en.wikipedia.org/wiki/Fine-structure_constant>`_

    =====  =====
    Unit   Value
    =====  =====
    'SI'   7.2973525693e-3
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Fine-structure constant : float
    """
    return __pick_unit(unit, {
        'SI': 7.2973525693e-3,
        })


def R(unit='SI'):
    """`Gas Constant
    <https://en.wikipedia.org/wiki/Gas_constant>`_
    
    =====  =====
    Unit   Value
    =====  =====
    'SI'   8.31446261815324 J/K/mol
    'eV'   5.189479388046824e+19 eV/K/mol
    'CGS'  8.31446261815324e+7 erg/K/mol
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Gas Constant : float
    """
    return __pick_unit(unit, {
        'SI': 8.31446261815324,
        'eV':5.189479388046824e+19,
        'CGS': 8.31446261815324e+7
        })


def G(unit='SI'):
    """`Gravitational constant
    <https://en.wikipedia.org/wiki/Gravitational_constant>`_
    
    =====  =====
    Unit   Value
    =====  =====
    'SI'   6.67430e-11 m^3/kg/s^2
    'CGS'  6.67430e-8 dyn cm^2/g^2
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Gravitational constant : float
    """
    return __pick_unit(unit, {
        'SI':  6.67430e-11,
        'CGS': 6.67430e-8,
        })


def muN(unit='SI'):
    """`Nuclear magneton 
    <https://en.wikipedia.org/wiki/Nuclear_magneton>`_

    =====  =====
    Unit   Value
    =====  =====
    'SI'   5.050783699e-27 J/T
    'eV'   3.1524512550e-8 eV/T
    'CGS'  5.050783699e-24 erg/T
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Nuclear magneton : float
    """
    return __pick_unit(unit, {
        'SI': 5.050783699e-27,
        'eV': 3.1524512550e-8,
        'CGS': 5.050783699e-24,
        })


def mp(unit='SI'):
    """`Proton rest mass 
    <https://en.wikipedia.org/wiki/Proton>`_

    ======  =====
    Unit    Value
    ======  =====
    'SI'    1.67262192369e-27 kg
    'CGS'   1.67262192369e-25 g
    'MeVc'  9.3827208816e+2 MeV/c^2
    'uamu'  1.007276466621e+0 Da
    ======  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Nuclear magneton : float
    """
    return __pick_unit(unit, {
        'SI':  1.67262192369e-27,
        'CGS':  1.67262192369e-25,
        'MeVc': 9.3827208816e+2,
        'uamu': 1.007276466621e+0,
        })



def h(unit='SI'):
    """`Planck constant
    <https://en.wikipedia.org/wiki/Planck_constant>`_
    
    =====  =====
    Unit   Value
    =====  =====
    'SI'   6.62607015e-34 J s
    'eV'   4.135667696e-15 eV s
    'CGS'  6.62607015e-27 erg s
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Planck constant : float
    """
    return __pick_unit(unit, {
        'SI':  6.62607015e-34,
        'eV':  4.135667696e-15,
        'CGS': 6.62607015e-27,
        })


def hbar(unit='SI'):
    """`Reduced Planck constant
    <https://en.wikipedia.org/wiki/Planck_constant>`_
    
    =====  =====
    Unit   Value
    =====  =====
    'SI'   1.054571817e-34 J s
    'eV'   6.582119569e-16 eV s
    'CGS'  1.054571817e-27 erg s
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Reduced Planck constant : float
    """
    return __pick_unit(unit, {
        'SI':  1.054571817e-34,
        'eV':  6.582119569e-16,
        'CGS': 1.054571817e-27,
        })


def c(unit='SI'):
    """`Speed of light 
    <https://en.wikipedia.org/wiki/Speed_of_light>`_
    
    =====  =====
    Unit   Value
    =====  =====
    'SI'   2.99792458e+8 m/s
    'CGS'  2.99792458e+10 cm/s
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the speed of light : float
    """
    return __pick_unit(unit, {
        'SI': 2.99792458e8,
        'CGS': 2.99792458e10,
        })


def mu0(unit='SI'):
    """`Vacuum permeability
    <https://en.wikipedia.org/wiki/Vacuum_permeability>`_
    
    =====  =====
    Unit   Value
    =====  =====
    'SI'   1.25663706212e-6 H/m
    'eV'   7.8433116265e+12 eV/A^2 
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Vacuum permeability : float
    """
    return __pick_unit(unit, {
        'SI':  1.25663706212e-6,
        'eV':  7.8433116265e+12,
        })


def ep0(unit='SI'):
    """`Vacuum permittivity
    <https://en.wikipedia.org/wiki/Vacuum_permittivity>`_
    
    =====  =====
    Unit   Value
    =====  =====
    'SI'   8.8541878128e-12 F/m
    'eV'   1.4185972826e-30 C^2/eV
    =====  =====

    Parameters
    ----------
    unit : str, optional
        The unit system to give the value in.

    Returns
    -------
    Value of the Vacuum permittivity : float
    """
    return __pick_unit(unit, {
        'SI':  8.8541878128e-12,
        'eV':  1.4185972826e-30,
        })

