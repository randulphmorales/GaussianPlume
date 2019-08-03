#!/usr/bin/env python
# python

""" Script to convert concentration in ug/m^3 to ppm

This script is used to convert a concentration in ug/m^3 to ppm and vice-versa. Default conversion setting is under normal temperature and pressure (i.e., ambient temperature = 20C and pressure = 1013.25 hPa )
"""

# REQUIRED BUILD-IN MODULES
import numpy as np

###########################
# Module to convert units #
###########################
# A. Berchet - March 2016


############################################################################
# CONVERT ppm to ug/m3 from temperature and pressure
############################################################################
def ppm2micro(conc, mass, temp=20., pres=1013.25):
    """
    conc in ppm
    temp in degC
    pres in hPa
    mass in g/mol
    """
    R = 8.31446
    conc_out = np.array(np.array(pres) / R /
                        (np.array(temp) + 273.15) * 1e2 * mass * np.array(conc))
    conc_out[np.array(conc) <= 0] = np.nan
    return conc_out


def micro2ppm(conc, mass, temp=20., pres=1013.25):
    """
    conc in ug/m3
    temp in degC
    pres in hPa
    mass in g/mol
    """
    R = 8.31446
    conc_out = np.array(
        (np.array(conc) * R * (np.array(temp) + 273.15) * 1e-2) / (mass * np.array(pres)))
    # conc_out[np.array(conc) <= 0] = np.nan
    return conc_out
############################################################################


############################################################################
# CONVERT ug/m3 to moles
############################################################################
def ug2mol(m):
    """
    molecular weight
    """
    return 6.022e23 / 1e12 / m


