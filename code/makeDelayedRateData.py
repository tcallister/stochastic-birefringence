import numpy as np
from scipy.interpolate import interp1d
import sys
import os
from constants import *
from astropy.cosmology import Planck18
import astropy.units as u
codeDir = os.path.dirname(os.path.realpath(__file__))

"""
Script to precompute formation and time delay data used in generating Omega(f) spectra.
This file loads in cosmological data from `redshiftData.dat` and computes the following:

   * `zs`: A 1D array of redshifts matching that in `redshiftData.dat`
   * `tds`: A 1D array of delay times (units of Gyr) between 0.01 Gyr -- 13.5 Gyr
   * `formationRedshifts`: A 2D array of formation redshifts corresponding to each combination of *merger* redshifts and time delays. The value `formationRedshifts[i,j]` corresponds to the formation redshift of a binary that mergers at zs[i] after 
    a time delay tds[j].
   * `formationRates`: A 2D array of the SFR (not metallicity weighted) at each of the redshifts in `formationRedshifts`

Data is saved to: `delatedRateData.npy`
"""

def timeDelay(zMerge,zForm):

    """
    Function to compute the proper time occuring between redshifts zForm and zMerge
    Returns delay time in Gyr
    """

    # We better be merging after we've formed...
    if zMerge>zForm:
        return 0

    else:

        # Take integral
        zs = np.linspace(zMerge,zForm,500)
        timeDelayIntegrands = (1./Gyr)*np.power((1.+zs)*H0*np.sqrt(OmgM*np.power(1.+zs,3.)+OmgL),-1.)
        return np.sum(timeDelayIntegrands)*(zs[1]-zs[0])


# MD rate density parameters
alpha = 2.7
beta = 5.6
zpeak = 1.9

# Set up grids of possible merger redshifts and evolutionary time delays (Gyr)
zs = np.arange(0,10,0.01)
tds = np.arange(0.005,13.5,0.005)

# Also set up a 2D grid that will hold the **formation redshifts** zf(zm,td)
# corresponding to a merger at z=zm after a time delay td
formationRedshifts = np.zeros((zs.size,tds.size))

# Prepare to save the SFR at these formation redshifts
formationRates = np.zeros((zs.size,tds.size))

# Loop across merger redshifts
for i,zm in enumerate(zs):

    print(zm)

    # At each merger redshift, build an interpolator to map from time delays back to a formation redshift
    # First define a reference array of formation redshifts
    # Our resolution is given by solving for the step size dz that will give us time delays < 10 Myr
    dz = (0.003*Gyr)*(1.+0.)*H0*np.sqrt(OmgM+OmgL)
    zf_data = np.arange(zm,15.,dz)               

    # Now compute time delays between the given merger redshift zm and each of the zf's
    timeDelayIntegrands = (1./Gyr)*np.power((1.+zf_data)*H0*np.sqrt(OmgM*np.power(1.+zf_data,3.)+OmgL),-1.)
    td_data = np.insert(np.cumsum(timeDelayIntegrands)*dz,0,0)[:-1]
    print(dz,td_data[:10])

    # Evaluate formation redshifts corresponding to each time delay in "tds"
    zfs = np.interp(tds,td_data,zf_data)
    formationRedshifts[i,:] = zfs

    # Get the star formation rate at this *formation* redshift
    formationRates[i,:] = np.power(1.+zfs,alpha)/(1.+np.power((1.+zfs)/(1.+zpeak),beta))
    formationRates[i,zfs!=zfs] = 0
    formationRates[i,zfs>10.] = 0

formationRates[formationRates!=formationRates] = 0.
formationRedshifts[formationRedshifts!=formationRedshifts] = 0.

delayedRateDict = {}
delayedRateDict['formationRates'] = formationRates
delayedRateDict['formationRedshifts'] = formationRedshifts
delayedRateDict['zs'] = zs
delayedRateDict['tds'] = tds

np.save('delayedRateData.npy',delayedRateDict)

