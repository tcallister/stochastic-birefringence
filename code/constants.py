import numpy as np
from astropy.cosmology import Planck18
import astropy.constants as const
import astropy.units as u

# Misc physical constants in SI units
c = const.c.value
G = const.G.value
Mpc = const.pc.value*1e6
Gpc = Mpc*1e3
km = 1.e3
year = 365.25*24*3600
Gyr = 1e9*year
Msun = const.M_sun.value
MsunToSec = Msun*G/np.power(c,3.)
R_earth = 6.371e6

# Cosmological parameters, again in SI units
H0 = Planck18.H0.to(u.s**(-1)).value
OmgM = Planck18.Om0
OmgL = Planck18.Ode0

rhoC = 3.*np.power(H0*c,2.)/(8.*np.pi*G)*np.power(Mpc,3.) # Converted to J/Mpc^3
