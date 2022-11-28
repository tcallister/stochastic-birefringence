import numpy as np

# Constants
c = 2.998e8
G = 6.67e-11
Mpc = 3.086e16*1e6
Gpc = Mpc*1e3
km = 1.e3
H0 = 67.9*km/Mpc # Units of 1/s
Gyr = 1.e9*365.*24*3600
year = 365.*24*3600
OmgM = 0.3065
OmgL = 0.6935
Msun = 1.99e30
MsunToSec = Msun*G/np.power(c,3.)
rhoC = 3.*np.power(H0*c,2.)/(8.*np.pi*G)*np.power(Mpc,3.) # Converted to J/Mpc^3
