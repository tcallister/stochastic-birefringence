from makeDelayedRateData import *
from gridded_likelihood import compute_likelihood_grids
import h5py
from scipy.special import gammainc

# Compute merger rate vs. redshift
# First load precomputed grids of merger redshifts, evolutionary time delays, and corresponding
# values of formation redshifts and star formation rates (following Madau+Dickinson)
rateData = generateTimeDelayData()
zs_merger = rateData['zs']
time_delays = rateData['tds']
zs_formation = rateData['formationRedshifts']
formationRates = rateData['formationRates']

# The following function is an approximation to the integrated fraction of star formation below
# Z=Z_sun/10 as a function of redshift
low_Z_fraction = gammainc(0.84,(0.1**2.)*np.power(10.,0.3*zs_formation))

# Multiply by total formation rate to get the rate of low metallicity star formation
weightedFormationRates = formationRates*low_Z_fraction

# Convolve formation rate with time-delay distribution
# Set t_min = 10 Myr
tdMin = 0.01
dpdt = np.power(time_delays,-1)
dpdt[time_delays<tdMin] = 0.
dRdV_delayed = weightedFormationRates.dot(dpdt)
dRdV_delayed /= dRdV_delayed[0]

# Define clipping function to skip likelihood calculations over gridpoints which we know
# in advance will return loglikelihood = -inf.
def toClip(kd,kz):
    if kz<-0.4-2.*kd or kz>0.4-2.*kd or kz<-0.2+3.*(kd-0.5) or kz>0.2+3.*(kd+0.5):
        return True
    else:
        return False

# Compute likelihood over grid of kappa_dc and kappa_z values
results = compute_likelihood_grids(zs_merger,dRdV_delayed,clippingFunction=toClip,kappaGridSize=400,baselines=['LVO3'])

# Create hdf5 file and write posterior samples
hfile = h5py.File('./../data/fixed_rate_delayedSFR_LVO3.hdf','w')
posterior = hfile.create_group('result')
for key,val in results.items():
    posterior.create_dataset(key,data=val)

# Add some metadata
hfile.attrs['Created_by'] = "run_birefringence_delayedSFR_LVO3.py"
hfile.attrs['Downloadable_from'] = "https://zenodo.org/doi/10.5281/zenodo.10384998"
hfile.attrs['Source_code'] = "https://github.com/tcallister/stochastic-birefringence"
hfile.close()

