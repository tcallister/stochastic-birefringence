from makeDelayedRateData import *
from gridded_likelihood import compute_likelihood_grids
import h5py
from scipy.special import gammainc

# Array of possible maximum redshifts
zMax_array = np.arange(10,16)

# Preemptively create hdf5 file to hold results 
hfile = h5py.File('./../data/fixed_rate_delayedSFR_variable_zMax.hdf','w')

# Loop across grid
for zMax in zMax_array:

    # Compute merger rate vs. redshift
    # First load precomputed grids of merger redshifts, evolutionary time delays, and corresponding
    # values of formation redshifts and star formation rates (following Madau+Dickinson)
    rateData = generateTimeDelayData(zMax=zMax)
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

    # Compute likelihood over grid of kappa_dc and kappa_z values
    results = compute_likelihood_grids(zs_merger,dRdV_delayed,joint=False)

    posterior = hfile.create_group('result_zMax_{0}'.format(int(zMax)))
    for key,val in results.items():
        posterior.create_dataset(key,data=val)

# Add some metadata
hfile.attrs['Created_by'] = "run_birefringence_delayedSFR_variable_zMax.py"
hfile.attrs['Downloadable_from'] = ""
hfile.attrs['Source_code'] = "https://github.com/tcallister/stochastic-birefringence"
hfile.close()

