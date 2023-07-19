import numpyro
nChains = 1
numpyro.set_host_device_count(nChains)
from numpyro.infer import NUTS,MCMC
from jax.config import config
config.update("jax_enable_x64", True)
from jax import random
import arviz as az
from likelihoods import birefringence_variable_evolution,generateMonteCarloEnergies
from load_data import get_all_data
from geometry import *
from astropy.cosmology import Planck18
import astropy.units as u
import sys

# Get dictionaries holding injections and posterior samples
spectra = get_all_data()

# Create ORFs
# First instantiate baseline objects
HL = Baseline(Detector.H1(),Detector.L1())
HV = Baseline(Detector.H1(),Detector.V1())
LV = Baseline(Detector.L1(),Detector.V1())

# Compute ORF values at each required frequency
HL_O1_gammaI,HL_O1_gammaV = HL.stokes_overlap_reduction_functions(spectra['H1L1_O1'][0])
HL_O2_gammaI,HL_O2_gammaV = HL.stokes_overlap_reduction_functions(spectra['H1L1_O2'][0])
HL_O3_gammaI,HL_O3_gammaV = HL.stokes_overlap_reduction_functions(spectra['H1L1_O3'][0])
HV_O3_gammaI,HV_O3_gammaV = HV.stokes_overlap_reduction_functions(spectra['H1V1_O3'][0])
LV_O3_gammaI,LV_O3_gammaV = LV.stokes_overlap_reduction_functions(spectra['L1V1_O3'][0])

# Store
spectra['H1L1_O1'].append(HL_O1_gammaI)
spectra['H1L1_O1'].append(HL_O1_gammaV)
spectra['H1L1_O2'].append(HL_O2_gammaI)
spectra['H1L1_O2'].append(HL_O2_gammaV)
spectra['H1L1_O3'].append(HL_O3_gammaI)
spectra['H1L1_O3'].append(HL_O3_gammaV)
spectra['H1V1_O3'].append(HV_O3_gammaI)
spectra['H1V1_O3'].append(HV_O3_gammaV)
spectra['L1V1_O3'].append(LV_O3_gammaI)
spectra['L1V1_O3'].append(LV_O3_gammaV)

# Get Monte Carlo weights to compute stochastic spectra
frequencies_to_sample = np.logspace(np.log10(spectra['H1L1_O1'][0][0]),np.log10(spectra['H1L1_O1'][0][-1]),300)
omg_weights,z_samples,dRdV_samples = generateMonteCarloEnergies(10000,frequencies_to_sample)
weight_dictionary = {\
    'zs':z_samples,
    'Dcs':Planck18.comoving_distance(z_samples).to(u.Gpc).value,
    'dRdVs':dRdV_samples,
    'omg_weights':omg_weights,
    'freqs':frequencies_to_sample}

# Premptively take outer products of frequencies with redshifts and distances
weight_dictionary['Dcs_outer_freqs'] = weight_dictionary['Dcs'][:,np.newaxis]*weight_dictionary['freqs'][np.newaxis,:]
weight_dictionary['zs_outer_freqs'] = weight_dictionary['zs'][:,np.newaxis]*weight_dictionary['freqs'][np.newaxis,:]

# Set up NUTS sampler over our likelihood
kernel = NUTS(birefringence_variable_evolution,dense_mass=True,target_accept_prob=0.9)
mcmc = MCMC(kernel,num_warmup=500,num_samples=1000,num_chains=nChains)

# Choose a random key and run over our model
rng_key = random.PRNGKey(114)
rng_key,rng_key_ = random.split(rng_key)

mcmc.run(rng_key_,spectra,weight_dictionary)
mcmc.print_summary()

# Save out data
data = az.from_numpyro(mcmc)
az.to_netcdf(data,"./../data/birefringence_variable_evolution_varyingAlphaR0.cdf")

