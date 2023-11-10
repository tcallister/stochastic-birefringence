import numpyro
nChains = 1
numpyro.set_host_device_count(nChains)
from numpyro.infer import NUTS,MCMC
from jax.config import config
config.update("jax_enable_x64", True)
from jax import random
import arviz as az
from numpyro_likelihoods import birefringence_variable_evolution,generateMonteCarloEnergies
from load_data import get_all_data
from geometry import *
from astropy.cosmology import Planck18
import astropy.units as u
import sys

# Matlab data
matlab_orf_freqs,matlab_orf_H1L1,matlab_orf_H1V1,matlab_orf_L1V1 = np.loadtxt('../input/matlab_orfs.dat',unpack=True)
f_H1L1_O3 = matlab_orf_freqs[matlab_orf_freqs<=1726]
matlab_orf_H1L1 = matlab_orf_H1L1[matlab_orf_freqs<=1726]
matlab_orf_H1V1 = matlab_orf_H1V1[matlab_orf_freqs<=1726]
matlab_orf_L1V1 = matlab_orf_L1V1[matlab_orf_freqs<=1726]

# Get dictionaries holding injections and posterior samples
# Also overwrite frequencies
spectra = get_all_data(trim_nans=False)
spectra['H1L1_O1'][0] = f_H1L1_O3
spectra['H1L1_O2'][0] = f_H1L1_O3
spectra['H1L1_O3'][0] = f_H1L1_O3
spectra['H1V1_O3'][0] = f_H1L1_O3
spectra['L1V1_O3'][0] = f_H1L1_O3

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
spectra['H1L1_O1'].append(matlab_orf_H1L1)
spectra['H1L1_O1'].append(HL_O1_gammaV)
spectra['H1L1_O2'].append(matlab_orf_H1L1)
spectra['H1L1_O2'].append(HL_O2_gammaV)
spectra['H1L1_O3'].append(matlab_orf_H1L1)
spectra['H1L1_O3'].append(HL_O3_gammaV)
spectra['H1V1_O3'].append(matlab_orf_H1V1)
spectra['H1V1_O3'].append(HV_O3_gammaV)
spectra['L1V1_O3'].append(matlab_orf_L1V1)
spectra['L1V1_O3'].append(LV_O3_gammaV)

# Get Monte Carlo weights to compute stochastic spectra
alpha_ref = 5
beta_ref = -5
zpeak_ref = 10
frequencies_to_sample = np.logspace(np.log10(spectra['H1L1_O3'][0][0]),np.log10(spectra['H1L1_O3'][0][-1]),300)
omg_weights,z_samples,dRdV_samples = generateMonteCarloEnergies(30000,frequencies_to_sample,alpha_ref,beta_ref,zpeak_ref)
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
kernel = NUTS(birefringence_variable_evolution,target_accept_prob=0.95,dense_mass=[("kappa_Dc","kappa_z")])
mcmc = MCMC(kernel,num_warmup=500,num_samples=1000,num_chains=nChains)

# Choose a random key and run over our model
rng_key = random.PRNGKey(114)
rng_key,rng_key_ = random.split(rng_key)

mcmc.run(rng_key_,spectra,weight_dictionary)
mcmc.print_summary()

# Save out data
data = az.from_numpyro(mcmc)
az.to_netcdf(data,"./../data/birefringence_variable_evolution.cdf")

