import numpyro
nChains = 3
numpyro.set_host_device_count(nChains)
from numpyro.infer import NUTS,MCMC
from numpyro.infer.initialization import init_to_value
from jax.config import config
config.update("jax_enable_x64", True)
from jax import random
import arviz as az
from numpyro_likelihoods import birefringence_variable_evolution_massGrid
from load_data import get_all_data
from geometry import *
from gwBackground import OmegaGW_BBH
from astropy.cosmology import Planck18
import astropy.units as u
import population_parameters

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

# Instantiate SGWB calculator
m_absolute_min = 2.001
m_absolute_max = 99.999
zs = np.linspace(0,17,200)
omg = OmegaGW_BBH(m_absolute_min,m_absolute_max,zs,gridSize=(30,29))

# Define hyperparameters describing mass distribution
R0 = population_parameters.R0
m_min = population_parameters.m_min
m_max = population_parameters.m_max
dm_min = population_parameters.dm_min
dm_max = population_parameters.dm_max
alpha_m = population_parameters.alpha_m
mu_peak = population_parameters.mu_peak
sig_peak = population_parameters.sig_peak
frac_peak = population_parameters.frac_peak
bq = population_parameters.bq

# Pass these to our SGWB calculator
omg.setProbs_plPeak(m_min,m_max,dm_min,dm_max,alpha_m,mu_peak,sig_peak,frac_peak,bq)

# Set up NUTS sampler over our likelihood
init_params = {'logit_kappa_x':0.,'logit_kappa_y':0.}
kernel = NUTS(birefringence_variable_evolution_massGrid,target_accept_prob=0.98,dense_mass=True,init_strategy=init_to_value(values=init_params))
mcmc = MCMC(kernel,num_warmup=2500,num_samples=1500,num_chains=nChains)

# Choose a random key and run over our model
rng_key = random.PRNGKey(116)
rng_key,rng_key_ = random.split(rng_key)

mcmc.run(rng_key_,spectra,omg)
mcmc.print_summary()

# Save out data
data = az.from_numpyro(mcmc)
az.to_netcdf(data,"./../data/birefringence_variable_evolution.cdf")

