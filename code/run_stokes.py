import numpyro
nChains = 3
numpyro.set_host_device_count(nChains)
from numpyro.infer import NUTS,MCMC
from jax.config import config
config.update("jax_enable_x64", True)
from jax import random
import arviz as az
from likelihoods import stokes
from load_data import get_all_data
from geometry import *

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

# Set up NUTS sampler over our likelihood
kernel = NUTS(stokes,target_accept_prob=0.95,dense_mass=True)
mcmc = MCMC(kernel,num_warmup=2000,num_samples=1000,num_chains=nChains)

# Choose a random key and run over our model
rng_key = random.PRNGKey(116)
rng_key,rng_key_ = random.split(rng_key)

mcmc.run(rng_key_,spectra)
mcmc.print_summary()

# Save out data
data = az.from_numpyro(mcmc)
az.to_netcdf(data,"./../data/stokes.cdf")

