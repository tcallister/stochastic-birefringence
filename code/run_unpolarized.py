import numpyro
nChains = 3
numpyro.set_host_device_count(nChains)
from numpyro.infer import NUTS,MCMC
from jax.config import config
config.update("jax_enable_x64", True)
from jax import random
import arviz as az
from likelihoods import unpolarized
from load_data import get_all_data

# Get dictionaries holding injections and posterior samples
spectra = get_all_data()

# Set up NUTS sampler over our likelihood
kernel = NUTS(unpolarized,target_accept_prob=0.95,dense_mass=True)
mcmc = MCMC(kernel,num_warmup=2000,num_samples=2000,num_chains=nChains)

# Choose a random key and run over our model
rng_key = random.PRNGKey(113)
rng_key,rng_key_ = random.split(rng_key)

mcmc.run(rng_key_,spectra)
mcmc.print_summary()

# Save out data
data = az.from_numpyro(mcmc)
az.to_netcdf(data,"./../data/unpolarized.cdf")

