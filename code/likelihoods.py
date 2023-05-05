import jax.numpy as jnp
from jax import vmap
import numpyro
import numpyro.distributions as dist

def unpolarized(spectra):

    # Sample power-law parameters
    log_Omega = numpyro.sample("log_Omega",dist.Uniform(-13,-5)) 
    alpha = numpyro.sample("alpha",dist.Normal(0,3.5))

    # Construct proposed model background
    def model_and_observe(freqs,Ys,sigmas):
        Omega_model = 10.**log_Omega * jnp.power(freqs/25.,alpha)
        logp = jnp.sum(-(Ys-Omega_model)**2/(2.*sigmas**2))
        return logp

    #logps = vmap(model_and_observe)(jnp.array([spectra[k] for k in spectra]).T)
    log_ps = jnp.array([model_and_observe(*spectra[k]) for k in spectra])
    numpyro.factor("logp",jnp.sum(log_ps))
