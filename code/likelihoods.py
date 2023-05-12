import jax.numpy as jnp
from jax import vmap
import numpyro
import numpyro.distributions as dist

logit_std = 2.5

def get_value_from_logit(logit_x,x_min,x_max):

    """
    Function to map a variable `logit_x`, defined on `(-inf,+inf)`, to a quantity `x`
    defined on the interval `(x_min,x_max)`.

    Parameters
    ----------
    logit_x : float
        Quantity to inverse-logit transform
    x_min : float
        Lower bound of `x`
    x_max : float
        Upper bound of `x`

    Returns
    -------
    x : float
       The inverse logit transform of `logit_x`
    dlogit_dx : float
       The Jacobian between `logit_x` and `x`; divide by this quantity to convert a uniform prior on `logit_x` to a uniform prior on `x`
    """

    exp_logit = jnp.exp(logit_x)
    x = (exp_logit*x_max + x_min)/(1.+exp_logit)
    dlogit_dx = 1./(x-x_min) + 1./(x_max-x)

    return x,dlogit_dx

def unpolarized(spectra):

    # Sample power-law parameters
    alpha = numpyro.sample("alpha",dist.Normal(0,3.5))

    logit_log_Omega = numpyro.sample("logit_log_Omega",dist.Normal(0,logit_std))
    log_Omega,jac_log_Omega = get_value_from_logit(logit_log_Omega,-13.,-5.)
    numpyro.factor("p_log_Omega",logit_log_Omega**2/(2.*logit_std**2)-jnp.log(jac_log_Omega))
    numpyro.deterministic("log_Omega",log_Omega)

    # Construct proposed model background
    def model_and_observe(freqs,Ys,sigmas):
        Omega_model = 10.**log_Omega * jnp.power(freqs/25.,alpha)
        logp = jnp.sum(-(Ys-Omega_model)**2/(2.*sigmas**2))
        return logp

    #logps = vmap(model_and_observe)(jnp.array([spectra[k] for k in spectra]).T)
    log_ps = jnp.array([model_and_observe(*spectra[k]) for k in spectra])
    numpyro.factor("logp",jnp.sum(log_ps))

def right_left(spectra):

    # Sample power-law parameters
    alpha_R = numpyro.sample("alpha_R",dist.Normal(0,3.5))
    alpha_L = numpyro.sample("alpha_L",dist.Normal(0,3.5))

    logit_log_Omega_R = numpyro.sample("logit_log_Omega_R",dist.Normal(0,logit_std))
    logit_log_Omega_L = numpyro.sample("logit_log_Omega_L",dist.Normal(0,logit_std))
    log_Omega_R,jac_log_Omega_R = get_value_from_logit(logit_log_Omega_R,-13.,-5.)
    log_Omega_L,jac_log_Omega_L = get_value_from_logit(logit_log_Omega_L,-13.,-5.)
    numpyro.factor("p_log_Omega_R",logit_log_Omega_R**2/(2.*logit_std**2)-jnp.log(jac_log_Omega_R))
    numpyro.factor("p_log_Omega_L",logit_log_Omega_L**2/(2.*logit_std**2)-jnp.log(jac_log_Omega_L))
    numpyro.deterministic("log_Omega_R",log_Omega_R)
    numpyro.deterministic("log_Omega_L",log_Omega_L)

    # Construct proposed model background
    def model_and_observe(freqs,Ys,sigmas,orfR,orfL):

        orfI = orfR + orfL
        Omega_model_R = 10.**log_Omega_R * jnp.power(freqs/25.,alpha_R)
        Omega_model_L = 10.**log_Omega_L * jnp.power(freqs/25.,alpha_L)

        total_model = Omega_model_R*orfR/orfI + Omega_model_L*orfL/orfI
        logp = jnp.sum(-(Ys-total_model)**2/(2.*sigmas**2))

        return logp

    log_ps = jnp.array([model_and_observe(*spectra[k]) for k in spectra])
    numpyro.factor("logp",jnp.sum(log_ps))

def stokes(spectra):

    # Sample power-law parameters
    alpha_I = numpyro.sample("alpha_I",dist.Normal(0,3.5))

    logit_log_Omega_I = numpyro.sample("logit_log_Omega_I",dist.Normal(0,logit_std))
    log_Omega_I,jac_log_Omega_I = get_value_from_logit(logit_log_Omega_I,-13.,-5.)
    numpyro.factor("p_log_Omega_I",logit_log_Omega_I**2/(2.*logit_std**2)-jnp.log(jac_log_Omega_I))
    numpyro.deterministic("log_Omega_I",log_Omega_I)

    logit_pol_fraction = numpyro.sample("logit_pol_fraction",dist.Normal(0,logit_std))
    pol_fraction,jac_pol_fraction = get_value_from_logit(logit_pol_fraction,-1.,1.)
    numpyro.factor("p_pol_fraction",logit_pol_fraction**2/(2.*logit_std**2)-jnp.log(jac_pol_fraction))
    numpyro.deterministic("pol_fraction",pol_fraction)

    # Construct proposed model background
    def model_and_observe(freqs,Ys,sigmas,orfI,orfV):

        Omega_model_I = 10.**log_Omega_I * jnp.power(freqs/25.,alpha_I)
        total_model = Omega_model_I*(1.+orfV/orfI*pol_fraction)

        logp = jnp.sum(-(Ys-total_model)**2/(2.*sigmas**2))

        return logp

    log_ps = jnp.array([model_and_observe(*spectra[k]) for k in spectra])
    numpyro.factor("logp",jnp.sum(log_ps))

