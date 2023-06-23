import jax.numpy as jnp
from jax import vmap
import numpy as np
import numpyro
import numpyro.distributions as dist
from constants import *
from gwBackground import dEdf

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

def generateMonteCarloEnergies(nsamples,freqs):

    # Parameters specifying BBH mass distribution
    m_min = 8.
    m_max = 100.
    alpha = -3.
    mu_peak = 35.
    sig_peak = 5.
    frac_peak = 0.05
    bq = 2 

    # Construct normalized probability distribution over m1 grid
    m1_grid = np.linspace(m_min,m_max,1000)
    p_m1_pl = (1.+alpha)*m1_grid**alpha/(m_max**(1.+alpha) - m_min**(1.+alpha))
    p_m1_peak = np.exp(-(m1_grid-mu_peak)**2/(2.*sig_peak**2))/np.sqrt(2.*np.pi*sig_peak**2)
    p_m1_grid = frac_peak*p_m1_peak + (1.-frac_peak)*p_m1_pl
    cdf_m1_grid = np.cumsum(p_m1_grid)/np.sum(p_m1_grid)

    # Draw random primary masses
    m1_samples = np.interp(np.random.random(nsamples),cdf_m1_grid,m1_grid)

    # Draw random secondary masses
    m2_samples = np.power(m_min**(1.+bq) + np.random.random(nsamples)*(m1_samples**(1.+bq) - m_min**(1.+bq)),1./(1.+bq))

    # Now construct integrand of redshift integral
    # First we need the comoving merger rate density
    # Sample from a fiducial Madau+Dickinson model
    R0 = 20./1e9/year   # Convert to number per Mpc^3 per sec to match units
    alpha = 2.7
    beta = 5.6
    zpeak = 1.9
    z_grid = np.linspace(0,10,3000)
    dRdV = np.power(1.+z_grid,alpha)/(1.+np.power((1.+z_grid)/(1.+zpeak),beta))
    dRdV *= R0/dRdV[0]

    # Construct full integrand and normalize to obtain a probability distribution
    # Note that this is **not** any kind of physical probability distribution over source redshifts,
    # but just a normalized version of the integrand we can use to draw monte carlo samples
    integrand = dRdV/((1.+z_grid)*np.sqrt(OmgM*(1.+z_grid)**3+OmgL))
    integrandNorm = np.trapz(integrand,z_grid)
    p_z = integrand/integrandNorm
    cdf_z = np.cumsum(p_z)/np.sum(p_z)

    # Draw redshifts
    z_samples = np.interp(np.random.random(nsamples),cdf_z,z_grid)
    
    # Compute energy spectra at each sample
    Mtot_samples = m1_samples+m2_samples
    q_samples = m2_samples/m1_samples
    eta_samples = q_samples/(1.+q_samples)**2
    dEdf_samples = np.array([dEdf(Mtot_samples[i],freqs*(1.+z_samples[i]),eta_samples[i]) for i in range(nsamples)])

    # Finally, compute relevant monte carlo weights
    dRdV_samples = np.power(1.+z_samples,alpha)/(1.+np.power((1.+z_samples)/(1.+zpeak),beta))
    mc_weights = integrandNorm*dEdf_samples*freqs/rhoC/H0

    return mc_weights,dRdV_samples

