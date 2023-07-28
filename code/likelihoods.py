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

    """
    Likelihood to perform a standard power-law inference on the stochastic background, for use within `numpyro`.
    Stochastic background is assumed to be unpolarized and governed by two parameters:
       * `alpha` : Power-law index on energy density
       * `log_Omega` : log10 of the energy-density amplitude at f=25 Hz

    Parameters
    ----------
    spectra : `dict`
        Dictionary containing cross-correlation measurements and uncertainties, as prepared by `load_data.get_all_data()`
    """

    # Sample power-law parameters
    alpha = numpyro.sample("alpha",dist.Normal(0,3.5))

    # Amplitude
    logit_log_Omega = numpyro.sample("logit_log_Omega",dist.Normal(0,logit_std))
    log_Omega,jac_log_Omega = get_value_from_logit(logit_log_Omega,-13.,-5.)
    numpyro.factor("p_log_Omega",logit_log_Omega**2/(2.*logit_std**2)-jnp.log(jac_log_Omega))
    numpyro.deterministic("log_Omega",log_Omega)

    # Define function to evalute likelihood of cross-correlation measurements
    def model_and_observe(freqs,Ys,sigmas):
        Omega_model = 10.**log_Omega * jnp.power(freqs/25.,alpha)
        logp = jnp.sum(-(Ys-Omega_model)**2/(2.*sigmas**2))
        return logp

    # Map log-likelihood function across all baselines and observing runs
    log_ps = jnp.array([model_and_observe(*spectra[k]) for k in spectra])
    numpyro.factor("logp",jnp.sum(log_ps))

def right_left(spectra):

    """
    Likelihood to perform inference on a possibly-circularly-polarized stochastic background, for use within `numpyro`.
    Stochastic background is parametrized by right vs. left energy-densities, with the following four parameters:
       * `alpha_R` : Power-law index on energy density in right-circular polarizations
       * `alpha_L` : Power-law index on energy density in left-circular polarizations
       * `log_Omega_R` : log10 of the energy-density amplitude at f=25 Hz in right-circular polarizations
       * `log_Omega_L` : log10 of the energy-density amplitude at f=25 Hz in left-circular polarizations

    Parameters
    ----------
    spectra : `dict`
        Dictionary containing cross-correlation measurements and uncertainties, as prepared by `load_data.get_all_data()`
    """

    # Sample power-law parameters
    alpha_R = numpyro.sample("alpha_R",dist.Normal(0,3.5))
    alpha_L = numpyro.sample("alpha_L",dist.Normal(0,3.5))

    # Amplitude
    logit_log_Omega_R = numpyro.sample("logit_log_Omega_R",dist.Normal(0,logit_std))
    logit_log_Omega_L = numpyro.sample("logit_log_Omega_L",dist.Normal(0,logit_std))
    log_Omega_R,jac_log_Omega_R = get_value_from_logit(logit_log_Omega_R,-13.,-5.)
    log_Omega_L,jac_log_Omega_L = get_value_from_logit(logit_log_Omega_L,-13.,-5.)
    numpyro.factor("p_log_Omega_R",logit_log_Omega_R**2/(2.*logit_std**2)-jnp.log(jac_log_Omega_R))
    numpyro.factor("p_log_Omega_L",logit_log_Omega_L**2/(2.*logit_std**2)-jnp.log(jac_log_Omega_L))
    numpyro.deterministic("log_Omega_R",log_Omega_R)
    numpyro.deterministic("log_Omega_L",log_Omega_L)

    # Define function to evalute likelihood of cross-correlation measurements
    def model_and_observe(freqs,Ys,sigmas,orfR,orfL):

        # Construct right- and left-handed spectra
        Omega_model_R = 10.**log_Omega_R * jnp.power(freqs/25.,alpha_R)
        Omega_model_L = 10.**log_Omega_L * jnp.power(freqs/25.,alpha_L)

        # Construct total model and compute log-likelihood
        # Note that `Ys` are normalized such that we need to divide model by the Stokes-I ORF
        orfI = orfR + orfL
        total_model = Omega_model_R*orfR/orfI + Omega_model_L*orfL/orfI
        logp = jnp.sum(-(Ys-total_model)**2/(2.*sigmas**2))

        return logp

    # Map log-likelihood function across all baselines and observing runs
    log_ps = jnp.array([model_and_observe(*spectra[k]) for k in spectra])
    numpyro.factor("logp",jnp.sum(log_ps))

def stokes(spectra):

    """
    Likelihood to perform inference on a possibly-circularly-polarized stochastic background, for use within `numpyro`.
    Stochastic background is parametrized by Stokes I and V amplitudes, with the following four parameters:
       * `alpha_I` : Power-law index on total energy density
       * `log_Omega_I` : log10 of the total energy-density amplitude at f=25 Hz 
       * `log_Omega_L` : log10 of the energy-density amplitude at f=25 Hz in left-circular polarizations

    Parameters
    ----------
    spectra : `dict`
        Dictionary containing cross-correlation measurements and uncertainties, as prepared by `load_data.get_all_data()`
    """

    # Sample power-law parameters
    alpha_I = numpyro.sample("alpha_I",dist.Normal(0,3.5))

    # Total energy-density amplitude
    logit_log_Omega_I = numpyro.sample("logit_log_Omega_I",dist.Normal(0,logit_std))
    log_Omega_I,jac_log_Omega_I = get_value_from_logit(logit_log_Omega_I,-13.,-5.)
    numpyro.factor("p_log_Omega_I",logit_log_Omega_I**2/(2.*logit_std**2)-jnp.log(jac_log_Omega_I))
    numpyro.deterministic("log_Omega_I",log_Omega_I)

    # Polarization fraction; when multiplyed by `Omega_I`, this is the energy density in Stokes V
    logit_pol_fraction = numpyro.sample("logit_pol_fraction",dist.Normal(0,logit_std))
    pol_fraction,jac_pol_fraction = get_value_from_logit(logit_pol_fraction,-1.,1.)
    numpyro.factor("p_pol_fraction",logit_pol_fraction**2/(2.*logit_std**2)-jnp.log(jac_pol_fraction))
    numpyro.deterministic("pol_fraction",pol_fraction)

    # Define function to evalute likelihood of cross-correlation measurements
    def model_and_observe(freqs,Ys,sigmas,orfI,orfV):

        # Construct model
        # Note that Stokes V term needs to be divided by the Stokes I ORF to match normalization of `Ys`
        Omega_model_I = 10.**log_Omega_I * jnp.power(freqs/25.,alpha_I)
        total_model = Omega_model_I*(1.+orfV/orfI*pol_fraction)

        # Compute likelihood
        logp = jnp.sum(-(Ys-total_model)**2/(2.*sigmas**2))
        return logp

    # Map log-likelihood function across all baselines and observing runs
    log_ps = jnp.array([model_and_observe(*spectra[k]) for k in spectra])
    numpyro.factor("logp",jnp.sum(log_ps))

def generateMonteCarloEnergies(nsamples,freqs):

    """
    Helper function to enable Monte Carlo calculation of stochastic energy-density spectra.
    Draws an ensemble of BBHs and their associated energy densities; these energy densities can then be
    reweighted to compute Omega(f) under a variety of BBH populations and/or birefringent scenarios

    Parameters
    ----------
    nsamples : `int`
        Size of BBH ensemble to draw
    freqs : `array`
        Array of frequencies at which to evaluate energy spectra

    Returns
    -------
    mc_weights : `array`
        The mean of these weights gives the stochastic energy density Omega(f) arising from our default BBH population
    z_samples : `array`
        Redshifts of each BBH in our ensemble
    dRdV_samples : `array`
        The comoving merger rate density at each redshift in `z_samples`; divide by this if you wish to reweight `mc_weights` to another redshift distribution
    """

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
    R0 = 20.
    alpha = 2.7
    beta = 5.6
    zpeak = 1.9
    z_grid = np.linspace(0,10,3000)
    dRdV = np.power(1.+z_grid,alpha)/(1.+np.power((1.+z_grid)/(1.+zpeak),beta))
    dRdV *= (R0/1e9/year)/dRdV[0]   # Convert to number per Mpc^3 per sec to match units

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
    mc_weights = integrandNorm*dEdf_samples*freqs/rhoC/H0

    # To enable reweighting, compute the merger rate density that went into our sample
    # We can divide by this if we want to reweight to some other population
    dRdV_samples = np.power(1.+z_samples,alpha)/(1.+np.power((1.+z_samples)/(1.+zpeak),beta))
    dRdV_0 = 1./(1.+np.power(1./(1.+zpeak),beta))
    dRdV_samples *= R0/dRdV_0

    return mc_weights,z_samples,dRdV_samples

def amplification_argument(kappa_Dc,kappa_z,Dcs_fs,zs_fs):

    """
    Helper function to compute birefringent amplification factor appearing in hyperbolic factors modifying Stokes I and Stokes V

    Parameters
    ----------
    kappa_Dc : float
        Birefringent coefficient corresponding to comoving-distance-based amplification
    kappa_z : float
        Birefringent coefficient corresponding to redshift-based amplification
    Dcs_fs : array
        Expected to be a 2D outer product of comoving distances (units of Gpc) and frequencies, with `Dcs_fs[i,j]` corresponding
        to distance i and frequency j
    zs_fs : array
        Expected to be a 2D outer product of redshifts and frequencies, as above.

    Returns
    -------
    amp_factor : array
        Total amplification factor at the corresponding comoving distance/redshift and frequency
    """

    # Combine
    # Note that kappas are defined with reference to a 100 Hz reference frequency
    amp_factor = 2.*np.pi*(kappa_Dc*Dcs_fs + kappa_z*zs_fs)/100.
    return amp_factor

def birefringence(spectra,weight_dictionary):

    """
    Likelihood to perform inference on a birefringently-amplified stochastic background, for use within `numpyro`.

    Parameters
    ----------
    spectra : `dict`
        Dictionary containing cross-correlation measurements and uncertainties, as prepared by `load_data.get_all_data()`
    weight_dictionary : `dict`
        Dictionary containing ensemble of BBHs and associated contributions to Omega(f); reweighted to perform Monte Carlo calculation of amplified energy-densities
    """
    
    # Draw comoving-distance birefringence parameter
    logit_kappa_Dc = numpyro.sample("logit_kappa_Dc",dist.Normal(0,logit_std))
    kappa_Dc,jac_kappa_Dc = get_value_from_logit(logit_kappa_Dc,0,0.4)
    numpyro.factor("p_kappa_Dc",logit_kappa_Dc**2/(2.*logit_std**2)-jnp.log(jac_kappa_Dc))
    numpyro.deterministic("kappa_Dc",kappa_Dc)

    # Draw redshift birefringence parameter
    logit_kappa_z = numpyro.sample("logit_kappa_z",dist.Normal(0,logit_std))
    kappa_z,jac_kappa_z = get_value_from_logit(logit_kappa_z,0,1)
    numpyro.factor("p_kappa_z",logit_kappa_z**2/(2.*logit_std**2)-jnp.log(jac_kappa_z))
    numpyro.deterministic("kappa_z",kappa_z)

    # Extract data from reference ensemble for reweighting
    sample_frequencies = weight_dictionary['freqs']
    sample_Dcs_fs = weight_dictionary['Dcs_outer_freqs']
    sample_zs_fs = weight_dictionary['zs_outer_freqs']
    sample_omg_weights = weight_dictionary['omg_weights']

    # Compute birefringent amplification and boost each sample event's contributions accordingly
    amp_argument = amplification_argument(kappa_Dc,kappa_z,sample_Dcs_fs,sample_zs_fs)
    Omg_I_weights = sample_omg_weights*jnp.cosh(amp_argument)
    Omg_V_weights = sample_omg_weights*jnp.sinh(amp_argument)

    # Save model as well as frequency-dependent effective sample counts
    Omg_I_model = numpyro.deterministic("Omg_I_model",jnp.mean(Omg_I_weights,axis=0))
    Omg_V_model = numpyro.deterministic("Omg_V_model",jnp.mean(Omg_V_weights,axis=0))
    Omg_I_neff = numpyro.deterministic("Omg_I_neff",jnp.sum(Omg_I_weights,axis=0)**2/jnp.sum(Omg_I_weights**2,axis=0))
    Omg_V_neff = numpyro.deterministic("Omg_V_neff",jnp.sum(Omg_V_weights,axis=0)**2/jnp.sum(Omg_V_weights**2,axis=0))

    # Define function to evalute likelihood of cross-correlation measurements
    def model_and_observe(freqs,Ys,sigmas,orfI,orfV):

        # Construct model
        Omega_I_interpolated = jnp.interp(freqs,sample_frequencies,Omg_I_model) 
        Omega_V_interpolated = jnp.interp(freqs,sample_frequencies,Omg_V_model) 
        total_model = Omega_I_interpolated + (orfV/orfI)*Omega_V_interpolated

        # Compute likelihood
        logp = jnp.sum(-(Ys-total_model)**2/(2.*sigmas**2))
        return logp

    # Map log-likelihood function across all baselines and observing runs
    log_ps = jnp.array([model_and_observe(*spectra[k]) for k in spectra])
    numpyro.factor("logp",jnp.sum(log_ps))

def birefringence_variable_evolution(spectra,weight_dictionary):

    """
    Likelihood to perform inference on a birefringently-amplified stochastic background, for use within `numpyro`.

    Parameters
    ----------
    spectra : `dict`
        Dictionary containing cross-correlation measurements and uncertainties, as prepared by `load_data.get_all_data()`
    weight_dictionary : `dict`
        Dictionary containing ensemble of BBHs and associated contributions to Omega(f); reweighted to perform Monte Carlo calculation of amplified energy-densities
    """
    
    # Draw comoving-distance birefringence parameter
    kappa_Dc = numpyro.sample("kappa_Dc",dist.Uniform(-0.4,0.4))
    """
    logit_kappa_Dc = numpyro.sample("logit_kappa_Dc",dist.Normal(0,logit_std))
    kappa_Dc,jac_kappa_Dc = get_value_from_logit(logit_kappa_Dc,0,0.4)
    numpyro.factor("p_kappa_Dc",logit_kappa_Dc**2/(2.*logit_std**2)-jnp.log(jac_kappa_Dc))
    numpyro.deterministic("kappa_Dc",kappa_Dc)
    """

    # Draw redshift birefringence parameter
    kappa_z = numpyro.sample("kappa_z",dist.Uniform(-1.,1.))
    """
    logit_kappa_z = numpyro.sample("logit_kappa_z",dist.Normal(0,logit_std))
    kappa_z,jac_kappa_z = get_value_from_logit(logit_kappa_z,0,0.8)
    numpyro.factor("p_kappa_z",logit_kappa_z**2/(2.*logit_std**2)-jnp.log(jac_kappa_z))
    numpyro.deterministic("kappa_z",kappa_z)
    """

    # Draw parameters governing rate of BBHs
    log_R0 = numpyro.sample("log_R0",dist.Normal(jnp.log(15.),jnp.log(20./15.)))
    alpha = numpyro.sample("alpha",dist.Normal(3.,1.5))
    zp = numpyro.sample("zp",dist.Uniform(0.5,4))
    beta = numpyro.sample("beta",dist.Uniform(0,10))
    R0 = jnp.exp(log_R0)

    # Extract data from reference ensemble for reweighting
    sample_frequencies = weight_dictionary['freqs']
    sample_Dcs_fs = weight_dictionary['Dcs_outer_freqs']
    sample_zs_fs = weight_dictionary['zs_outer_freqs']
    sample_zs = weight_dictionary['zs']
    sample_omg_weights = weight_dictionary['omg_weights']
    sample_old_dRdVs = weight_dictionary['dRdVs']

    # Compute new merger rate factors
    dRdV_norm = 1./(1.+jnp.power(1./(1.+zp),beta))
    sample_new_dRdVs = jnp.power(1.+sample_zs,alpha)/(1.+jnp.power((1.+sample_zs)/(1.+zp),beta))
    sample_new_dRdVs *= R0/dRdV_norm

    # Compute birefringent amplification and boost each sample event's contributions accordingly
    amp_argument = amplification_argument(kappa_Dc,kappa_z,sample_Dcs_fs,sample_zs_fs)
    Omg_weights_unamplified = sample_omg_weights*(sample_new_dRdVs/sample_old_dRdVs)[:,jnp.newaxis]
    Omg_I_weights = Omg_weights_unamplified*jnp.cosh(amp_argument)
    Omg_V_weights = Omg_weights_unamplified*jnp.sinh(amp_argument)

    # Save model as well as frequency-dependent effective sample counts
    Omg_I_model = numpyro.deterministic("Omg_I_model",jnp.mean(Omg_I_weights,axis=0))
    Omg_V_model = numpyro.deterministic("Omg_V_model",jnp.mean(Omg_V_weights,axis=0))
    Omg_I_neff = numpyro.deterministic("Omg_I_neff",jnp.sum(Omg_I_weights,axis=0)**2/jnp.sum(Omg_I_weights**2,axis=0))
    Omg_V_neff = numpyro.deterministic("Omg_V_neff",jnp.sum(Omg_V_weights,axis=0)**2/jnp.sum(Omg_V_weights**2,axis=0))

    # Define function to evalute likelihood of cross-correlation measurements
    def model_and_observe(freqs,Ys,sigmas,orfI,orfV):

        # Construct model
        Omega_I_interpolated = jnp.interp(freqs,sample_frequencies,Omg_I_model) 
        Omega_V_interpolated = jnp.interp(freqs,sample_frequencies,Omg_V_model) 
        total_model = Omega_I_interpolated + (orfV/orfI)*Omega_V_interpolated

        # Compute likelihood
        logp = jnp.sum(-(Ys-total_model)**2/(2.*sigmas**2))
        return logp

    # Map log-likelihood function across all baselines and observing runs
    log_ps = jnp.array([model_and_observe(*spectra[k]) for k in spectra])
    numpyro.factor("logp",jnp.sum(log_ps))
