import numpy as np
from gwBackground import *
from geometry import *
import population_parameters
from tqdm import tqdm
import jax
from jax.config import config
config.update("jax_enable_x64", True)

def compute_likelihood_grids(zs,dRdV,clippingFunction=lambda x,y : False,massGridSize=(30,29),kappaGridSize=100,baselines=['HLO1','HLO2','HLO3','HVO3','LVO3'],joint=True):

    """
    Function to directly compute likelihoods over grids of birefringence coefficients.
    In particular, three calculations are done:

    1. A 1D calculation for the likelihood of :math:`\kappa_D`, fixing :math:`\kappa_z = 0`.
    2. A 1D calculation for the likelihood of :math:`\kappa_z`, fixing :math:`\kappa_D = 0`.
    3. A 2D calculation across both non-zero :math:`\kappa_D` and :math:`\kappa_z`.

    This function is called by

    * `run_birefringence_uniformRate.py`
    * `run_birefringence_SFR.py`
    * `run_birefringence_delayedSFR.py`
    * `run_birefringence_delayedSFR_HLO1.py`
    * `run_birefringence_delayedSFR_HLO2.py`
    * `run_birefringence_delayedSFR_HLO3.py`
    * `run_birefringence_delayedSFR_HVO3.py`
    * `run_birefringence_delayedSFR_LVO3.py`

    Parameters
    ----------
    zs : `array`
        An array of redshifts over which to integrate when computing the stochastic background
    dRdV : `array`
        An array, defined at `zs`, giving the BBH merger rate as a function of redshift
    clippingFunction : `func`
        Function to speed up calculation by skipping points in our 2D grid where the likelihood is known to be zero.
        Takes in two arguments, `kappa_D` and `kappa_z` and returns a Boolean value.
        If True, the point in question will be skipped and the log-likelihood manually fixed to -inf.
        Defaults to `lambda x,y : False`
    massGridSize : `tuple`
        A two element tuple determining the number of mass values over which to integrate when computing the stochastic background.
        Defaults to `(30,29)`.
    kappaGridSize : `int`
        Determines the number of gridpoints at which we compute likelihoods in the 2D calculation.
        Defaults to `100`
    baselines : `list`
        Allows for variations in which set of baselines/observing runs to include as observational input.
        Options are

        * `HLO1` : Hanford-Livingston O1
        * `HLO2` : Hanford-Livingston O2
        * `HLO3` : Hanford-Livingston O3
        * `HVO3` : Hanford-Virgo O3
        * `LVO3` : Livingston-Virgo O3

        Default is ['HLO1','HLO2','HLO3','HVO3','LVO3']
    joint : `bool`
        If False, will skip likelihood calculation over 2D grid. Default `True`

    Returns
    -------
    resultsDict : `dict`
        Dictionary containing results of the three direct likelihood calculations.
        Keys are the following:

        * `kappa_dcs_1D` : Array of $\kappa_D$ values across which 1D likelihood is computing
        * `probability_kappa_dc_1D` : Corresponding array of posterior probabilities
        * `kappa_zs_1D` : Array of $\kappa_z$ values across which 1D likelihood is computed
        * `probability_kappa_z_1D` : Corresponding array of posterior probabilities
        * `kappa_dcs_2D` : Array of $\kappa_D$ values specifying gridpoints in 2D calculation
        * `kappa_zs_2D` : Array of $\kappa_z$ values specifying gridpoints in 2D calculation
        * `probabilities` : 2D array of posterior probabilities, where `probabilities[i,j]` gives the posterior at `kappa_dcs_2D[i]` and `kappa_zs_2D[j]`
    """

    # Instantiate SGWB calculator
    m_absolute_min = 2.
    m_absolute_max = 100.
    omg = OmegaGW_BBH(m_absolute_min,m_absolute_max,zs,gridSize=massGridSize)

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

    # Load data
    f_H1L1_O1,C_H1L1_O1,sigma_H1L1_O1 = np.loadtxt('../input/H1L1_O1.dat',unpack=True,skiprows=1)
    f_H1L1_O2,C_H1L1_O2,sigma_H1L1_O2 = np.loadtxt('../input/H1L1_O2.dat',unpack=True,skiprows=1)
    f_H1L1_O3,C_H1L1_O3,sigma_H1L1_O3 = np.loadtxt('../input/H1L1_O3.dat',unpack=True,skiprows=1)
    f_H1V1_O3,C_H1V1_O3,sigma_H1V1_O3 = np.loadtxt('../input/H1V1_O3.dat',unpack=True,skiprows=1)
    f_L1V1_O3,C_L1V1_O3,sigma_L1V1_O3 = np.loadtxt('../input/L1V1_O3.dat',unpack=True,skiprows=1)

    # We have to do some cleanup of mistakes/oddities introduced upstream into the above data files

    # First, as discussed in the paper text, when constructing our signal model below it is necessary to divide
    # the Stokes-I overlap reduction function (ORF). As the ORF is an oscillatory function about zero, the resulting
    # signal model diverges to infinity at the roots of the ORF. Differences between the ORFs computed with python (our code) and
    # matlab libraries (LVK analysis code stochastic.m) place very high-order roots of the ORF at very slightly different locations, which
    # nevertheless yield statistically signficant differences in the likelihood calculation. When dividing by the ORF to construct
    # our signal model, it is therefore necessary to use the *same identical matlab calculation* of the ORF as in the upstream stochastic.m
    # analysis. Load this matlab-based ORF in preparation.
    matlab_orf_freqs,matlab_orf_H1L1,matlab_orf_H1V1,matlab_orf_L1V1 = np.loadtxt('../input/matlab_orfs.dat',unpack=True)

    # Second, the frequency arrays associated with the above LVK data files have been rounded to an insufficiently 
    # low number of significant digits. As a result, the precise frequencies are incorrect at the mHz level.
    # The frequency arrays should be regular grids with 1/32 Hz = 0.03125 Hz spacing.
    # At low frequencies the above arrays behave as expected:
    # 
    # >>> print(f_H1L1_O1[0:3])
    #   [20.      20.03125 20.0625 ]
    #
    # But at high frequencies rounding means that the precise grid spacing is no longer respected:
    #
    # >>> print(f_H1L1_O1[15360:15363])
    #   [500.     500.0312 500.0625]
    #
    # These small differences in frequencies are enough to noticeably perturb the signal model in places where the ORF
    # is near zero (e.g. the signal model is sent to infinity in slightly different locations)
    # Because of this, overwrite frequencies with array loaded in from original matlab calculation, which preserves
    # the correct frequency spacing.
    f_H1L1_O3 = matlab_orf_freqs[matlab_orf_freqs<=1726]
    matlab_orf_H1L1 = matlab_orf_H1L1[matlab_orf_freqs<=1726]
    matlab_orf_H1V1 = matlab_orf_H1V1[matlab_orf_freqs<=1726]
    matlab_orf_L1V1 = matlab_orf_L1V1[matlab_orf_freqs<=1726]

    # Instantiate baseline objects
    H1L1 = Baseline(Detector.H1(),Detector.L1())
    H1V1 = Baseline(Detector.H1(),Detector.V1())
    L1V1 = Baseline(Detector.L1(),Detector.V1())

    # Use baseline objects to compute overlap reduction functions for Stokes-I and Stokes-V signals
    H1L1_gammaI,H1L1_gammaV = H1L1.stokes_overlap_reduction_functions(f_H1L1_O3)
    H1V1_gammaI,H1V1_gammaV = H1V1.stokes_overlap_reduction_functions(f_H1L1_O3)
    L1V1_gammaI,L1V1_gammaV = L1V1.stokes_overlap_reduction_functions(f_H1L1_O3)

    # Dictionary to store results
    resultsDict = {}

    def log_likelihood(kd,kz):

        # Construct model backgrounds
        OmgI,OmgV = omg.eval(R0,dRdV,f_H1L1_O3,kd,kz)
        
        # Construct baseline-dependent models
        model_background_HL = OmgI + (H1L1_gammaV/matlab_orf_H1L1)*OmgV
        model_background_HV = OmgI + (H1V1_gammaV/matlab_orf_H1V1)*OmgV
        model_background_LV = OmgI + (L1V1_gammaV/matlab_orf_L1V1)*OmgV
        
        # Given this prediction, add the log-likelihoods of our observations from each baseline
        log_likelihoods_dc = 0
        if 'HLO1' in baselines:
            log_likelihoods_dc += jnp.sum(-(C_H1L1_O1-model_background_HL)**2/(2.*sigma_H1L1_O1**2))
        if 'HLO2' in baselines:
            log_likelihoods_dc += jnp.sum(-(C_H1L1_O2-model_background_HL)**2/(2.*sigma_H1L1_O2**2))
        if 'HLO3' in baselines:
            log_likelihoods_dc += jnp.sum(-(C_H1L1_O3-model_background_HL)**2/(2.*sigma_H1L1_O3**2))
        if 'HVO3' in baselines:
            log_likelihoods_dc += jnp.sum(-(C_H1V1_O3-model_background_HV)**2/(2.*sigma_H1V1_O3**2))
        if 'LVO3' in baselines:
            log_likelihoods_dc += jnp.sum(-(C_L1V1_O3-model_background_LV)**2/(2.*sigma_L1V1_O3**2))

        return log_likelihoods_dc

    ##########################
    # 1. kappa_D
    ##########################

    # Range of kappa parameters
    # Instantiate an array to hold log likelihood values, and loop across kappas
    kappa_dcs_1D = np.linspace(-0.2,0.2,400)
    log_likelihoods_dc = np.zeros_like(kappa_dcs_1D)

    for i,k in tqdm(enumerate(kappa_dcs_1D),total=kappa_dcs_1D.size):
        log_likelihoods_dc[i] = log_likelihood(k,0)
        
    # Subtract off the max so numpy doesn't freak out when we exponentiate, then exponentiate
    log_likelihoods_dc -= np.max(log_likelihoods_dc)
    likelihoods_dc = np.exp(log_likelihoods_dc)

    # Normalize to obtain a proper probability distribution
    p_kappa_dc_uniform = likelihoods_dc/np.trapz(likelihoods_dc,kappa_dcs_1D)

    # Store
    resultsDict['kappa_dcs_1D'] = kappa_dcs_1D
    resultsDict['probability_kappa_dc_1D'] = p_kappa_dc_uniform

    ##########################
    # 2. kappa_z
    ##########################

    # Grid of kappas
    kappa_zs_1D = np.linspace(-0.4,0.4,400)

    # Loop across kappas
    log_likelihoods_z = np.zeros_like(kappa_zs_1D)
    for i,k in tqdm(enumerate(kappa_zs_1D),total=kappa_zs_1D.size):
        log_likelihoods_z[i] = log_likelihood(0,k)

    # Exponentiate and form a normalized probability distribution
    log_likelihoods_z -= np.max(log_likelihoods_z)
    likelihoods_z = np.exp(log_likelihoods_z)
    p_kappa_z_uniform = likelihoods_z/np.trapz(likelihoods_z,kappa_zs_1D)

    # Store
    resultsDict['kappa_zs_1D'] = kappa_zs_1D
    resultsDict['probability_kappa_z_1D'] = p_kappa_z_uniform

    ##########################
    # 3. kappa_z
    ##########################

    if joint==True:

        # Grid over both kappas
        kappa_dcs = np.linspace(-0.5,0.5,kappaGridSize)
        kappa_zs = np.linspace(-1.,1.,kappaGridSize-1)

        # Instantiate array to hold log likelihoods
        log_likelihoods_joint = np.zeros((kappa_dcs.size,kappa_zs.size))
        clipped = np.zeros((kappa_dcs.size,kappa_zs.size))

        # Loop over pairs of kappas
        for i,kd in tqdm(enumerate(kappa_dcs),total=kappa_dcs.size):
            for j,kz in enumerate(kappa_zs):
                
                # Don't bother evaluating regions that are way outside the region of likelihood support
                if clippingFunction(kd,kz):
                    log_likelihoods_joint[i,j] = -np.inf
                    clipped[i,j] = 1

                else:
                    log_likelihoods_joint[i,j] = log_likelihood(kd,kz)

        log_likelihoods_joint[np.isnan(log_likelihoods_joint)] = -np.inf

        # Exponentiate and normalize
        log_likelihoods_joint -= np.max(log_likelihoods_joint)
        likelihoods_joint = np.exp(log_likelihoods_joint)
        p_joint_uniform = likelihoods_joint/(np.sum(likelihoods_joint)*(kappa_dcs[1]-kappa_dcs[0])*(kappa_zs[1]-kappa_zs[0]))

        # Store
        resultsDict['kappa_dcs_2D'] = kappa_dcs
        resultsDict['kappa_zs_2D'] = kappa_zs
        resultsDict['probabilities'] = p_joint_uniform

    return resultsDict

