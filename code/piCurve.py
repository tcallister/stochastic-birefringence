import numpy as np
from geometry import Detector,Baseline
import sys,os
script_directory = os.path.dirname(os.path.realpath(__file__))

def pi_curve(frequencies,sigmas,alpha_min,alpha_max):

    """
    Function to compute PI curve corresponding to a set of narrowband sigma(f) uncertainties.
    CAUTION: This code assumes that sigma(f) follows the convention in which the cross-correlation
    measurements C(f) are direct estimates of Omega_I(f), rather than gamma_I*Omega_I + gamma_V*Omega_V.

    Parameters
    ----------
    frequencies : `array`
        Array of frequencies at which data are defined
    sigmas : `array`
        Corresponding array of narrowband uncertainties. See note above about normalization
    alpha_min : `float`
        Minimum signal power-law index to consider
    alpha_max : `float`
        Maximum signal power-law index to consider

    Returns
    -------
    pi : `array`
        Power-law integrated curve defined across `frequencies`
    """

    # Range of power-law indices
    alphas = np.linspace(alpha_min,alpha_max,50)
    amplitudes = np.zeros(alphas.size)

    # Loop across indices
    for i,alpha in enumerate(alphas):

        # Compute power-law model with a reference amplitude of one
        omega_unscaled = np.power(frequencies/25.,alpha)

        # Compute reference snr
        reference_snr = np.sqrt(np.sum(omega_unscaled**2/sigmas**2))
        detectable_amp = 1./reference_snr
        amplitudes[i] = detectable_amp

    # Define arrays holding all power-laws
    power_law_locus = np.array([amplitudes[i]*(frequencies/25.)**alphas[i] for i in range(alphas.size)])

    pi = np.max(power_law_locus,axis=0)
    return pi

def stokes_I_PI():

    """
    Function to compute and return PI curve for a Stokes-I gravitational-wave background, integrated
    over all detector pairs and observing runs, using the `pi_curve` function.

    Parameters
    ----------
    None

    Returns
    -------
    frequencies : `array`
        Array of frequencies at which PI curve will be defined
    manual_PI_curve : `array`
        PI curve indicating the one-sigma sensitivity of all HLV baselines across all observing run
    """

    # Read data
    HL_O1_freqs,HL_O1_sigmas = np.loadtxt(script_directory+'/../input/H1L1_O1.dat',unpack=True,skiprows=1,usecols=(0,2))
    HL_O2_freqs,HL_O2_sigmas = np.loadtxt(script_directory+'/../input/H1L1_O2.dat',unpack=True,skiprows=1,usecols=(0,2))
    HL_O3_freqs,HL_O3_sigmas = np.loadtxt(script_directory+'/../input/H1L1_O3.dat',unpack=True,skiprows=1,usecols=(0,2))
    HV_O3_freqs,HV_O3_sigmas = np.loadtxt(script_directory+'/../input/H1V1_O3.dat',unpack=True,skiprows=1,usecols=(0,2))
    LV_O3_freqs,LV_O3_sigmas = np.loadtxt(script_directory+'/../input/L1V1_O3.dat',unpack=True,skiprows=1,usecols=(0,2))

    # Optimally combine uncertainties
    inv_full_sigmas_squared = 1./HL_O1_sigmas**2 + 1./HL_O2_sigmas**2 + 1./HL_O3_sigmas**2 + 1./HV_O3_sigmas**2 + 1./LV_O3_sigmas**2
    full_sigmas = 1./np.sqrt(inv_full_sigmas_squared)

    # And compute PI curve
    manual_PI_curve = pi_curve(HL_O1_freqs,full_sigmas,-10,10)
    return HL_O1_freqs,manual_PI_curve

def stokes_V_PI():

    """
    Function to compute and return PI curve for a Stokes-V gravitational-wave background, integrated
    over all detector pairs and observing runs, using the `pi_curve` function.

    Parameters
    ----------
    None

    Returns
    -------
    frequencies : `array`
        Array of frequencies at which PI curve will be defined
    manual_PI_curve : `array`
        PI curve indicating the one-sigma sensitivity of all HLV baselines across all observing run
    """

    # Load data
    HL_O1_freqs,HL_O1_sigmas = np.loadtxt(script_directory+'/../input/H1L1_O1.dat',unpack=True,skiprows=1,usecols=(0,2))
    HL_O2_freqs,HL_O2_sigmas = np.loadtxt(script_directory+'/../input/H1L1_O2.dat',unpack=True,skiprows=1,usecols=(0,2))
    HL_O3_freqs,HL_O3_sigmas = np.loadtxt(script_directory+'/../input/H1L1_O3.dat',unpack=True,skiprows=1,usecols=(0,2))
    HV_O3_freqs,HV_O3_sigmas = np.loadtxt(script_directory+'/../input/H1V1_O3.dat',unpack=True,skiprows=1,usecols=(0,2))
    LV_O3_freqs,LV_O3_sigmas = np.loadtxt(script_directory+'/../input/L1V1_O3.dat',unpack=True,skiprows=1,usecols=(0,2))

    # The above uncertainty spectra have been divided by the Stokes-I overlap reduction functions, in order to normalize
    # the corresponding cross-correlation spectra such that they are direct estimates of the Stokes-I energy density.
    # Here, our goal is to instead consider sensitivities to purely *Stokes-V* signals, and so we need to renormalize
    # our uncertainty spectra accordingly. In particular, we need to undo the division by Stokes-I overlap reduction functions,
    # and instead divide by the Stokes-V overlap reduction functions.
    # Along the way, we have to do some cleanup of mistakes/oddities introduced upstream into the above data files

    # As the ORF is an oscillatory function about zero, the resulting signal model diverges to infinity at the roots of the ORF.
    # Differences between the ORFs computed with python (our code) and matlab libraries (LVK analysis code stochastic.m)
    # place very high-order roots of the ORF at very slightly different locations, which nevertheless yield statistically signficant 
    # differences in the likelihood calculation. When dividing by the ORF to construct our signal model, it is therefore necessary to use
    # the *same identical matlab calculation* of the ORF as in the upstream stochastic.m analysis. Load this matlab-based ORF in preparation.
    matlab_orf_freqs,matlab_orf_H1L1,matlab_orf_H1V1,matlab_orf_L1V1 = np.loadtxt(script_directory+'/../input/matlab_orfs.dat',unpack=True)

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

    # Instantiate baseline objects to compute Stokes V ORF
    H1L1 = Baseline(Detector.H1(),Detector.L1())
    H1V1 = Baseline(Detector.H1(),Detector.V1())
    L1V1 = Baseline(Detector.L1(),Detector.V1())

    # Use baseline objects to compute overlap reduction functions for Stokes-I and Stokes-V signals
    H1L1_gammaI,H1L1_gammaV = H1L1.stokes_overlap_reduction_functions(f_H1L1_O3)
    H1V1_gammaI,H1V1_gammaV = H1V1.stokes_overlap_reduction_functions(f_H1L1_O3)
    L1V1_gammaI,L1V1_gammaV = L1V1.stokes_overlap_reduction_functions(f_H1L1_O3)

    # Rescale each baseline's sensitivity accordingly
    # Note that we need to specifically multiply by the *exact same* matlab ORF that went into the original
    # normalization of the uncertainty spectra
    HL_O1_sigmas = HL_O1_sigmas*(matlab_orf_H1L1/H1L1_gammaV)
    HL_O2_sigmas = HL_O2_sigmas*(matlab_orf_H1L1/H1L1_gammaV)
    HL_O3_sigmas = HL_O3_sigmas*(matlab_orf_H1L1/H1L1_gammaV)
    HV_O3_sigmas = HV_O3_sigmas*(matlab_orf_H1V1/H1V1_gammaV)
    LV_O3_sigmas = LV_O3_sigmas*(matlab_orf_L1V1/L1V1_gammaV)

    # Optimally combine uncertainties
    inv_full_sigmas_squared = 1./HL_O1_sigmas**2 + 1./HL_O2_sigmas**2 + 1./HL_O3_sigmas**2 + 1./HV_O3_sigmas**2 + 1./LV_O3_sigmas**2
    full_sigmas = 1./np.sqrt(inv_full_sigmas_squared)

    # And compute PI curve
    manual_PI_curve = pi_curve(HL_O1_freqs,full_sigmas,-10,10)
    return HL_O1_freqs,manual_PI_curve
