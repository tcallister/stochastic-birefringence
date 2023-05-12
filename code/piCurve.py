import numpy as np
from load_data import *

def pi_curve(frequencies,sigmas,alpha_min,alpha_max):

    # Range of power-law indices
    alphas = np.linspace(alpha_min,alpha_max,30)
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

spectra = get_all_data()
freqs = spectra['H1L1_O3'][0]
sigmas = spectra['H1L1_O3'][2]

pi_curve(freqs,sigmas,-8,8)
