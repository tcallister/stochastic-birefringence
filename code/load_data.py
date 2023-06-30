import numpy as np
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

def read_data(filename,f_max=400):

    # Load 
    frequencies,Ys,sigmas = np.loadtxt(filename,unpack=True,skiprows=1)

    # Limit to low and unnotched frequencies
    elements_to_keep = (~np.isinf(sigmas))
    frequencies = frequencies[elements_to_keep]
    Ys = Ys[elements_to_keep]
    sigmas = sigmas[elements_to_keep]

    return frequencies,Ys,sigmas

def get_all_data():

    H1L1_O1_freqs, H1L1_O1_Ys, H1L1_O1_sigmas = read_data('./../input/H1L1_O1.dat',f_max=1000)
    H1L1_O2_freqs, H1L1_O2_Ys, H1L1_O2_sigmas = read_data('./../input/H1L1_O2.dat',f_max=1000)
    H1L1_O3_freqs, H1L1_O3_Ys, H1L1_O3_sigmas = read_data('./../input/H1L1_O3.dat',f_max=1000)
    H1V1_O3_freqs, H1V1_O3_Ys, H1V1_O3_sigmas = read_data('./../input/H1V1_O3.dat',f_max=1000)
    L1V1_O3_freqs, L1V1_O3_Ys, L1V1_O3_sigmas = read_data('./../input/L1V1_O3.dat',f_max=1000)

    c = 1
    spectra_dict = {\
        'H1L1_O1':[H1L1_O1_freqs, c*H1L1_O1_Ys, c*H1L1_O1_sigmas],
        'H1L1_O2':[H1L1_O2_freqs, c*H1L1_O2_Ys, c*H1L1_O2_sigmas],
        'H1L1_O3':[H1L1_O3_freqs, c*H1L1_O3_Ys, c*H1L1_O3_sigmas],
        'H1V1_O3':[H1V1_O3_freqs, c*H1V1_O3_Ys, c*H1V1_O3_sigmas],
        'L1V1_O3':[L1V1_O3_freqs, c*L1V1_O3_Ys, c*L1V1_O3_sigmas]
        }

    return spectra_dict
