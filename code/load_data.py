import numpy as np

def read_data(filename,f_max=400,trim_nans=True):

    """
    Helper function to load cross-correlation data from file.

    Parameters
    ----------
    filename : `str`
        Target filename
    f_max : `float`
        Maximum frequency to consider (default 400 Hz)
    trim_nans : `bool`
        If True, will remove frequencies that have been notched by data quality flags (default True)

    """

    # Load 
    frequencies,Ys,sigmas = np.loadtxt(filename,unpack=True,skiprows=1)

    # Limit to low and unnotched frequencies
    if trim_nans:
        elements_to_keep = (~np.isinf(sigmas))
        frequencies = frequencies[elements_to_keep]
        Ys = Ys[elements_to_keep]
        sigmas = sigmas[elements_to_keep]

    return frequencies,Ys,sigmas

def get_all_data(f_max=1000,trim_nans=True):

    H1L1_O1_freqs, H1L1_O1_Ys, H1L1_O1_sigmas = read_data('./../input/H1L1_O1.dat',f_max=f_max,trim_nans=trim_nans)
    H1L1_O2_freqs, H1L1_O2_Ys, H1L1_O2_sigmas = read_data('./../input/H1L1_O2.dat',f_max=f_max,trim_nans=trim_nans)
    H1L1_O3_freqs, H1L1_O3_Ys, H1L1_O3_sigmas = read_data('./../input/H1L1_O3.dat',f_max=f_max,trim_nans=trim_nans)
    H1V1_O3_freqs, H1V1_O3_Ys, H1V1_O3_sigmas = read_data('./../input/H1V1_O3.dat',f_max=f_max,trim_nans=trim_nans)
    L1V1_O3_freqs, L1V1_O3_Ys, L1V1_O3_sigmas = read_data('./../input/L1V1_O3.dat',f_max=f_max,trim_nans=trim_nans)

    spectra_dict = {\
        'H1L1_O1':[H1L1_O1_freqs, H1L1_O1_Ys, H1L1_O1_sigmas],
        'H1L1_O2':[H1L1_O2_freqs, H1L1_O2_Ys, H1L1_O2_sigmas],
        'H1L1_O3':[H1L1_O3_freqs, H1L1_O3_Ys, H1L1_O3_sigmas],
        'H1V1_O3':[H1V1_O3_freqs, H1V1_O3_Ys, H1V1_O3_sigmas],
        'L1V1_O3':[L1V1_O3_freqs, L1V1_O3_Ys, L1V1_O3_sigmas]
        }

    return spectra_dict
