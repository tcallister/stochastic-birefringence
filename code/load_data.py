import numpy as np

def read_data(filename,trim_nans=True):

    """
    Helper function to load cross-correlation data from file.

    Parameters
    ----------
    filename : str
        Target filename
    trim_nans : bool
        If True, will remove frequencies that have been notched by data quality flags (default True)

    Returns
    -------
    frequencies : np.array

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

def get_all_data(trim_nans=True):

    """
    Helper function used by `run_birefringence_variable_evolution.py` to load data.

    Parameters
    ----------
    trim_nans : bool
        If True, will remove frequencies that have been notched by data quality flags (default True)

    Returns
    -------
    spectra_dict : dict
        Dictionary containing frequencies, cross-correlation measurements, and uncertainty spectra for all baselines and observing runs
    """

    H1L1_O1_freqs, H1L1_O1_Ys, H1L1_O1_sigmas = read_data('./../input/H1L1_O1.dat',trim_nans=trim_nans)
    H1L1_O2_freqs, H1L1_O2_Ys, H1L1_O2_sigmas = read_data('./../input/H1L1_O2.dat',trim_nans=trim_nans)
    H1L1_O3_freqs, H1L1_O3_Ys, H1L1_O3_sigmas = read_data('./../input/H1L1_O3.dat',trim_nans=trim_nans)
    H1V1_O3_freqs, H1V1_O3_Ys, H1V1_O3_sigmas = read_data('./../input/H1V1_O3.dat',trim_nans=trim_nans)
    L1V1_O3_freqs, L1V1_O3_Ys, L1V1_O3_sigmas = read_data('./../input/L1V1_O3.dat',trim_nans=trim_nans)

    spectra_dict = {\
        'H1L1_O1':[H1L1_O1_freqs, H1L1_O1_Ys, H1L1_O1_sigmas],
        'H1L1_O2':[H1L1_O2_freqs, H1L1_O2_Ys, H1L1_O2_sigmas],
        'H1L1_O3':[H1L1_O3_freqs, H1L1_O3_Ys, H1L1_O3_sigmas],
        'H1V1_O3':[H1V1_O3_freqs, H1V1_O3_Ys, H1V1_O3_sigmas],
        'L1V1_O3':[L1V1_O3_freqs, L1V1_O3_Ys, L1V1_O3_sigmas]
        }

    return spectra_dict
