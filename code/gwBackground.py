from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import time
import sys
import os
import matplotlib.pyplot as plt
from constants import *
from scipy.special import erf
from astropy.cosmology import Planck18
import astropy.units as u
codeDir = os.path.dirname(os.path.realpath(__file__))

def v(Mtot,f):

    """
    Helper function computing the PN expansion parameter (pi*M*f)**(1/3) and returning a tuple of its first three powers.

    Parameters
    ----------
    Mtot : float
        Total binary mass in units of solar masses
    f : float or np.array
        Frequency or frequencies at which to evaluate

    Returns
    -------
    pn_params : np.array
        Array containing the first, second, and third powers of the PN expansion parameter
    """

    return np.array([(np.pi*Mtot*MsunToSec*f)**(1./3.), (np.pi*Mtot*MsunToSec*f)**(2./3.), (np.pi*Mtot*MsunToSec*f)])

def dEdf(Mtot,freqs,eta=0.25,PN=True):

    """
    Function to compute the energy spectrum radiated by a CBC
    
    Parameters
    ----------
    Mtot : float
        Total mass in units of Msun
    freqs : np.array
        Array of frequencies at which we want to evaluate dEdf
    eta : float
        Reduced mass ratio. Defaults to 0.25 (equal mass)
    PN: bool
        If True, will include PN corrections to radiated energy spectrum (Default True) 

    Returns
    -------
    dEdf : np.array
        Energy spectrum in units of J/Hz
    """

    # Fix spin magnitudes to zero for convenience
    chi = 0.

    # Initialize energy density
    dEdf_spectrum = np.zeros(freqs.shape)

    if PN:

        # Waveform model from Ajith+ 2011 (10.1103/PhysRevLett.106.241101)
        
        # PN corrections to break frequencies bounding different waveform regimes
        # See Eq. 2 and Table 1
        eta_arr = np.array([eta,eta*eta,eta*eta*eta])
        chi_arr = np.array([1,chi,chi*chi]).T
        fM_corrections = np.array([[0.6437,0.827,-0.2706],[-0.05822,-3.935,0.],[-7.092,0.,0.]])
        fR_corrections = np.array([[0.1469,-0.1228,-0.02609],[-0.0249,0.1701,0.],[2.325,0.,0.]])
        fC_corrections = np.array([[-0.1331,-0.08172,0.1451],[-0.2714,0.1279,0.],[4.922,0.,0.]])
        sig_corrections = np.array([[-0.4098,-0.03523,0.1008],[1.829,-0.02017,0.],[-2.87,0.,0.]])

        # Define frequencies
        # See Eq. 2 and Table 1
        fMerge = (1. - 4.455*(1.-chi)**0.217 + 3.521*(1.-chi)**0.26 + eta_arr.dot(fM_corrections).dot(chi_arr))/(np.pi*Mtot*MsunToSec)
        fRing = (0.5 - 0.315*(1.-chi)**0.3 + eta_arr.dot(fR_corrections).dot(chi_arr))/(np.pi*Mtot*MsunToSec)
        fCut = (0.3236 + 0.04894*chi + 0.01346*chi*chi + eta_arr.dot(fC_corrections).dot(chi_arr))/(np.pi*Mtot*MsunToSec)
        sigma = (0.25*(1.-chi)**0.45 - 0.1575*(1.-chi)**0.75 + eta_arr.dot(sig_corrections).dot(chi_arr))/(np.pi*Mtot*MsunToSec)

        # Identify piecewise components
        inspiral = freqs<fMerge
        merger = (freqs>=fMerge)*(freqs<fRing)
        ringdown = (freqs>=fRing)*(freqs<fCut)

        # Define PN amplitude corrections
        # See Eq. 1 and following text
        alpha = np.array([0., -323./224. + 451.*eta/168., (27./8.-11.*eta/6.)*chi])
        eps = np.array([1.4547*chi-1.8897, -1.8153*chi+1.6557, 0.])
        vs = v(Mtot,freqs)

        # Compute multiplicative scale factors to enforce continuity of dEdf across boundaries
        # Note that w_m and w_r are the ratios (inspiral/merger) and (merger/ringdown), as defined below
        v_m = v(Mtot,fMerge)
        v_r = v(Mtot,fRing)
        w_m = np.power(fMerge,-1./3.)*np.power(1.+alpha.dot(v_m),2.)/(np.power(fMerge,2./3.)*np.power(1.+eps.dot(v_m),2.)/fMerge)
        w_r = (w_m*np.power(fRing,2./3.)*np.power(1.+eps.dot(v_r),2.)/fMerge)/(np.square(fRing)/(fMerge*fRing**(4./3.)))

        # Energy spectrum
        dEdf_spectrum[inspiral] = np.power(freqs[inspiral],-1./3.)*np.power(1.+alpha.dot(vs[:,inspiral]),2.)
        dEdf_spectrum[merger] = w_m*np.power(freqs[merger],2./3.)*np.power(1.+eps.dot(vs[:,merger]),2.)/fMerge
        dEdf_spectrum[ringdown] = w_r*np.square(freqs[ringdown]/(1.+np.square((freqs[ringdown]-fRing)/(sigma/2.))))/(fMerge*fRing**(4./3.))

    else:

        # Waveform model from Ajith+ 2008 (10.1103/PhysRevD.77.104017)
        # Define IMR parameters
        # See Eq. 4.19 and Table 1
        fMerge = (0.29740*eta**2. + 0.044810*eta + 0.095560)/(np.pi*Mtot*MsunToSec)
        fRing = (0.59411*eta**2. + 0.089794*eta + 0.19111)/(np.pi*Mtot*MsunToSec)
        fCut = (0.84845*eta**2. + 0.12828*eta + 0.27299)/(np.pi*Mtot*MsunToSec)
        sigma = (0.50801*eta**2. + 0.077515*eta + 0.022369)/(np.pi*Mtot*MsunToSec)

        # Identify piecewise components
        inspiral = freqs<fMerge
        merger = (freqs>=fMerge)*(freqs<fRing)
        ringdown = (freqs>=fRing)*(freqs<fCut)

        # Energy spectrum
        dEdf_spectrum[inspiral] = np.power(freqs[inspiral],-1./3.)
        dEdf_spectrum[merger] = np.power(freqs[merger],2./3.)/fMerge
        dEdf_spectrum[ringdown] = np.square(freqs[ringdown]/(1.+np.square((freqs[ringdown]-fRing)/(sigma/2.))))/(fMerge*fRing**(4./3.))

    # Normalization
    Mc = np.power(eta,3./5.)*Mtot*Msun
    amp = np.power(G*np.pi,2./3.)*np.power(Mc,5./3.)/3.

    return amp*dEdf_spectrum

class OmegaGW(object):

    """
    Base class used to compute the stochastic energy density of a CBC population.
    To make the evaluation as fast as possible, we'll implement integration over redshift and mass distribution
    via array multiplication. Different mass distributions are imposed by specifying probability weights in mass space.
    """

    def __init__(self,ref_mMin,ref_mMax,ref_zs,fmax,Mtots=[],qs=[],gridSize=(70,65)):

        """
        Initializes class by setting up a grid of masses and mass ratios, and evaluating the radiated energy spectrum dE/df
        across this grid. This allows us to precompute dE/df, only evaluating it *once*. When computing a *population-averaged*
        energy spectrum, we will take a weighted sum across this grid. Weights are imposed by redefining `self.probs`, implemented
        in child classes via the `setProbs()` function.

        Parameters
        ----------
        ref_mMin: float
            Minimum component mass to consider in mass grid
        ref_mMax: float
            Maximum component mass to consider in mass grid
        ref_zs: np.array
            Redshift array across which we will integrate to compute Omega(f)
        fMax: float
            Maximum detector-frame frequency to consider
        """

        # Save reference grid of redshifts
        self.ref_zs = ref_zs
        self.comoving_distances = Planck18.comoving_distance(self.ref_zs).to(u.Gpc).value

        # Initialize grid of masses over which we'll compute dE/df
        # In particular, we'll grid log-uniformly in total mass and uniformly in mass ratio q
        self.ref_mMin = ref_mMin
        self.ref_mMax = ref_mMax

        if len(Mtots)!=0:
            self.ref_Mtots = Mtots
        else:
            self.ref_Mtots = np.logspace(np.log10(2.*self.ref_mMin),np.log10(2.*self.ref_mMax),gridSize[0])

        if len(qs)!=0:
            self.ref_qs = qs
        else:
            qMin = ref_mMin/ref_mMax #max(0.05,ref_mMin/ref_mMax)
            self.ref_qs = np.linspace(qMin,1,gridSize[1])

        # Grid
        self.Mtots_2d,self.qs_2d = np.meshgrid(self.ref_Mtots,self.ref_qs)

        # Compute the component masses and reduced mass ratios at each grid point
        self.m1s_2d = self.Mtots_2d/(1.+self.qs_2d)
        self.m2s_2d = self.qs_2d*self.Mtots_2d/(1.+self.qs_2d)
        self.ref_etas = self.ref_qs/(1.+self.ref_qs)/(1.+self.ref_qs)

        # Array of frequencies at which dE/df will be evaluated for each point in our mass grid
        self.ref_freqs = np.logspace(np.log10(10),np.log10(fmax),250)

        # Remember that we need to consider binaries placed across a *range* of redshifts.
        # For a given system mass, we need to know not just dE/df(f), but the *redshifted* spectrum dE/df(f*(1+z))
        # at both difference frequencies f and merger redshifts z

        # Set up a 2D array of source-frame frequencies f(1+z)
        # i.e. self.ref_redshiftedFreqs[i,j] corresponds to self.ref_freqs[j]*(1.+self.ref_zs[i])
        self.ref_redshiftedFreqs = np.array([self.ref_freqs*(1.+z) for z in self.ref_zs])

        # Now evaluate the energy spectrum at each of these source-frame frequencies, for every point in our mass grid
        # self.ref_energySpectra[i,j,k,:] is the energy contribution from a CBC with reduced mass ratio self.ref_etas[i],
        # total mass self.ref_Mtots[j], and at redshift self.ref_zs[k].
        self.ref_energySpectra = np.array([[dEdf(M,self.ref_redshiftedFreqs,eta=eta) for M in self.ref_Mtots] for eta in self.ref_etas])

        # Initialize weights for mass grid
        self.probs = np.ones(self.Mtots_2d.shape)
        self.probs /= np.sum(self.probs)

    def amplification(self,kappa_d,kappa_z):

        """
        Helper function to compute exponential factor by which GW energies are birefringently amplified

        Parameters
        ----------
        kappa_d : float
            Coefficient determining degree of amplitude birefringence with comoving distance
        kappa_z : float
            Coefficient determining degree of amplitude birefringence with redshift

        Returns
        -------
        cosh_amplification_factor : np.array
            2D array of cosh amplification factors modifying the energy radiated at detector-frame frequencies `self.ref_freqs` from sources at `self.ref_zs`; used to compute Stokes I background
        sinh_amplification_factor : np.array
            2D array of sinh amplification factors modifying the energy radiated at detector-frame frequencies `self.ref_freqs` from sources at `self.ref_zs`; used to compute Stokes V background
        """

        # Compute argument of amplification exponentials
        amplification_factor = 2.*jnp.pi*(kappa_d*self.comoving_distances+kappa_z*self.ref_zs)[jnp.newaxis,:]*(self.ref_freqs[:,jnp.newaxis]/100.)

        # Get cosh and sinh of this factor, for Stokes I and V respectively
        cosh_amplification_factor = jnp.cosh(amplification_factor)
        sinh_amplification_factor = jnp.sinh(amplification_factor)

        return cosh_amplification_factor,sinh_amplification_factor

    def eval(self,R0,dRdV,targetFreqs,kappa_d,kappa_z):

        """
        Given a prescription for the local merger rate and its evolution over redshift, compute Omega(f)

        Parameters
        ----------
        R0 : float
            Local merger rate density in units Gpc^{-3} yr^{-1}
        dRdV : np.array
            Arbitrarily normalized merger rate density as a function of redshift. Should be defined at the same redshifts specified in `self.ref_zs`
        targetFreqs : np.array
            Array of frequencies at which we want Omega(f). Must be above 10 and below the `fmax` used to initialize the object
        kappa_d : float
            Coefficient governing birefringence with respect to comoving distance
        kappa_z : float
            Coefficient governing birefringence with respect to redshift

        Returns
        -------
        final_Omg_I_spectrum : np.array
            Array containing Stokes I energy density spectrum
        final_Omg_V_spectrum : np.array
            Array containing Stokes V energy density spectrum
        """

        # Convert to number per Mpc^3 per sec and normalize merger rate density
        R0 = R0/1e9/year
        dRdV_norm = R0*dRdV/dRdV[0]

        # Compute weighted average of energy-density spectrum, integrated over total mass and mass ratio space
        # The result is a 2D array, with dedf[i,j] the population-averaged energy contributed at detector-frame
        # frequency i by binaries at redshift j
        dedf = jnp.tensordot(self.probs,self.ref_energySpectra,axes=2).T
        dedf_alt = np.tensordot(self.probs,self.ref_energySpectra,axes=2).T

        # Birefringently amplify
        cosh_amp,sinh_amp = self.amplification(kappa_d,kappa_z)
        dedf_I = dedf*cosh_amp
        dedf_V = dedf*sinh_amp
        
        # Redshift integrand
        R_invE = dRdV_norm/jnp.sqrt(OmgM*(1.+self.ref_zs)**3.+OmgL)/(1.+self.ref_zs)

        # Integrate over redshifts via a dot product between dedf and the redshift-dependent R_invE
        dz = self.ref_zs[1]-self.ref_zs[0]
        Omg_I_spectrum = (self.ref_freqs/rhoC/H0)*dedf_I.dot(R_invE)*dz
        Omg_V_spectrum = (self.ref_freqs/rhoC/H0)*dedf_V.dot(R_invE)*dz

        # Interpolate onto desired frequencies and return
        final_Omg_I_spectrum = jnp.interp(targetFreqs,self.ref_freqs,Omg_I_spectrum,left=0.,right=0.)
        final_Omg_V_spectrum = jnp.interp(targetFreqs,self.ref_freqs,Omg_V_spectrum,left=0.,right=0.)

        return final_Omg_I_spectrum,final_Omg_V_spectrum

class OmegaGW_BBH(OmegaGW):

    """
    Subclass of `OmegaGW`, used to compute energy density due to BBHs under a "Power-Law + Peak" mass model

    Implements a function `setProbs` to fix the weights used in integrating over the mass grid to compute
    a population averaged energy spectrum
    """

    def __init__(self,ref_mMin,ref_mMax,ref_zs,Mtots=[],qs=[],gridSize=(70,65)):
        super(OmegaGW_BBH,self).__init__(ref_mMin,ref_mMax,ref_zs,3000,Mtots=Mtots,qs=qs,gridSize=gridSize)

    def setProbs_plPeak(self,mMin,mMax,dmMin,dmMax,lmbda,mu_peak,sig_peak,frac_peak,bq):

        """
        Function to set mass-dependent weights over precomputed energy-density spectra in order
        to implement and integrate over a realistic black hole mass distribution.

        Parameters
        ----------
        mMin : float
            Mass below which the black hole primary mass distribution is truncated to zero
        mMax : float
            Mass above which the black hole primary mass distribution is truncated to zero
        dmMin : float
            Scale of the exponential truncation applied below `mMin`
        dmMax : float
            Scale of the exponential truncation applied above `mMax`
        lmbda : float
            Power-law exponent describing the bulk of the primary mass distribution
        mu_peak : float
            The location of a Gaussian excess in the primary mass distribution
        sig_peak : float
            Standard deviation of the gaussian excess
        frac_peak : float
            The mixture fraction between power law and Gaussian components of the primary mass distribution.
            (Note that the exponential truncations make this mixture fraction only approximate)
        bq : float
            Power-law exponent governing mass ratio distribution
        """
        
        # Jacobian with which to convert integration over d(lnM)dq to d(m1)d(m2)
        probs_jacobian = self.Mtots_2d**2./(1.+self.qs_2d)**2.

        # Power law component in m1
        p_m1_pl = (1.+lmbda)*self.m1s_2d**lmbda/(100.**(1.+lmbda) - 2.**(1.+lmbda))

        # Gaussian component in m1
        p_m1_peak = np.exp(-(self.m1s_2d-mu_peak)**2/(2.*sig_peak**2))/np.sqrt(2.*np.pi*sig_peak**2)

        # Compute low- and high-mass filters
        low_filter = np.exp(-(self.m1s_2d-mMin)**2/(2.*dmMin**2))
        low_filter = np.where(self.m1s_2d<mMin,low_filter,1.)
        high_filter = np.exp(-(self.m1s_2d-mMax)**2/(2.*dmMax**2))
        high_filter = np.where(self.m1s_2d>mMax,high_filter,1.)

        # Apply filters to combined power-law and peak
        probs_m1 = (frac_peak*p_m1_peak + (1.-frac_peak)*p_m1_pl)*low_filter*high_filter
        probs_m1[self.m1s_2d>=100.] = 0.

        # Probability on secondary mass
        probs_m2 = (1.+bq)*np.power(self.m2s_2d,bq)/(np.power(self.m1s_2d,1.+bq)-2.**(1.+bq))  
        probs_m2[self.m2s_2d<=2.] = 0.

        # Combine and set
        probs = probs_jacobian*probs_m1*probs_m2
        probs /= np.sum(probs)
        self.probs = probs

