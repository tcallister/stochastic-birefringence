import numpy as np
import constants as const
from scipy.special import spherical_jn

# Hanford
# http://www.ligo.org/scientists/GW100916/detectors.txt
# See also LIGO DCC T980044
# Also https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lal/lib/tools/LALDetectors.h
H1x = np.array([-2.16141492636e6,-3.83469517889e6,4.60035022664e6])
H1u = np.array([-0.22389266154,0.79983062746,0.55690487831])
H1v = np.array([-0.91397818574,0.02609403989,-0.40492342125])

# Livingston
L1x = np.array([-7.42760447238e4,-5.49628371971e6,3.22425701744e6])
L1u = np.array([-0.95457412153,-0.14158077340,-0.26218911324])
L1v = np.array([0.29774156894,-0.48791033647,-0.82054461286])

# Virgo
V1x = np.array([4.54637409900e6,8.42989697626e5,4.37857696241e6])
V1u = np.array([-0.70045821479,0.20848948619,0.68256166277])
V1v = np.array([-0.05379255368,-0.96908180549,0.24080451708])

class Detector():

    """
    Class representing a single detector object

    Parameters
    ----------
    position : array
        Position vector of the detector's vertex
    xArm : array
        Vector parallel to the detector's `x` arm
    yArm : array
        Vector parallel to the detector's `y` arm
    
    Attributes
    ----------
    x : array
        Vector denoting the cartesian coordinates of the detector vertex
    u : array
        Unit vector parallel to the detector's 'x' arm
    v : array
        Unit vector parallel to the detector's 'y' arm
    """

    def __init__(self,position,xArm,yArm):

        self.x = position
        self.u = xArm/np.linalg.norm(xArm)
        self.v = yArm/np.linalg.norm(yArm)

    def pos(self):
        """
        Function returning detector's Cartesian coordinates
        Returns
        -------
        x : array
            Vector denoting detector position (units of meters)
        """
        return self.x

    def xArm(self):
        """
        Function returning detector's `x` arm direction
        Returns
        -------
        u : array
            Unit vector parallel to `x` arm
        """
        return self.u

    def yArm(self):
        """
        Function returning detector's `y` arm direction
        Returns
        -------
        v : array
            Unit vector parallel to `y` arm
        """
        return self.v

    def D(self):
        """
        Function that computes detector's response tensor
        Returns
        -------
        d : array (3x3)
            Response tensor
        """
        d = 0.5*(np.outer(self.u,self.u)-np.outer(self.v,self.v))
        return d

    def Fp(self,theta,phi):
        """
        Function computing antenna response pattern to plus-polarized signals
        
        Parameters
        ----------
        theta : float
            Polar angle defining target sky direction
        phi : float
            Azimuthal angle defining target sky direction

        Returns
        -------
        Fp : float
            Antenna response to the specified direction
        """
        return np.tensordot(self.D(),ep(theta,phi),axes=2)

    def Fc(self,theta,phi):
        """
        Function computing antenna response pattern to cross-polarized signals
        
        Parameters
        ----------
        theta : float
            Polar angle defining target sky direction
        phi : float
            Azimuthal angle defining target sky direction

        Returns
        -------
        Fc : float
            Antenna response to the specified direction
        """
        return np.tensordot(self.D(),ec(theta,phi),axes=2)

    @classmethod
    def H1(cls):
        """
        Creates detector consistent with LIGO-Hanford geometry
        Returns
        -------
        Detector object
        """
        return cls(H1x,H1u,H1v)

    @classmethod
    def L1(cls):
        """
        Creates detector consistent with LIGO-Livingston geometry
        Returns
        -------
        Detector object
        """
        return cls(L1x,L1u,L1v)

    @classmethod
    def V1(cls):
        """
        Creates detector consistent with Virgo geometry
        Returns
        -------
        Detector object
        """
        return cls(V1x,V1u,V1v)

class Baseline():

    """
    Class representing a two-detector baseline

    Parameters
    ----------
    Detector1 : Detector
       Detector object comprising the first instrument in our baseline 
    Detector2 : Detector
       Detector object comprising the second instrument in our baseline 
    """

    def __init__(self,Detector1,Detector2):

        self.D1 = Detector1
        self.D2 = Detector2

    def _get_baseline_angles(self):

        """
        Helper function computing angles characterizing the baseline and detector
        orientations, for use in computing overlap reduction functions, following
        Sect. 3 of https://arxiv.org/pdf/0707.0535.pdf

        Returns
        -------
        beta : float
            Angular separation of detectors as viewed from Earth's center
        sigma1 : float
            Azimuthal rotation of detector 1 relative to great circle connecting instruments
        sigma2 : float
            Azimuthal rotation of detector 2 relative to great circle connecting instruments
        """

        # Define unit vectors normal to the Earth's surface at the location of each instrument
        r1 = self.D1.pos()/np.sqrt(self.D1.pos()@self.D1.pos())
        r2 = self.D2.pos()/np.sqrt(self.D2.pos()@self.D2.pos())
        
        # Separation angle between instruments
        beta = np.arccos(r1@r2)

        # Next, we want to define an orthonormal frame at the location of each detector,
        # leveraging the above normal vectors and the great circle connecting both instruments
        # First, define a unit vector orthogonal to r1, r2, and the great circle
        r1_cross_r2 = np.cross(r1,r2)
        r1_cross_r2 /= np.sqrt(r1_cross_r2@r1_cross_r2)

        # Now complete orthonormal triads by computing unit vectors tangent to the great circle at each location
        r1_gc = np.cross(r1_cross_r2,r1)
        r2_gc = np.cross(r1_cross_r2,r2)

        # Define unit vector along the bisector of each detector
        self.D1_bisector = (self.D1.xArm()+self.D1.yArm())/np.sqrt(2)
        self.D2_bisector = (self.D2.xArm()+self.D2.yArm())/np.sqrt(2)

        # Project bisectors onto triad vectors on the Earth's surface and compute rotation angle
        # relative to the great circle joining instruments
        self.D1_bisector_a = self.D1_bisector@r1_gc
        self.D1_bisector_b = self.D1_bisector@r1_cross_r2
        sig1 = np.arctan2(self.D1_bisector_b,self.D1_bisector_a)

        # Repeat with second instrument
        self.D2_bisector_a = self.D2_bisector@r2_gc
        self.D2_bisector_b = self.D2_bisector@r1_cross_r2
        sig2 = np.arctan2(self.D2_bisector_b,self.D2_bisector_a)

        return beta,sig1,sig2

    def stokes_overlap_reduction_functions(self,frequencies):

        """
        Function to compute overlap reduction functions for Stokes I and V parameters
  
        Parameters
        ----------
        frequencies : array
            Set of frequencies at which we want to evaluate ORFs

        Returns
        -------
        GammaI : array
            Overlap reduction function for Stokes I at desired frequencies
        GammaV : array
            Overlap reduction function for Stokes V at desired frequencies
        """

        # Get angles
        beta,sig1,sig2 = self._get_baseline_angles()
        Delta = (sig1+sig2)/2.
        delta = (sig1-sig2)/2.

        # Evaluate necessary bessel functions
        bessel_arg = 4.*np.pi*frequencies*const.R_earth/const.c*np.sin(beta/2)
        j0 = spherical_jn(0,bessel_arg)
        j1 = spherical_jn(1,bessel_arg)
        j2 = spherical_jn(2,bessel_arg)
        j3 = spherical_jn(3,bessel_arg)
        j4 = spherical_jn(4,bessel_arg)

        # Compute Eqs. 23, 24, and 27 of https://arxiv.org/pdf/0801.4185.pdf
        Theta1 = np.cos(beta/2)**4*(j0 + 5.*j2/7. + 3.*j4/112.)
        Theta2 = ((-3./8.)*j0 + (45./56.)*j2 - (169./896.)*j4) \
                    + (j0/2. - (5./7.)*j2 - (27./224.)*j4)*np.cos(beta) \
                    + (-j0/8. - (5./56.)*j2 - (3./896.)*j4)*np.cos(2.*beta)
        Theta3 = -np.sin(beta/2.)*((-j1 + (7./8.)*j3) \
                    + (j1 + (3./8.)*j3)*np.cos(beta))

        # Combine and return ORFs
        GammaI = Theta1*np.cos(4*delta) + Theta2*np.cos(4*Delta)
        GammaV = Theta3*np.sin(4.*Delta)

        return GammaI,GammaV

    def circular_overlap_reduction_functions(self,frequencies):

        """
        Function to compute overlap reduction functions for R and L circular polarizations
  
        Parameters
        ----------
        frequencies : array
            Set of frequencies at which we want to evaluate ORFs

        Returns
        -------
        GammaR : array
            Overlap reduction function for right-circular polarization at desired frequencies
        GammaL : array
            Overlap reduction function for left-circular polarization at desired frequencies
        """

        GammaI,GammaV = self.stokes_overlap_reduction_functions(frequencies)
        GammaR = GammaI + GammaV
        GammaL = GammaI - GammaV
        return GammaR,GammaL

def Omega(theta,phi):
    """
    Unit vector along the direction of a GW's propagation.
    Vectors `Omega`, `m`, and `n` form an orthonormal triad.
    Parameters
    ----------
    theta : float
        Polar angle from which plane wave originates (range 0-pi)
    phi : float
        Azimuthal angle of wave (range 0-2pi)
    Returns
    -------
    Unit vector (shape (3,) array) along direction of propagation
    """
    return -np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])

def m(theta,phi):
    """
    Unit vector in the plane transverse to a GW's direction of propagation.
    Vectors `Omega`, `m`, and `n` form an orthonormal triad.
    Parameters
    ----------
    theta : float
        Polar angle from which plane wave originates (range 0-pi)
    phi : float
        Azimuthal angle of wave (range 0-2pi)
    Returns
    -------
    Unit vector (shape (3,) array) transverse to direction of propagation.
    """
    return np.array([np.sin(phi),-np.cos(phi),0.])

def n(theta,phi):
    """
    Unit vector in the plane transverse to a GW's direction of propagation.
    Vectors `Omega`, `m`, and `n` form an orthonormal triad.
    Parameters
    ----------
    theta : float
        Polar angle from which plane wave originates (range 0-pi)
    phi : float
        Azimuthal angle of wave (range 0-2pi)
    Returns
    -------
    Unit vector (shape (3,) array) transverse to direction of propagation.
    """
    return np.array([np.cos(theta)*np.cos(phi),np.cos(theta)*np.sin(phi),-np.sin(theta)])

def ep(theta,phi):
    """
    Computes the plus-polarization basis tensor for a GW from a particular sky location.
    Parameters
    ----------
    theta : float
        Polar angle from which plane wave originates (range 0-pi)
    phi : float
        Azimuthal angle of wave (range 0-2pi)
    Returns
    -------
    Basis tensor (shape (3,3) array)
    """
    m = np.array([-np.sin(phi),np.cos(phi),0.])
    n = np.array([np.cos(theta)*np.cos(phi),np.cos(theta)*np.sin(phi),-np.sin(theta)])
    return np.outer(m,m)-np.outer(n,n)

def ec(theta,phi):
    """
    Computes the cross-polarization basis tensor for a GW from a particular sky location.
    Parameters
    ----------
    theta : float
        Polar angle from which plane wave originates (range 0-pi)
    phi : float
        Azimuthal angle of wave (range 0-2pi)
    Returns
    -------
    Basis tensor (shape (3,3) array)
    """
    m = np.array([-np.sin(phi),np.cos(phi),0.])
    n = np.array([np.cos(theta)*np.cos(phi),np.cos(theta)*np.sin(phi),-np.sin(theta)])
    return np.outer(m,n)+np.outer(n,m)

