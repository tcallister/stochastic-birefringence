import numpy as np

# Hanford
# http://www.ligo.org/scientists/GW100916/detectors.txt
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
        return np.tensordot(self.D(),ep(theta,phi),axes=2)

    def Fc(self,theta,phi):
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



if __name__=="__main__":

    print("Hanford:")
    hanford = Detector.H1()
    print(hanford.pos())
    print(hanford.xArm())
    print(hanford.yArm())

