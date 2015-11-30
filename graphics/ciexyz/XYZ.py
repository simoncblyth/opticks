#!/usr/bin/env python
"""
http://www.fourmilab.ch/documents/specrend/

"""
import numpy as np
import matplotlib.pyplot as plt
import ciexyz.ciexyz as c   # see "ufunc-cd ciexyz"


class XYZ(object):
    """
    Wavelength range is not an input, its a feature of
    human vision and CIE weighting functions
    """
    w = np.arange(399.,675.+1.,1.)   # adjusted to avoid 0,0,0 in the matching functions 
    wls = np.array([400,460,470,480,490,500,520,540,560,580,600,620,675])

    @classmethod
    def index(cls, wl):
        return np.where(wl==cls.w)[0][0] 

    def __init__(self):
        w = self.w
        X = c.X(w) 
        Y = c.Y(w)
        Z = c.Z(w)
        Ysum = np.sum(Y)
 
        XYZ = np.empty((len(w),3))
        XYZ[:,0] = X
        XYZ[:,1] = Y
        XYZ[:,2] = Z
        #XYZ /= Ysum    # color matching functions normalized by y integral 

        xyz = XYZ/np.repeat(np.sum(XYZ, axis=1),3).reshape(-1,3) # (X,Y,Z)/(X+Y+Z)

        self.X = X
        self.Y = Y
        self.Z = Z
        self.Ysum = Ysum
        self.XYZ = XYZ  
        self.xyz = xyz


    @classmethod
    def plot(cls):
        mono = cls()

        plt.plot( mono.xyz[:,0], mono.xyz[:,1] )
 
        # dots every 5 nm
        i0 =XYZ.index(XYZ.wls[0])
        plt.scatter( mono.xyz[i0::5,0], mono.xyz[i0::5,1] )

        # label wavelengths for some, non-uniformly 
        for i in range(len(mono.wls)):
            wl = mono.wls[i]            
            ix = mono.index(wl)
            xy = mono.xyz[ix,:2]
            plt.annotate(wl, xy = xy, xytext = (0.5, 0.5), textcoords = 'offset points')




def planck(nm, K):
    #from scipy.constants import h,c,k
    h = 6.62606957e-34
    c = 299792458.0
    k = 1.3806488e-23

    wav = nm/1e9
    a = 2.0*h*c*c
    b = (h*c)/(k*K)/wav
    return a/(np.power(wav,5) * np.expm1(b))


class Spectrum(object):
    def __init__(self, s):
        assert type(s) is np.ndarray
        self.s = s
        self.xyz = XYZ() 

    def to_xyz(self):
        XYZ_ = self.to_XYZ()
        return XYZ_/XYZ_.sum()

    def to_XYZ(self):
        s = self.s
        X = np.sum(s*self.xyz.X) 
        Y = np.sum(s*self.xyz.Y) 
        Z = np.sum(s*self.xyz.Z) 
        return np.array([X,Y,Z])/self.xyz.Ysum

    w = property(lambda self:self.xyz.w)

    @classmethod
    def blackbody(cls, K):
        s = planck(XYZ.w, K)
        return cls(s)

    @classmethod
    def plot_blackbody(cls):
        for i,K in enumerate(np.arange(1000,7000,500)):
            s = Spectrum.blackbody(K) 
            xyz = s.to_xyz()
            plt.scatter(xyz[0], xyz[1], label=K)

            if K in [3000,5000,6000]:
                plt.annotate(K, xy = xyz[:2], xytext = (0.5, 0.5), textcoords = 'offset points')

    @classmethod
    def check_blackbody(cls):
        for K in np.arange(4000,7000,500):
            s = Spectrum.blackbody(K) 
            o = s.to_xyz()
            oo = s.to_xyz()

            print "K %10.1f o %s oo %s " % (K, str(o), str(oo))

            if K == 5000. or K == 6000:
                if K == 5000.:
                    sa = Spectrum(c.BB5K(XYZ.w))  
                else:
                    sa = Spectrum(c.BB6K(XYZ.w))  
                pass 
                ao = sa.to_XYZ()
                aoo = sa.to_xyz()
                print "alt ufunc spectrum %s %s %s " % ( K,str(ao), str(aoo))



    @classmethod
    def mono(cls, nm):
        s = np.zeros_like(XYZ.w)
        i = XYZ.index(nm)
        s[i] = 1
        return cls(s)

    @classmethod
    def check_mono(cls):
        for i,wl in enumerate(XYZ.w):

            j = XYZ.index(wl) 
            assert i == j

            s = Spectrum.mono(wl)
            o = s.to_XYZ()
            c = o/o.sum()

            m = s.xyz.XYZ[i]
            m /= m.sum()

            n = s.xyz.xyz[i]

            assert np.allclose(n,c)
            assert np.allclose(n,m)

            print "wl %10.1f o %s c %s m %s n %s " % (wl, str(o), str(c), str(m), str(n))






if __name__ == '__main__':
    pass

    plt.ion()
    XYZ.plot()
    #Spectrum.check_mono()
    Spectrum.check_blackbody()
    Spectrum.plot_blackbody()


    plt.show()


