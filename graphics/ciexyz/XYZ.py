#!/usr/bin/env python
"""
http://www.fourmilab.ch/documents/specrend/

"""
import numpy as np
import matplotlib.pyplot as plt
import ciexyz.ciexyz as c   # see "ufunc-cd ciexyz"


class XYZ(object):
    def __init__(self, w=None):
        if w is None:
            w = np.linspace(380,780,401) 
        pass
        X = c.X(w) 
        Y = c.Y(w)
        Z = c.Z(w)
        Ysum = np.sum(Y)
 
        XYZ = np.empty((len(w),3))
        XYZ[:,0] = X
        XYZ[:,1] = Y
        XYZ[:,2] = Z
        XYZ /= Ysum    # color matching functions normalized by y integral 

        xyz = XYZ/np.repeat(np.sum(XYZ, axis=1),3).reshape(-1,3) # (X,Y,Z)/(X+Y+Z)

        self.w = w
        self.X = X
        self.Y = Y
        self.Z = Z
        self.Ysum = Ysum
        self.XYZ = XYZ  
        self.xyz = xyz


class Spectrum(object):
    def __init__(self, v, w=None,):
        if w is None:
            w = np.linspace(380,780,401) 
        self.xyz = XYZ(w) 
        if type(v) is np.ndarray: 
            s = v
        else:
            i = self.index(v)
            s = np.zeros_like(w)
            s[i] = 1
        pass
        self.s = s

    w = property(lambda self:self.xyz.w)

    def index(self, wl):
        return np.where(wl==self.w)[0][0] 

    def asXYZ(self):
        s = self.s
        X = np.sum(s*self.xyz.X) 
        Y = np.sum(s*self.xyz.Y) 
        Z = np.sum(s*self.xyz.Z) 
        return np.array([X,Y,Z])/self.xyz.Ysum




class BB(XYZ):
    names = ["d5k","d6k"]
    def __init__(self, w=None):
        XYZ.__init__(self, w)
        self.d5k = Spectrum(c.BB5K(self.w), self.w)   # blackbody spectrum
        self.d6k = Spectrum(c.BB6K(self.w), self.w)   # blackbody spectrum

    @classmethod
    def check(cls):
        bb = cls()
        for name in BB.names:
            s = getattr(bb,name)
            o = s.asXYZ()
            c = o/o.sum()
            print "%s XYZ %s xyz %s  " % (name, str(o), str(c))
        pass




class Mono(XYZ):
    def __init__(self, w=None):
        XYZ.__init__(self, w)

    @classmethod
    def plot(cls):
        mono = cls()

        plt.plot( mono.xyz[:,0], mono.xyz[:,1] )
 
        # label a slice of the wavelengths
        wls = mono.w[::20]
        xyn = mono.xyz[::20,:2]

        plt.scatter( xyn[:,0], xyn[:,1] )
        for i in range(len(wls)):
            plt.annotate(wls[i], xy = xyn[i], xytext = (0.5, 0.5), textcoords = 'offset points')

    @classmethod
    def check(cls):
        mono = cls()
        for wl in np.arange(450,630,10,dtype=np.float64):

            s = Spectrum(wl)
            o = s.asXYZ()
            c = o/o.sum()

            i = s.index(wl) 

            m = s.xyz.XYZ[i]
            m /= m.sum()

            n = s.xyz.xyz[i]

            assert np.allclose(n,c)
            assert np.allclose(n,m)

            print "wl %10.1f o %s c %s m %s n %s " % (wl, str(o), str(c), str(m), str(n))



if __name__ == '__main__':
    pass

    plt.ion()
    Mono.check()
    Mono.plot()

    plt.show()


