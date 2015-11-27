#!/usr/bin/env python

import os, logging
import numpy as np
from env.python.utils import *
from env.numerics.npy.types import *
import env.numerics.npy.PropLib as PropLib 
log = logging.getLogger(__name__)

X,Y,Z,W = 0,1,2,3

def theta(xyz):
    """
    :param xyz: array of cartesian coordinates
    :return: array of Spherical Coordinare polar angle, theta in degrees

    First subtract off any needed translations to align
    the coordinate system with focal points.

    Spherical Coordinates (where theta is polar angle 0:pi, phi is azimuthal 0:2pi)

     x = r sin(th) cos(ph)  = r st cp   
     y = r sin(th) sin(ph)  = r st sp  
     z = r cos(th)          = r ct     


     sqrt(x*x + y*y) = r sin(th)
                  z  = r cos(th)

     atan( sqrt(x*x+y*y) / z ) = th 

    """
    r = np.linalg.norm(xyz, ord=2, axis=1)
    z = xyz[:,2]
    th = np.arccos(z/r)*180./np.pi
    return th

costheta_ = lambda a,b:np.sum(a * b, axis = 1)/(np.linalg.norm(a, 2, 1)*np.linalg.norm(b, 2, 1)) 
ntile_ = lambda vec,N:np.tile(vec, N).reshape(-1, len(vec))


def scatter3d(fig,  xyz): 
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2])

def histo(fig,  vals): 
    ax = fig.add_subplot(111)
    ax.hist(vals, bins=91,range=[0,90])

def xyz3d(fig, path):
    xyz = np.load(path).reshape(-1,3)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2])




class Rat(object):
    def __init__(self, n, d, label=""):
        n = len(n)
        d = len(d)
        r = float(n)/float(d)
        self.n = n
        self.d = d
        self.r = r
        self.label = label 
    def __repr__(self):
        return "Rat %s %s/%s %5.3f " % (self.label,self.n, self.d, self.r)


class Evt(object):
    def __init__(self, tag="1", src="torch", det="dayabay", label=""):
        self.tag = str(tag)
        self.label = label
        self.src = src
        self.ox = load_("ox"+src,tag,det) 
        self.rx = load_("rx"+src,tag,det) 

        self.ph = load_("ph"+src,tag,det)
        self.seqhis = self.ph[:,0,0]
        self.seqmat = self.ph[:,0,1]

        # photon level qtys, 
        # ending values as opposed to the compressed step by step records
        self.wavelength = self.ox[:,2,W] 
        self.post = self.ox[:,0] 

        self.flags = self.ox.view(np.uint32)[:,3,3]
        self.fdom = np.load(idp_("OPropagatorF.npy"))

    def msize(self):
        return float(self.ox.shape[0])/1e6

    def __repr__(self):
        return "Evt(%s,\"%s\",\"%s\")" % (self.tag, self.src, self.label)

    def history_table(self):
        seqhis = self.seqhis
        cu = count_unique(seqhis)
        seqhis_table(cu)
        tot = cu[:,1].astype(np.int32).sum()
        print "tot:", tot
        return cu

    def flags_table(self):
        flags = self.flags
        cu = count_unique(flags)
        gflags_table(cu)
        tot = cu[:,1].astype(np.int32).sum()
        print "tot:", tot
        brsa = maskflags_int("BR|SA|TORCH")
        return cu

    def seqhis_or(self, *args):
        """
        :param args: sequence strings excluding source, eg "BR SA" "BR AB"
        :return: selection boolean array of photon length

        photon level selection based on history sequence 
        """
        seqs = map(lambda _:"%s %s" % (self.src.upper(),_), args)
        s_seqhis = map(lambda _:self.seqhis == seqhis_int(_), seqs)
        return np.logical_or.reduce(s_seqhis)      


    def recwavelength(self, recs, irec):
        """
        ::

            In [1]: np.load("OPropagatorF.npy")
            Out[1]: 
            array([[[   0.,    0.,    0.,  700.]],

                   [[   0.,  200.,    7.,    0.]],

                   [[  60.,  810.,   20.,  750.]]], dtype=float32)


        cu/wavelength_lookup.h::     

            float nmi = (nm - boundary_domain.x)/boundary_domain.z + 0.5f ;

            #  (60.  - 60.)/20. + 0.5 ->    0.0 + 0.5 
            #  (810. - 60.)/20. + 0.5 ->   37.0 + 0.5  

        cu/photon.h:rsave::

            143     float nwavelength = 255.f*(p.wavelength - boundary_domain.x)/boundary_domain.w ; // 255.f*0.f->1.f 
            ///
            ///        255.*(60. - 60.)/750. -> 255.*0.
            ///        255.*(810.- 60.)/750. -> 255.*1.
            ///
            144         
            145     qquad qpolw ;
            146     qpolw.uchar_.x = __float2uint_rn((p.polarization.x+1.f)*127.f) ;
            147     qpolw.uchar_.y = __float2uint_rn((p.polarization.y+1.f)*127.f) ;
            148     qpolw.uchar_.z = __float2uint_rn((p.polarization.z+1.f)*127.f) ;
            149     qpolw.uchar_.w = __float2uint_rn(nwavelength)  ;
            150 
            151     // tightly packed, polarization and wavelength into 4*int8 = 32 bits (1st 2 npy columns) 
            152     hquad polw ;    // union of short4, ushort4
            153     polw.ushort_.x = qpolw.uchar_.x | qpolw.uchar_.y << 8 ;
            154     polw.ushort_.y = qpolw.uchar_.z | qpolw.uchar_.w << 8 ;
            155    
 
        :: 

            recs = pc.s.rx
            pzwl = recs[:,0,1,1]
            wl = (pzwl & np.uint16(0xFF00)) >> 8

            count_unique(wl)  # gives expected 10 possibilities form the comb

        ::

            In [24]: count_unique(p_wavelength)
            Out[24]: 
            array([[   101.176,  12145.   ],
                   [   168.824,  12037.   ],
                   [   239.412,  12232.   ],
                   [   310.   ,  11995.   ],
                   [   380.588,  13467.   ],
                   [   451.176,  14516.   ],
                   [   518.823,  14841.   ],
                   [   589.412,  15151.   ],
                   [   660.   ,  15621.   ],
                   [   730.588,  15780.   ]])


            p.wavelength = float(photon_id % 10)*70.f + 100.f ; // toothcomb wavelength distribution 

            In [28]: np.arange(10, dtype=np.float32)*70. + 100.
            Out[28]: array([ 100.,  170.,  240.,  310.,  380.,  450.,  520.,  590.,  660.,  730.], dtype=float32)

        """
        boundary_domain = self.fdom[2,0]

        pzwl = recs[:,irec,1,1]

        nwavelength = (pzwl & np.uint16(0xFF00)) >> 8

        p_wavelength = nwavelength.astype(np.float32)*boundary_domain[W]/255.0 + boundary_domain[X]

        return p_wavelength 


    def recpolarization(self, recs, irec):
        """
        ::
 
            In [40]: count_unique(px)
            Out[40]: array([[   127, 137785]])

            In [41]: count_unique(py)
            Out[41]: array([[   127, 137785]])

            In [42]: count_unique(pz)
            Out[42]: array([[   254, 137785]])

        Yep (0,0,1) looks expected for SPOL with ggv-prism::

            350           float3 surfaceNormal = ts.pol ;
            351           float3 photonPolarization = ts.polz == M_SPOL ? normalize(cross(p.direction, surfaceNormal)) : p.direction ;
            352           p.polarization = photonPolarization ;

        """ 
        pxpy = recs[:,irec,1,0]
        pzwl = recs[:,irec,1,1]

        ipx = pxpy & np.uint16(0xFF)
        ipy = (pxpy & np.uint16(0xFF00)) >> 8
        ipz = pzwl & np.uint16(0xFF) 

        px = ipx.astype(np.float32)/127. - 1.
        py = ipy.astype(np.float32)/127. - 1.
        pz = ipz.astype(np.float32)/127. - 1.

        # TODO: form empty of correct shape and populate with these
        # thats easy if irec is just an integer, but it could be a slice



    def post_center_extent(self):
        p_center = self.fdom[0,0,:W] 
        p_extent = self.fdom[0,0,W] 

        t_center = self.fdom[1,0,X]
        t_extent = self.fdom[1,0,Y]
       
        center = np.zeros(4)
        center[:W] = p_center 
        center[W] = t_center

        extent = np.zeros(4)
        extent[:W] = p_extent*np.ones(3)
        extent[W] = t_extent

        return center, extent 

    def recpost(self, recs, irec):
        center, extent = self.post_center_extent()
        p = recs[:,irec,0].astype(np.float32)*extent/32767.0 + center 
        return p 



class Selection(object):
    """
    Apply photon and record level selections 
    """
    def __init__(self, evt, *seqs):
        nrec = 10  # TODO: get from idom ?
        if len(seqs) > 0:
            psel = evt.seqhis_or(*seqs)
            rsel = np.repeat(psel, nrec) 
            ox = evt.ox[psel]
            wl = evt.wavelength[psel]
            rx = evt.rx[rsel].reshape(-1,nrec,2,4)  
        else:
            psel = None
            rsel = None
            ox = evt.ox
            wl = evt.wavelength
            rx = evt.rx.reshape(-1,nrec,2,4)
        pass
        self.evt = evt 
        self.psel = psel
        self.rsel = rsel
        self.ox = ox
        self.rx = rx
        self.wl = wl

    def recpost(self, irec):
        return self.evt.recpost(self.rx, irec)

    def recwavelength(self, irec):
        """
        Hmm empty records yielding the offset 60.::

            In [12]: pc.sel.wavelength(slice(0,10))
            Out[12]: 
            array([[ 660.   ,  660.   ,  660.   , ...,   60.   ,   60.   ,   60.   ],
                   [ 239.412,  239.412,  239.412, ...,   60.   ,   60.   ,   60.   ],
                   [ 451.176,  451.176,  451.176, ...,   60.   ,   60.   ,   60.   ],
                   ..., 
                   [ 380.588,  380.588,  380.588, ...,   60.   ,   60.   ,   60.   ],
                   [ 451.176,  451.176,  451.176, ...,   60.   ,   60.   ,   60.   ],
                   [ 730.588,  730.588,  730.588, ...,   60.   ,   60.   ,   60.   ]], dtype=float32)

        """
        return self.evt.recwavelength(self.rx, irec)

    def recpolarization(self, irec):
        return self.evt.recpolarization(self.rx, irec)

    def check_wavelength(self):
        """
        Compare uncompressed photon final wavelength 
        with first record compressed wavelength
        """
        pwl = self.wl
        rwl = self.recwavelength(0)
        dwl = np.absolute(pwl - rwl)
        assert dwl.max() < 2. 



if __name__ == '__main__':
    e1 = Evt(1)




