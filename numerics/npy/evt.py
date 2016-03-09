#!/usr/bin/env python

import os, logging
import numpy as np

import matplotlib.pyplot as plt

from env.python.utils import *
from env.numerics.npy.types import SeqHis, seqhis_int
from env.numerics.npy.nload import A, I, II
from env.numerics.npy.seq import SeqAna
from env.numerics.npy.history import HisType 
from env.numerics.npy.material import MatType 

costheta_ = lambda a,b:np.sum(a * b, axis = 1)/(np.linalg.norm(a, 2, 1)*np.linalg.norm(b, 2, 1)) 
ntile_ = lambda vec,N:np.tile(vec, N).reshape(-1, len(vec))
cross_ = lambda a,b:np.cross(a,b)/np.repeat(np.linalg.norm(a, 2, 1),3).reshape(-1,3)/np.repeat(np.linalg.norm(b, 2, 1),3).reshape(-1,3)
norm_ = lambda a:a/np.repeat(np.linalg.norm(a, 2,1 ), 3).reshape(-1,3)

deg = np.pi/180.

log = logging.getLogger(__name__)

X,Y,Z,W,T = 0,1,2,3,3


class Evt(object):

    RPOST = {"X":X,"Y":Y,"Z":Z,"W":W,"T":T} 
    RPOL = {"A":X,"B":Y,"C":Z} 

    RQWN_BINSCALE = {"X":100,"Y":100,"Z":100,"W":10,"T":10,"A":1,"B":1,"C":1} 


    def __init__(self, tag="1", src="torch", det="dayabay", seqs=[], not_=False, label=None, nrec=10, rec=True, dbg=False):

        self.nrec = nrec
        self.seqs = seqs

        if label is None:
            label = "%s/%s/%s : %s" % (det, src, tag, ",".join(seqs)) 

        self.label = label
        self.rec = rec

        self.init_metadata(tag, src, det, dbg)
        self.init_photons(tag, src, det, dbg)

        if rec:
            self.init_records(tag, src, det, dbg)
            self.init_selection(seqs, not_ )
        else:
            self.history = None
        pass

    def init_metadata(self, tag, src, det, dbg):
        self.tag = str(tag)
        self.src = src
        self.det = det  

        fdom = A.load_("fdom"+src,tag,det, dbg=dbg) 
        idom = A.load_("idom"+src,tag,det, dbg=dbg) 
        assert idom[0,0,3] == self.nrec

        td = I.load_("md"+src,tag,det, name="t_delta.ini", dbg=dbg)
        tdii = II.load_("md"+src,tag,det, name="t_delta.ini", dbg=dbg)

        self.td = td
        self.tdii = tdii

        self.fdom = fdom
        self.idom = idom

    def init_photons(self, tag, src, det, dbg):
        """
        #. c4 uses shape changing dtype splitting the 32 bits into 4*8 bits  
        """
        ox = A.load_("ox"+src,tag,det) 
        wl = ox[:,2,W] 
        c4 = ox[:,3,2].copy().view(dtype=[('x',np.uint8),('y',np.uint8),('z',np.uint8),('w',np.uint8)]).view(np.recarray)

        log.debug("ox shape %s " % str(ox.shape))

        self.c4 = c4
        self.ox = ox
        self.wl = wl
        self.post = ox[:,0] 
        self.dirw = ox[:,1]
        self.polw = ox[:,2]
        self.flags = ox.view(np.uint32)[:,3,3]

    def init_records(self, tag, src, det, dbg):

        rx = A.load_("rx"+src,tag,det,dbg).reshape(-1, self.nrec, 2, 4)
        ph = A.load_("ph"+src,tag,det,dbg)

        log.debug("rx shape %s " % str(rx.shape))

        seqhis = ph[:,0,0]
        seqmat = ph[:,0,1]

        histype = HisType()
        mattype = MatType()

        cn = "%s:%s" % (str(tag), det)
        # full history without selection
        all_history = SeqAna(seqhis, histype , cnames=[cn])  
        all_material = SeqAna(seqmat, mattype , cnames=[cn])  

        self.rx = rx
        self.ph = ph

        self.seqhis = seqhis
        self.seqmat = seqmat

        self.all_history = all_history
        self.history = all_history

        self.all_material = all_material
        self.material = all_material

        self.histype = histype
        self.mattype = mattype



    def init_selection(self, seqs, not_):
        if not self.rec or len(seqs) == 0:return  

        log.debug("Evt seqs %s " % repr(seqs))
        psel = self.all_history.seq_or(seqs, not_=not_)

        self.ox = self.ox[psel]
        self.c4 = self.c4[psel]
        self.wl = self.wl[psel]
        self.rx = self.rx[psel]

        self.history = SeqAna(self.seqhis[psel], self.histype)   # history with selection applied

        # TODO: material with history selection applied
 

    x = property(lambda self:self.ox[:,0,0])
    y = property(lambda self:self.ox[:,0,1])
    z = property(lambda self:self.ox[:,0,2])
    t = property(lambda self:self.ox[:,0,3])

    def __repr__(self):
        return "Evt(%s,\"%s\",\"%s\",\"%s\", seqs=\"%s\")" % (self.tag, self.src, self.det,self.label, repr(self.seqs))


    def msize(self):
        return float(self.ox.shape[0])/1e6

    def unique_wavelength(self):
        uwl = np.unique(self.wl)
        assert len(uwl) == 1 
        return uwl[0]

    def history_table(self, sli=slice(None)):
        if not self.history:return 
        print self
        self.history.table.sli = sli 
        print self.history.table

    def material_table(self):
        seqmat = self.seqmat
        cu = count_unique(seqmat)
        seqmat_table(cu)
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

    def seqhis_or_not(self, args):
        return np.logical_not(self.seqhis_or(args))    


    def recwavelength(self, irec, recs=None):
        """
        """
        if recs is None:
            recs = self.rx

        boundary_domain = self.fdom[2,0]

        pzwl = recs[:,irec,1,1]

        nwavelength = (pzwl & np.uint16(0xFF00)) >> 8

        p_wavelength = nwavelength.astype(np.float32)*boundary_domain[W]/255.0 + boundary_domain[X]

        return p_wavelength 


    def rpol_(self, irec, recs=None):
        """
        """ 
        if recs is None:
            recs = self.rx

        pxpy = recs[:,irec,1,0]
        pzwl = recs[:,irec,1,1]
        m1m2 = recs[:,irec,1,2]
        bdfl = recs[:,irec,1,3]

        ipx = pxpy & np.uint16(0xFF)
        ipy = (pxpy & np.uint16(0xFF00)) >> 8
        ipz = pzwl & np.uint16(0xFF) 

        m1 = m1m2 & np.uint16(0xFF)  
        m2 = (m1m2 & np.uint16(0xFF00)) >> 8 
        bd = bdfl & np.uint16(0xFF)  
        fl = (bdfl & np.uint16(0xFF00)) >> 8   

        px = ipx.astype(np.float32)/127. - 1.
        py = ipy.astype(np.float32)/127. - 1.
        pz = ipz.astype(np.float32)/127. - 1.

        pol = np.empty( (len(px), 3))
        pol[:,0] = px
        pol[:,1] = py
        pol[:,2] = pz

        return pol

    def rpol_bins(self):
        """
        Avoiding artifacts for char compressed, means 
        using the compression binning.

        Suspect one bin difference in handling somewhere ?

        :: 

              py = ipy/127. - 1.
              plt.hist(py, bins=np.linspace(-1,1,255) )
              plt.show()
              plt.hist(py, bins=np.linspace(-1,1,255) )
              hist -n

        """
        #lo = np.uint16(0x0)/127. - 1. 
        #hi = np.uint16(0xFF)/127. - 1.  # 1.0078740157480315 
        #lb = np.linspace(lo, hi, 255+1+1)
        lb = np.linspace(-1, 1, 255+1)   # 
        return lb 

    def recflags(self, recs, irec):
        m1m2 = recs[:,irec,1,2]
        bdfl = recs[:,irec,1,3]

        m1 = m1m2 & np.uint16(0xFF)  
        m2 = (m1m2 & np.uint16(0xFF00)) >> 8 
        bd = bdfl & np.uint16(0xFF)  
        fl = (bdfl & np.uint16(0xFF00)) >> 8   

        flgs = np.empty( (len(m1), 4), dtype=np.int32)
        flgs[:,0] = m1
        flgs[:,1] = m2
        flgs[:,2] = np.int8(bd)
        flgs[:,3] = fl
        return flgs

    def rflgs_(self, irec, recs=None):
        if recs is None:
            recs = self.rx 
        return self.recflags(recs,irec) 

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

    def tbins(self):
        t_center = self.fdom[1,0,X]
        t_extent = self.fdom[1,0,Y]
        assert(t_center == 0.)
        tb = np.linspace(t_center, t_center + t_extent, 32767+1)
        return tb 

    def pbins(self):
        p_center = self.fdom[0,0,:W]
        p_extent = self.fdom[0,0,W]
        assert p_center[0] == p_center[1] == p_center[2] == 0., p_center
        pb = np.linspace(p_center[0] - p_extent, p_center[1] + p_extent, 2*32767+1)
        return pb 

    def rpost_(self, irec, recs=None):
        """
        Record compression can be regarded as a very early choice of binning, 
        as cannot use other binnings without suffering from artifacts
        so need to access the "compression bins" somehow::

            In [24]: cf.a.rx[:,1,0,3].min()
            Out[24]: A(16, dtype=int16)

            In [25]: cf.a.rx[:,1,0,3].max()
            Out[25]: A(208, dtype=int16)

        The range within some selection is irrelevant, what matters is the 
        domain of the compression, need to find that binning then throw
        away unused edge bins according to the plotted range. 
        """
        if recs is None:
            recs = self.rx 
        center, extent = self.post_center_extent()
        p = recs[:,irec,0].astype(np.float32)*extent/32767.0 + center 
        return p 


    @classmethod
    def _incident_angle(cls, p0, p1, center=[0,0,0]):
        """
        Thinking of parallel beam incident on a sphere at center
        """
        pass
        cen = np.tile(center, len(p1)).reshape(-1,3)
        pin = p1 - p0
        nrm = p1 - cen 
        ct = costheta_(nrm, -pin)
        return ct 

    def incident_angle(self, center=[0,0,0]):
        """
        Thinking of parallel beam incident on a sphere at center
        """
        pass
        p0 = self.rpost_(0)[:,:3]
        p1 = self.rpost_(1)[:,:3]
        return self._incident_angle(p0,p1, center)


    @classmethod
    def _deviation_angle(cls, p_out, side=None, incident=None):

        if side is None:
            side = np.array([1,0,0]) 

        if incident is None:
            incident = np.array([0,0,-1]) 

        if len(side.shape) == 1:
            side = np.tile(side, len(p_out)).reshape(-1,3)

        if len(incident.shape) == 1:
            incident  = np.tile(incident, len(p_out)).reshape(-1,3)

        assert np.sum(side*incident) == 0., "side vector must be perpendicular to incident vectors"

        cside = costheta_(p_out, side)

        cdv = costheta_(incident, p_out)
        dv = np.piecewise( cdv, [cside>=0, cside<0], [np.arccos,lambda _:2*np.pi - np.arccos(_)])  

        return dv 



    def a_recside(self, axis=Z):
        """
        Use a coordinate of the initial position to define side of incidence
        unit vector.
     
        Those at zero are aribitrarily put on one side

        This can be used for defining 0-360 degrees deviation angles 
        """
        a0 = self.p0[:,axis]
        aside  = np.zeros((len(a0),3))
        aside[:,axis] = np.piecewise( a0, [a0>0, a0<0, a0==0], [-1,1,1] ) 
        return aside 

    def a_side(self, axis=X):
        """
        :return: (N,3) unit vector array pointing along the axis of initial generated position  

        #. generation side of initial photon position
        #. NB does not require photon step records, just the c4 photon flags  

        """
        posaxis = (self.c4.x & (0x1 << axis)) != 0   
        vside = np.zeros( (len(posaxis), 3), dtype=np.float32)
        vside[:,axis][posaxis] = -1.
        vside[:,axis][~posaxis] = 1.
        return vside

    def a_deviation_angle(self, incident=None, axis=Z):
        vside = self.a_side(axis=axis)
        return self._deviation_angle(self.p_out, side=vside, incident=incident)  

    def deviation_angle(self, side=None, incident=None):
        """
        Deviation angle for parallel squadrons of incident photons 
        without assuming a bounce count
        """
        p_out = self.ox[:,1, :3]  # final/last direction (bounce limited)
        return self._deviation_angle(p_out, side=side, incident=incident)

    p0 = property(lambda self:self.rpost_(0))
    p_out = property(lambda self:self.ox[:,1, :3], doc="final/last direction (bounce limited)")

    def deviation_angle(self, side=None, incident=None):
        """
        Deviation angle for parallel squadrons of incident photons 
        without assuming a bounce count
        """
        return self._deviation_angle(self.p_out, side=side, incident=incident)


    def rsmry_(self, i):
        flgs = self.rflgs_(i)

        m1s = np.unique(flgs[:,0])
        m2s = np.unique(flgs[:,1])
        bns = np.unique(flgs[:,2])
        fls = np.unique(flgs[:,3])

        flab = self.histype.label
        mlab = self.mattype.label

        if len(m1s) == 1 and len(m2s) == 1 and len(bns) == 1 and len(fls) == 1:
            m1 = m1s[0]
            m2 = m2s[0]
            bn = bns[0]
            abn = abs(bn) - 1
            fl = fls[0]
            smry="m1/m2 %3d/%3d %2s/%2s %4d (%3d) %3d:%s " % (m1,m2,mlab(m1),mlab(m2),bn,abn,fl,flab(fl))
        else:
            smry=repr(flgs) 
        pass
        return smry


    def zrt_profile(self, n, pol=True):
        slab = "z r t"
        if pol:
            slab += " lx ly lz"

        labs = slab.split()
        nqwn = 3
        zrt = np.zeros((n,len(labs)*nqwn))
        tfmt = "%10.3f " * nqwn
        fmt = " ".join(["%s: %s " % (lab, tfmt) for lab in labs])

        for i in range(n):
            p = self.rpost_(i)
            l = self.rpol_(i)
            lx = l[:,0]
            ly = l[:,1]
            lz = l[:,2]

            r = np.linalg.norm(p[:,:2],2,1)
            z = p[:,2]
            t = p[:,3]

            assert len(r)>0
            assert len(z)>0

            zrt[i][0:3] = mmm(z)
            zrt[i][3:6] = mmm(r)
            zrt[i][6:9] = mmm(t)

            if pol:
                zrt[i][9:12] = mmm(lx)
                zrt[i][12:15] = mmm(ly)
                zrt[i][15:18] = mmm(lz)

            smry = self.rsmry_(i)

            szrt =  fmt % tuple(zrt[i].tolist())
            print "%3d %s smry %s " % (i, szrt, smry )

        pass
        return zrt 


def mmm(a):
    amin = a.min()
    amax = a.max()
    amid = (amin+amax)/2.
    return amin, amax, amid


def check_wavelength(evt):
    """
    Compare uncompressed photon final wavelength 
    with first record compressed wavelength
    """

    pwl = evt.wl
    rwl = evt.recwavelength(0)
    dwl = np.absolute(pwl - rwl)
    assert dwl.max() < 2. 



if __name__ == '__main__':

    rec = True  

    cat, src, tag = "rainbow", "torch", "-5"
    #cat, src, tag = "juno", "cerenkov", "1"

    evt = Evt(tag, src, cat, label="tag %s" % tag, rec=rec)

    evt.history_table()

    dv = evt.a_deviation_angle(axis=X)

    plt.close()
    plt.ion()

    plt.hist(dv/deg, bins=360, log=True, histtype="step") 

    plt.show()


  

