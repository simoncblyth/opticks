#!/usr/bin/env python

import os, logging
import numpy as np

import matplotlib.pyplot as plt

from env.python.utils import *
from env.numerics.npy.types import SeqHis, seqhis_int
from env.numerics.npy.nload import A, I, II
from env.numerics.npy.history import History

costheta_ = lambda a,b:np.sum(a * b, axis = 1)/(np.linalg.norm(a, 2, 1)*np.linalg.norm(b, 2, 1)) 
ntile_ = lambda vec,N:np.tile(vec, N).reshape(-1, len(vec))
cross_ = lambda a,b:np.cross(a,b)/np.repeat(np.linalg.norm(a, 2, 1),3).reshape(-1,3)/np.repeat(np.linalg.norm(b, 2, 1),3).reshape(-1,3)
norm_ = lambda a:a/np.repeat(np.linalg.norm(a, 2,1 ), 3).reshape(-1,3)

deg = np.pi/180.

log = logging.getLogger(__name__)

X,Y,Z,W = 0,1,2,3


class Evt(object):
    def __init__(self, tag="1", src="torch", det="dayabay", seqs=[], not_=False, label="", nrec=10, rec=True):

        self.nrec = nrec
        self.seqs = seqs
        self.label = label
        self.rec = rec

        self.init_metadata(tag, src, det)
        self.init_photons(tag, src, det)

        if rec:
            self.init_records(tag, src, det)
            self.init_selection(seqs, not_ )
        else:
            self.history = None
        pass

    def init_metadata(self, tag, src, det):
        self.tag = str(tag)
        self.src = src
        self.det = det  

        fdom = A.load_("fdom"+src,tag,det) 
        idom = A.load_("idom"+src,tag,det) 
        assert idom[0,0,3] == self.nrec

        td = I.load_("md"+src,tag,det, name="t_delta.ini")
        tdii = II.load_("md"+src,tag,det, name="t_delta.ini")

        self.td = td
        self.tdii = tdii

        self.fdom = fdom
        self.idom = idom

    def init_photons(self, tag, src, det):
        """
        #. c4 uses shape changing dtype splitting the 32 bits into 4*8 bits  
        """
        ox = A.load_("ox"+src,tag,det) 
        wl = ox[:,2,W] 
        c4 = ox[:,3,2].copy().view(dtype=[('x',np.uint8),('y',np.uint8),('z',np.uint8),('w',np.uint8)]).view(np.recarray)

        log.info("ox shape %s " % str(ox.shape))

        self.c4 = c4
        self.ox = ox
        self.wl = wl
        self.post = ox[:,0] 
        self.dirw = ox[:,1]
        self.polw = ox[:,2]
        self.flags = ox.view(np.uint32)[:,3,3]

    def init_records(self, tag, src, det):

        rx = A.load_("rx"+src,tag,det).reshape(-1, self.nrec, 2, 4)
        ph = A.load_("ph"+src,tag,det)

        log.info("rx shape %s " % str(rx.shape))

        seqhis = ph[:,0,0]
        seqmat = ph[:,0,1]

        cn = "%s:%s" % (str(tag), det)
        all_history = History(seqhis, cnames=[cn])  # full history without selection

        self.rx = rx
        self.ph = ph
        self.seqhis = seqhis
        self.seqmat = seqmat
        self.all_history = all_history
        self.history = all_history

    def init_selection(self, seqs, not_):
        if not self.rec or len(seqs) == 0:return  

        log.info("Evt seqs %s " % repr(seqs))
        psel = self.all_history.seqhis_or(seqs, not_=not_)

        self.ox = self.ox[psel]
        self.c4 = self.c4[psel]
        self.wl = self.wl[psel]
        self.rx = self.rx[psel]
        self.history = History(self.seqhis[psel])   # history with selection applied
 

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
        flgs[:,2] = bd
        flgs[:,3] = fl
        return flgs

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

    def rpost_(self, irec, recs=None):
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


  

