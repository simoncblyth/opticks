#!/usr/bin/env python

import os, logging, stat, datetime
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from collections import OrderedDict 

from opticks.ana.base import opticks_environment
from opticks.ana.base import opticks_main
from opticks.ana.nbase import count_unique, vnorm
from opticks.ana.nload import A, I, II, tagdir_
from opticks.ana.seq import SeqAna
from opticks.ana.histype import HisType
from opticks.ana.hismask import HisMask
from opticks.ana.mattype import MatType 
from opticks.ana.metadata import Metadata

costheta_ = lambda a,b:np.sum(a * b, axis = 1)/(vnorm(a)*vnorm(b)) 
ntile_ = lambda vec,N:np.tile(vec, N).reshape(-1, len(vec))
cross_ = lambda a,b:np.cross(a,b)/np.repeat(vnorm(a),3).reshape(-1,3)/np.repeat(vnorm(b),3).reshape(-1,3)
norm_ = lambda a:a/np.repeat(vnorm(a), 3).reshape(-1,3)

msk_ = lambda n:(1 << 4*n) - 1  # msk_(0)=0x0 msk_(1)=0xf msk_(2)=0xff msk_(3)=0xfff  


def stamp_(path, fmt="%Y%m%d-%H%M"): 
   if path is None:
       return None
   elif not os.path.exists(path):
       return None
   else:
       return datetime.datetime.fromtimestamp(os.stat(path).st_ctime).strftime(fmt)
   pass

deg = np.pi/180.

log = logging.getLogger(__name__)

X,Y,Z,W,T = 0,1,2,3,3

  



class Evt(object):

    RPOST = {"X":X,"Y":Y,"Z":Z,"W":W,"T":T} 
    RPOL = {"A":X,"B":Y,"C":Z} 

    RQWN_BINSCALE = {"X":100,"Y":100,"Z":100,"W":10,"R":100, "T":10,"A":1,"B":1,"C":1} 

    @classmethod
    def selection(cls, evt, seqs=[], not_=False, label=None, dbg=False):
        if label is None:
            label = evt.label + " sel %s " % (repr(seqs)) 
        pass
        sel = cls(tag=evt.tag, src=evt.src, det=evt.det, seqs=seqs, not_=not_ , label=label, nrec=evt.nrec, rec=evt.rec, dbg=dbg )
        return sel 


    def __init__(self, tag="1", src="torch", det="dayabay", seqs=[], not_=False, label=None, nrec=10, rec=True, dbg=False, terse=False, dbgseqhis=0, dbgseqmat=0):

        self.valid = True   ## load failures signalled by setting False
        self.nrec = nrec
        self.seqs = seqs
        self.terse = terse
        self.dbgseqhis = dbgseqhis
        self.dbgseqmat = dbgseqmat
       
 
        if label is None:
            label = "%s/%s/%3s : %s" % (det, src, tag, ",".join(seqs)) 

        self.label = label
        self.rec = rec
        self.desc = OrderedDict()

        ok = self.init_metadata(tag, src, det, dbg)

        if not ok:
           log.warning("FAILED TO LOAD EVT %s " % label )
           return   

        self.init_types()
        self.init_gensteps(tag, src, det, dbg)
        self.init_photons(tag, src, det, dbg)
        self.init_hits(tag, src, det, dbg)

        if rec:
            self.init_records(tag, src, det, dbg)
            self.init_sequence(tag, src, det, dbg)
            self.init_selection(seqs, not_ )
            self.init_index(tag, src, det, dbg)

        pass
        self.check_stamps()


    def init_types(self):
        log.debug("init_types")
        self.hismask = HisMask()
        self.histype = HisType()
        self.mattype = MatType()
        log.debug("init_types DONE")


    def init_metadata(self, tag, src, det, dbg):
        self.tag = str(tag)
        self.src = src
        self.det = det  
        self.tagdir = tagdir_(det, src, tag)

        metadata = Metadata(self.tagdir)
        log.debug("loaded metadata from %s : %s " % (self.tagdir, repr(metadata)))
        self.metadata = metadata  

        fdom = A.load_("fdom",src,tag,det, dbg=dbg) 
        idom = A.load_("idom",src,tag,det, dbg=dbg) 

        if idom is None and fdom is None:
            log.warning("failed to load idom and fdom")
            self.fdom = None
            self.idom = None
            self.valid = False
            return False

        if idom[0,0,3] != self.nrec:
            log.warning(" non-standard idom %s nrec %s " % (repr(idom), self.nrec ))  

        #assert idom[0,0,3] == self.nrec

        td = I.load_(src,tag,det, name="t_delta.ini", dbg=dbg)
        #log.info("loaded td %s " % str(td.keys()))
        tdii = II.load_(src,tag,det, name="t_delta.ini", dbg=dbg)

        self.td = td
        self.tdii = tdii

        self.fdom = fdom
        self.idom = idom

        fdom.desc = "(metadata) 3*float4 domains of position, time, wavelength (used for compression)"
        self.desc['fdom'] = fdom.desc
        self.desc['idom'] = "(metadata) int domain"
        return True         

    def init_gensteps(self, tag, src, det, dbg):
        """
        """
        gs = A.load_("gs",src,tag,det, optional=True) 

        self.gs = gs
        self.desc['gs'] = "(gensteps)"


    def init_photons(self, tag, src, det, dbg):
        """
        #. c4 uses shape changing dtype splitting the 32 bits into 4*8 bits  
        """
        ox = A.load_("ox",src,tag,det, optional=True ) 
        self.ox = ox
        self.desc['ox'] = "(photons) final photon step"

        if ox.missing:return 

        wl = ox[:,2,W] 
        c4 = ox[:,3,2].copy().view(dtype=[('x',np.uint8),('y',np.uint8),('z',np.uint8),('w',np.uint8)]).view(np.recarray)

        log.debug("ox shape %s " % str(ox.shape))

        self.wl = wl
        self.post = ox[:,0] 
        self.dirw = ox[:,1]
        self.polw = ox[:,2]
        self.pflags = ox.view(np.uint32)[:,3,3]
        self.c4 = c4

        cn = "%s:%s" % (str(tag), det)
        self.pflags_ana = SeqAna( self.pflags, self.hismask, cnames=[cn] )

        self.desc['wl'] = "(photons) wavelength"
        self.desc['post'] = "(photons) final photon step: position, time"
        self.desc['dirw'] = "(photons) final photon step: direction, weight "
        self.desc['polw'] = "(photons) final photon step: polarization, wavelength "
        self.desc['pflags'] = "(photons) final photon step: flags "
        self.desc['c4'] = "(photons) final photon step: dtype split uint8 view of ox flags"


    def init_hits(self, tag, src, det, dbg):
        ht = A.load_("ht",src,tag,det, optional=True) 
        self.ht = ht
        self.desc['ht'] = "(hits) surface detect SD final photon steps"

        if ht.missing:return

        hwl = ht[:,2,W] 
        hc4 = ht[:,3,2].copy().view(dtype=[('x',np.uint8),('y',np.uint8),('z',np.uint8),('w',np.uint8)]).view(np.recarray)

        self.hwl = hwl
        self.hpost = ht[:,0] 
        self.hdirw = ht[:,1]
        self.hpolw = ht[:,2]
        self.hflags = ht.view(np.uint32)[:,3,3]
        self.hc4 = hc4

        cn = "%s:%s" % (str(tag), det)
        self.hflags_ana = SeqAna( self.hflags, self.hismask, cnames=[cn] )
 
        self.desc['hwl'] = "(hits) wavelength"
        self.desc['hpost'] = "(hits) final photon step: position, time"
        self.desc['hdirw'] = "(hits) final photon step: direction, weight "
        self.desc['hpolw'] = "(hits) final photon step: polarization, wavelength "
        self.desc['hflags'] = "(hits) final photon step: flags "
        self.desc['hc4'] = "(hits) final photon step: dtype split uint8 view of ox flags"


    def init_records(self, tag, src, det, dbg):
        """
        reshaping no longer needed ?
        """
        # rx_raw = A.load_("rx",src,tag,det,dbg)
        # self.rx_raw = rx_raw
        # self.desc['rx_raw'] = "(records) photon step records RAW:before reshaping"
        # rx = rx_raw.reshape(-1, self.nrec, 2, 4)

        rx = A.load_("rx",src,tag,det,dbg, optional=True)
        self.rx = rx
        self.desc['rx'] = "(records) photon step records"

        if rx.missing:return 
       
        log.debug("rx shape %s " % str(rx.shape))

    def init_sequence(self, tag, src, det, dbg):
        """
        Sequence values seqhis and seqmat for each photon::

            In [8]: a.ph.shape
            Out[8]: (100, 1, 2)

            In [9]: np.set_printoptions(formatter={'int':hex})

            In [10]: a.ph
            Out[10]: 
            A(torch,1,laser)-
            A([[[0x8ccccdL, 0x343231L]],
               [[0x8ccccdL, 0x343231L]],
               [[0x8cccc6dL, 0x3432311L]],
               [[0x8ccccdL, 0x343231L]],
               [[0x8ccccdL, 0x343231L]],
               [[0xbcbccccc6dL, 0x3333342311L]],

        """
        log.debug("init_sequence START")

        ph = A.load_("ph",src,tag,det,dbg, optional=True)
        self.ph = ph
        self.desc['ph'] = "(records) photon history flag/material sequence"
        if ph.missing:
            log.debug(" ph missing ==> no history aka seqhis_ana  ")
            return 

        seqhis = ph[:,0,0]
        seqmat = ph[:,0,1]

        cn = "%s:%s" % (str(tag), det)
        # full history without selection
        all_seqhis_ana = SeqAna(seqhis, self.histype , cnames=[cn], dbgseq=self.dbgseqhis)  
        all_seqmat_ana = SeqAna(seqmat, self.mattype , cnames=[cn], dbgseq=self.dbgseqmat)  

        self.seqhis = seqhis
        self.seqmat = seqmat

        useqhis = len(np.unique(seqhis))
        useqmat = len(np.unique(seqmat))

        if useqhis <= 1:
            log.warning("init_records %s finds too few (ph)seqhis uniques : %s : EMPTY HISTORY" % (self.label,useqhis) ) 
        if useqmat <= 1:
            log.warning("init_records %s finds too few (ph)seqmat uniques : %s : EMPTY HISTORY" % (self.label,useqmat) ) 

        self.all_seqhis_ana = all_seqhis_ana
        self.seqhis_ana = all_seqhis_ana

        self.all_seqmat_ana = all_seqmat_ana
        self.seqmat_ana = all_seqmat_ana

        for imsk in range(1,10):
            msk = msk_(imsk) 
            setattr(self, "seqhis_ana_%d" % imsk, SeqAna(seqhis & msk, self.histype, cnames=[cn])) 

        log.debug("init_sequence DONE")


    def init_index(self, tag, src, det, dbg):
        """
        Sequence indices for each photon::
 
            In [2]: a.ps.shape
            Out[2]: (100, 1, 4)

            In [1]: a.ps
            Out[1]: 
            A([[[ 1,  1,  0,  0]],
               [[ 1,  1,  0,  0]],
               [[ 3,  3,  0,  0]],
               [[ 1,  1,  0,  0]],

            In [6]: a.rs.shape      ## same information as a.ps duped by maxrec for record access
            Out[6]: (100, 10, 1, 4)

        """
        ps = A.load_("ps",src,tag,det,dbg, optional=True)

        if not ps is None and not ps.missing:
            ups = len(np.unique(ps))
        else:
            ups = -1 

        rs = A.load_("rs",src,tag,det,dbg, optional=True)
        if not rs is None and not rs.missing :
            urs = len(np.unique(rs))
        else: 
            urs = -1

        if not rs is None and not rs.missing:
            rsr = rs.reshape(-1, self.nrec, 1, 4)        
        else: 
            rsr = None

        if not rsr is None:
            ursr = len(np.unique(rsr))
        else:
            ursr = -1


        if ups <= 1 and not ps.missing:
            log.warning("init_index %s finds too few (ps)phosel uniques : %s" % (self.label,ups) ) 
        if urs <= 1 and not rs.missing:
            log.warning("init_index %s finds too few (rs)recsel uniques : %s" % (self.label,urs) ) 
        if ursr <= 1 and not rs.missing:
            log.warning("init_index %s finds too few (rsr)reshaped-recsel uniques : %s" % (self.label,ursr) ) 


        self.ps = ps
        self.rs = rs 
        self.rsr = rsr 

        if not ps is None:
            ps.desc = "(photons) phosel sequence frequency index lookups (uniques %d)"  % ups
            self.desc['ps'] = ps.desc

        if not rs is None:
            rs.desc = "(records) RAW recsel sequence frequency index lookups (uniques %d)"  % urs 
            self.desc['rs'] = rs.desc

        if not rsr is None:
            rsr.desc = "(records) RESHAPED recsel sequence frequency index lookups (uniques %d)"  % ursr 
            self.desc['rsr'] = rsr.desc


    def init_selection(self, seqs, not_):
        if not self.rec or len(seqs) == 0:return  

        log.debug("Evt seqs %s " % repr(seqs))
        psel = self.all_seqhis_ana.seq_or(seqs, not_=not_)

        nsel = len(psel[psel == True])
        if nsel == 0:
            log.warning("empty selection seqs %s " % repr(seqs))

        self.nsel = nsel 
        self.psel = psel 

        self.ox = self.ox[psel]
        self.c4 = self.c4[psel]
        self.wl = self.wl[psel]
        self.rx = self.rx[psel]

        self.seqhis_ana = SeqAna(self.seqhis[psel], self.histype)   # sequence history with selection applied
    
 
    x = property(lambda self:self.ox[:,0,0])
    y = property(lambda self:self.ox[:,0,1])
    z = property(lambda self:self.ox[:,0,2])
    t = property(lambda self:self.ox[:,0,3])


    description = property(lambda self:"\n".join(["%5s : %20s : %s " % (k, repr(getattr(self,k).shape),  label) for k,label in self.desc.items()]))
    paths = property(lambda self:"\n".join(["%5s : %s " % (k, repr(getattr(getattr(self,k),'path','-'))) for k,label in self.desc.items()]))

    def _path(self):
        if self.fdom is None:
             return None
        else:
             return self.fdom.path
    path = property(_path)
    stamp = property(lambda self:stamp_(self.path))


    def check_stamps(self, names="fdom idom ox rx ht"):
        sst = set()

        lines = []
        for name in names.split():
            a = getattr(self, name, None)
            if a is None:continue
            p = getattr(a,"path",None)
            t = stamp_(p)
            sst.add(t)
            lines.append("%10s  %s  %s " % (name, p, t ))

        nstamp = len(sst)
        if nstamp > 1:
            log.warning("MIXED TIMESTAMP EVENT DETECTED")
            print "\n".join(lines)

        return nstamp


    def _brief(self):
        if self.valid:
             return "%s %s %s" % (self.label, self.stamp, self.path)
        else:
             return "%s %s" % (self.label, "EVT LOAD FAILED")

    brief = property(_brief)
    summary = property(lambda self:"Evt(%3s,\"%s\",\"%s\",\"%s\", seqs=\"%s\") %s %s" % 
            (self.tag, self.src, self.det,self.label, repr(self.seqs), self.stamp, self.tagdir))

    def __repr__(self):
        if self.terse:
            elem = [self.summary]
        else:
            elem = [self.summary, self.description]
        pass
        return "\n".join(elem)

    def msize(self):
        return float(self.ox.shape[0])/1e6

    def unique_wavelength(self):
        uwl = np.unique(self.wl)
        assert len(uwl) == 1 
        return uwl[0]

    def present_table(self, analist,sli=slice(None)):
        """
        TODO: make the table directly sliceable
        """
        for ana_ in analist:
            ana = getattr(self, ana_, None) 
            if ana:
                log.debug("history_table %s " % ana_ )
                ana.table.title = ana_ 
                ana.table.sli = sli 
                print ana.table
            else:
                log.debug("%s noattr " % ana_ )
        pass

    def history_table(self, sli=slice(None)):
        self.present_table( 'seqhis_ana seqmat_ana hflags_ana pflags_ana'.split(), sli)

    def material_table(self, sli=slice(None)):
        self.present_table( 'seqmat_ana'.split(), sli)

    @classmethod
    def compare_table(cls, a, b, analist='seqhis_ana seqmat_ana'.split(), lmx=20, c2max=None, cf=True):
        if not (a.valid and b.valid):
            log.fatal("need two valid events to compare ")
            sys.exit(1)

        cft = {}
        for ana_ in analist:
            a_ana = getattr(a, ana_, None)
            b_ana = getattr(b, ana_, None)

            if a_ana is None:
                log.warning("missing a_ana %s " % ana_ )  
                continue
            if b_ana is None:
                log.warning("missing b_ana  %s " % ana_ )  
                continue
                 
            a_tab = a_ana.table
            b_tab = b_ana.table

            if cf:
                c_tab = a_tab.compare(b_tab)
                c_tab.title = ana_

                cft[ana_] = c_tab 

                if len(c_tab.lines) > lmx:
                    c_tab.sli = slice(0,lmx)

                print c_tab
                if c2max is not None:
                    assert c_tab.c2p < c2max, "c2p deviation for table %s c_tab.c2p %s >= c2max %s " % ( ana_, c_tab.c2p, c2max )

            else:
                a_tab.title = "A:%s " % ana_
                if len(a_tab.lines) > lmx:
                    a_tab.sli = slice(0,lmx)
                print a_tab

                b_tab.title = "B:%s " % ana_
                if len(b_tab.lines) > lmx:
                    b_tab.sli = slice(0,lmx)
                print b_tab
            pass
        pass
        return cft





    def material_table_old(self):
        seqmat = self.seqmat
        cu = count_unique(seqmat)
        ## TODO: fix this  
        seqmat_table(cu)
        tot = cu[:,1].astype(np.int32).sum()
        print "tot:", tot
        return cu

    def flags_table_old(self):
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
        TODO: rearrange to go direct from recs to the 
              result without resorting to new allocation
              (maybe by viewing as recarray of appropriate types)

              This then allows irec to be a slice

        In [80]: evt.rx[:,0,1,0:2].copy().view(np.uint8).astype(np.float32)/127.-1.
        Out[80]: 
        A([[-0.378,  0.929,  0.   , -0.157],
               [-0.244,  0.969,  0.   , -0.157],
               [-1.   ,  0.   ,  0.   , -0.157],
               ..., 
               [-0.992,  0.094,  0.   , -0.157],
               [-0.78 , -0.63 ,  0.   , -0.157],
               [ 0.969, -0.244,  0.   , -0.157]], dtype=float32)

        In [81]: evt.rpol_(0)
        Out[81]: 
        array([[-0.378,  0.929,  0.   ],
               [-0.244,  0.969,  0.   ],
               [-1.   ,  0.   ,  0.   ],
               ..., 
               [-0.992,  0.094,  0.   ],
               [-0.78 , -0.63 ,  0.   ],
               [ 0.969, -0.244,  0.   ]])


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

        log.info(" pbins p_center %s p_extent %s " % (repr(p_center), repr(p_extent)))

        #assert p_center[0] == p_center[1] == p_center[2] == 0., p_center
        pb = np.linspace(p_center[0] - p_extent, p_center[1] + p_extent, 2*32767+1)
        return pb 

    def rpost_(self, irec, recs=None):
        """
        NB irec can be a slice, eg slice(0,5)

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
        """
        :param n: number of bounce steps 
        :return: min, max, mid triplets for z, r and t  at n bounce steps

        ::

            In [7]: a_zrt
            Out[7]: 
            array([[ 300.    ,  300.    ,  300.    ,    1.1748,   97.0913,   49.133 ,    0.1001,    0.1001,    0.1001],
                   [  74.2698,  130.9977,  102.6337,    1.1748,   97.0913,   49.133 ,    0.9357,    1.2165,    1.0761],
                   [  56.0045,  127.9946,   91.9996,    1.1748,   98.1444,   49.6596,    0.9503,    1.3053,    1.1278]])


        """
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

            #r = np.linalg.norm(p[:,:2],2,1)
            r = vnorm(p[:,:2])
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
    """
    :param a: numpy array
    :return: min,max,mid
    """
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
    np.set_printoptions(precision=4, linewidth=200)

    opticks_environment()

    args = opticks_main(doc=__doc__, tag="10", src="torch", det="PmtInBox", c2max=2.0, tagoffset=0)

    rec = True  
    evt = Evt(tag=args.utag, src=args.src, det=args.det, label="utag %s" % args.utag, rec=rec)

    evt.history_table(slice(0,20))

    #dv = evt.a_deviation_angle(axis=X)
    #
    #if plt:
    #    plt.close()
    #    plt.ion()
    #
    #    plt.hist(dv/deg, bins=360, log=True, histtype="step") 
    #
    #    plt.show()


  

