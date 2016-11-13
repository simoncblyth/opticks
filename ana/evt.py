#!usr/bin/env python

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
from opticks.ana.seq import SeqAna, seq2msk
from opticks.ana.histype import HisType
from opticks.ana.hismask import HisMask
from opticks.ana.mattype import MatType 
from opticks.ana.metadata import Metadata



pdict_ = lambda d:" ".join(["%s:%s" % kv for kv in d.items()])
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
    """
    Using interactive high level *sel* selection::

        In [9]: a.sel = "TO BT BT BT BT DR SA"
        [2016-11-11 13:23:06,182] p78027 {/Users/blyth/opticks/ana/evt.py:546} INFO - psel array([False, False, False, ..., False, False, False], dtype=bool) 
        [2016-11-11 13:23:06,184] p78027 {/Users/blyth/opticks/ana/evt.py:553} INFO - _init_selection nsel 7540 len(psel) 1000000  

        In [10]: a.his
        Out[10]: 
        .                                noname 
        .                                  7540         1.00 
           0              89ccccd        1.000           7540         [7 ] TO BT BT BT BT DR SA
        .                                  7540         1.00 

        In [15]: a.sel = None    # clear selection with None

        In [16]: a.his[:10]
        Out[16]: 
        .                                noname 
        .                               1000000         1.00 
           0               8ccccd        0.670         669843         [6 ] TO BT BT BT BT SA
           1                   4d        0.084          83950         [2 ] TO AB
           2              8cccc6d        0.045          45490         [7 ] TO SC BT BT BT BT SA
           3               4ccccd        0.029          28955         [6 ] TO BT BT BT BT AB
           4                 4ccd        0.023          23187         [4 ] TO BT BT AB
           5              8cccc5d        0.020          20239         [7 ] TO RE BT BT BT BT SA
           6              8cc6ccd        0.010          10214         [7 ] TO BT BT SC BT BT SA
           7              86ccccd        0.010          10176         [7 ] TO BT BT BT BT SC SA
           8              89ccccd        0.008           7540         [7 ] TO BT BT BT BT DR SA
           9             8cccc55d        0.006           5970         [8 ] TO RE RE BT BT BT BT SA
        .                               1000000         1.00 


    Using interative low level *psel* selection, implemented via property setter which invokes _init_selection, to 
    select photons that have SA in the topslot ::

        In [7]: e1.psel = e1.seqhis & np.uint64(0xf000000000000000) == np.uint64(0x8000000000000000)
        [2016-11-07 11:44:22,464] p45629 {/Users/blyth/opticks/ana/evt.py:367} INFO - _init_selection nsel 2988 len(psel) 1000000  

        In [9]: e1.psel = e1.seqhis & 0xf000000000000000 == 0x8000000000000000         ## no need to specify the types
        [2016-11-07 11:46:25,879] p45629 {/Users/blyth/opticks/ana/evt.py:367} INFO - _init_selection nsel 2988 len(psel) 1000000  

        In [14]: e1.psel = ( e1.seqhis & ( 0xf << 4*15 )) >> 4*15 == 0x8     ## express with bitshifts
        [2016-11-07 11:57:06,754] p45629 {/Users/blyth/opticks/ana/evt.py:367} INFO - _init_selection nsel 2988 len(psel) 1000000  


        In [8]: e1.his
        Out[8]: 
        .                                noname 
        .                                  2988         1.00 
           0     8cccc6cccc9ccccd        0.189            566         [16] TO BT BT BT BT DR BT BT BT BT SC BT BT BT BT SA
           1     8cccccccc9cccc6d        0.122            365         [16] TO SC BT BT BT BT DR BT BT BT BT BT BT BT BT SA
           2     8cccc5cccc9ccccd        0.074            222         [16] TO BT BT BT BT DR BT BT BT BT RE BT BT BT BT SA
           3     8cccc6cccc6ccccd        0.052            155         [16] TO BT BT BT BT SC BT BT BT BT SC BT BT BT BT SA
           4     8cccccccc9cccc5d        0.050            150         [16] TO RE BT BT BT BT DR BT BT BT BT BT BT BT BT SA
           5     8cccccc6cc9ccccd        0.041            123         [16] TO BT BT BT BT DR BT BT SC BT BT BT BT BT BT SA
           6     8cc6cccccc9ccccd        0.040            119         [16] TO BT BT BT BT DR BT BT BT BT BT BT SC BT BT SA
           7     8cccccccc6cccc6d        0.038            115         [16] TO SC BT BT BT BT SC BT BT BT BT BT BT BT BT SA
           8     86cccccccc9ccccd        0.034            101         [16] TO BT BT BT BT DR BT BT BT BT BT BT BT BT SC SA
        ....

        In [11]: e1.mat
        Out[11]: 
        .                                noname 
        .                                  2988         1.00 
           0     3432311323443231        0.326            975         [16] Gd Ac LS Ac MO MO Ac LS Ac Gd Gd Ac LS Ac MO Ac
           1     3432313234432311        0.228            681         [16] Gd Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac MO Ac
           2     3432313234443231        0.085            253         [16] Gd Ac LS Ac MO MO MO Ac LS Ac Gd Ac LS Ac MO Ac
           3     3443231323443231        0.080            239         [16] Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac MO MO Ac
           4     3432231323443231        0.076            226         [16] Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS LS Ac MO Ac
           5     3432313223443231        0.073            219         [16] Gd Ac LS Ac MO MO Ac LS LS Ac Gd Ac LS Ac MO Ac
           6     3432313234432231        0.050            149         [16] Gd Ac LS LS Ac MO MO Ac LS Ac Gd Ac LS Ac MO Ac
           7     3432344323443231        0.026             79         [16] Gd Ac LS Ac MO MO Ac LS Ac MO MO Ac LS Ac MO Ac
           8     3432344323132231        0.016             47         [16] Gd Ac LS LS Ac Gd Ac LS Ac MO MO Ac LS Ac MO Ac
        ....


    """
    RPOST = {"X":X,"Y":Y,"Z":Z,"W":W,"T":T} 
    RPOL = {"A":X,"B":Y,"C":Z} 

    RQWN_BINSCALE = {"X":1000,"Y":1000,"Z":1000,"W":10,"R":1000, "T":1000,"A":1,"B":1,"C":1} 

    @classmethod
    def selection(cls, evt, seqs=[], not_=False, label=None, dbg=False):
        if label is None:
            label = evt.label + " sel %s " % (repr(seqs)) 
        pass
        sel = cls(tag=evt.tag, src=evt.src, det=evt.det, seqs=seqs, not_=not_ , label=label, maxrec=evt.maxrec, rec=evt.rec, dbg=dbg )
        return sel 


    def __init__(self, tag="1", src="torch", det="dayabay", args=None, maxrec=10, rec=True, dbg=False, label=None, seqs=[], not_=False, nom="?" ):
        log.info("%s.__init__ START " % nom)
        self.nom = nom
        self._psel = None
        self._labels = []

        self.valid = True   ## load failures signalled by setting False
        self.maxrec = maxrec
        self.seqs = seqs
        self.flv = "seqhis"  # default for selections


        self.terse = args.terse
        self.dbgseqhis = args.dbgseqhis
        self.dbgmskhis = args.dbgmskhis
        self.dbgseqmat = args.dbgseqmat
        self.dbgmskmat = args.dbgmskmat
        self.dbgzero = args.dbgzero 
        self.cmx = args.cmx

        log.debug( " seqs %s " % repr(seqs))
        log.debug(" dbgseqhis %x dbgmskhis %x dbgseqmat %x dbgmskmat %x " % (args.dbgseqhis, args.dbgmskhis, args.dbgseqmat, args.dbgmskmat ))
 
        if label is None:
            label = "%s/%s/%3s : %s" % (det, src, tag, ",".join(self.seqs)) 

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

            if len(seqs) == 0:
                psel = None
            else:
                psel = self.make_selection(seqs, not_)
            pass
            self.psel = psel      # psel property setter

            #self.init_index(tag, src, det, dbg)
        pass
        self.check_stamps()
        log.info("%s.__init__ DONE " % nom)


    def init_types(self):
        log.info("init_types")
        self.hismask = HisMask()
        self.histype = HisType()
        self.mattype = MatType()
        log.debug("init_types DONE")


    def init_metadata(self, tag, src, det, dbg):
        log.info("init_metadata")
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

        td = I.load_(src,tag,det, name="t_delta.ini", dbg=dbg)
        tdii = II.load_(src,tag,det, name="t_delta.ini", dbg=dbg)

        self.td = td
        self.tdii = tdii

        self.fdom = fdom
        self.fdomd = { 
                        'tmin':fdom[1,0,0],
                        'tmax':fdom[1,0,1],
                        'animtmax':fdom[1,0,2],
                        'xmin':fdom[0,0,0]-fdom[0,0,3],
                        'xmax':fdom[0,0,0]+fdom[0,0,3],
                        'ymin':fdom[0,0,1]-fdom[0,0,3],
                        'ymax':fdom[0,0,1]+fdom[0,0,3],
                        'zmin':fdom[0,0,2]-fdom[0,0,3],
                        'zmax':fdom[0,0,2]+fdom[0,0,3],
                     }
        self.idom = idom
        self.idomd = dict(maxrec=idom[0,0,3],maxbounce=idom[0,0,2],maxrng=idom[0,0,1])

        fdom.desc = "(metadata) 3*float4 domains of position, time, wavelength (used for compression)"
        self.desc['fdom'] = fdom.desc
        self.desc['idom'] = "(metadata) %s " % pdict_(self.idomd)
        return True         



    def init_gensteps(self, tag, src, det, dbg):
        """
        """
        log.info("init_gensteps")
        gs = A.load_("gs",src,tag,det, optional=True) 

        self.gs = gs
        self.desc['gs'] = "(gensteps)"


    def init_photons(self, tag, src, det, dbg):
        """
        #. c4 uses shape changing dtype splitting the 32 bits into 4*8 bits  
        """
        log.info("init_photons")
        ox = A.load_("ox",src,tag,det, optional=True ) 
        self.ox = ox
        self.desc['ox'] = "(photons) final photon step"

        if ox.missing:return 

        self.check_ox_fdom()

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
        self.pflags_ana = SeqAna( self.pflags, self.hismask, cnames=[cn], dbgseq=self.dbgmskhis, dbgzero=self.dbgzero, cmx=self.cmx )

        self.desc['wl'] = "(photons) wavelength"
        self.desc['post'] = "(photons) final photon step: position, time"
        self.desc['dirw'] = "(photons) final photon step: direction, weight "
        self.desc['polw'] = "(photons) final photon step: polarization, wavelength "
        self.desc['pflags'] = "(photons) final photon step: flags "
        self.desc['c4'] = "(photons) final photon step: dtype split uint8 view of ox flags"


    def check_ox_fdom(self):
        chks = [ 
                  [ 'x', self.ox[:,0,0], self.fdomd["xmin"], self.fdomd["xmax"] ],
                  [ 'y', self.ox[:,0,1], self.fdomd["ymin"], self.fdomd["ymax"] ],
                  [ 'z', self.ox[:,0,2], self.fdomd["zmin"], self.fdomd["zmax"] ],
                  [ 't', self.ox[:,0,3], self.fdomd["tmin"], self.fdomd["tmax"] ],
               ]
        tot = 0 
        for l,q,mi,mx in chks:
            nmx = np.count_nonzero( q > mx ) 
            nmi = np.count_nonzero( q < mi ) 
            nto = len(q)

            tot += nmx
            tot += nmi

            fmt = "%5.3f"
            if nto > 0:
                fmx = fmt % (float(nmx)/float(nto)) 
                fmi = fmt % (float(nmi)/float(nto)) 
            else:
                fmx = "-"
                fmi = "-"

            msg = " %s : %7.3f %7.3f : tot %d over %d %s  under %d %s : mi %10.3f mx %10.3f  " % (l, mi, mx, nto, nmx,fmx,  nmi, fmi, q.min(), q.max() )
            if nmx == 0 and nmi == 0: 
                log.debug(msg)
            else:
                log.warning(msg)
            pass
        pass
        return tot


    def init_hits(self, tag, src, det, dbg):
        log.info("init_hits")
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
        self.hflags_ana = SeqAna( self.hflags, self.hismask, cnames=[cn], dbgseq=self.dbgmskhis, dbgzero=self.dbgzero, cmx=self.cmx)
 
        self.desc['hwl'] = "(hits) wavelength"
        self.desc['hpost'] = "(hits) final photon step: position, time"
        self.desc['hdirw'] = "(hits) final photon step: direction, weight "
        self.desc['hpolw'] = "(hits) final photon step: polarization, wavelength "
        self.desc['hflags'] = "(hits) final photon step: flags "
        self.desc['hc4'] = "(hits) final photon step: dtype split uint8 view of ox flags"


    def init_records(self, tag, src, det, dbg):
        """
        """
        log.info("init_records")
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
        log.info("init_sequence START")

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
        all_seqhis_ana = SeqAna(seqhis, self.histype , cnames=[cn], dbgseq=self.dbgseqhis, dbgmsk=self.dbgmskhis, dbgzero=self.dbgzero, cmx=self.cmx)  
        all_seqmat_ana = SeqAna(seqmat, self.mattype , cnames=[cn], dbgseq=self.dbgseqmat, dbgmsk=self.dbgmskmat, dbgzero=self.dbgzero, cmx=self.cmx)  

        self.seqhis = seqhis
        self.pflags2 = seq2msk(seqhis) 

        self.msk_mismatch = self.pflags != self.pflags2

        if np.all( self.msk_mismatch == False  ):
            log.debug("pflags2(=seq2msk(seqhis)) and pflags  match")
        else:
            log.info("pflags2(=seq2msk(seqhis)) and pflags  MISMATCH (msk_mismatch)")
        pass

        self.seqmat = seqmat

        useqhis = len(np.unique(seqhis))
        useqmat = len(np.unique(seqmat))

        if useqhis <= 1:
            log.warning("init_records %s finds too few (ph)seqhis uniques : %s : EMPTY HISTORY" % (self.label,useqhis) ) 
        if useqmat <= 1:
            log.warning("init_records %s finds too few (ph)seqmat uniques : %s : EMPTY HISTORY" % (self.label,useqmat) ) 

        self.all_seqhis_ana = all_seqhis_ana
        self.seqhis_ana = all_seqhis_ana
        ## when a selection is used seqhis_ana gets trumped by init_selection

        self.all_seqmat_ana = all_seqmat_ana
        self.seqmat_ana = all_seqmat_ana

        for imsk in range(1,10):
            msk = msk_(imsk) 
            setattr(self, "seqhis_ana_%d" % imsk, SeqAna(seqhis & msk, self.histype, cnames=[cn])) 

        log.info("init_sequence DONE")

    his = property(lambda self:self.seqhis_ana.table)
    mat = property(lambda self:self.seqmat_ana.table)
    ahis = property(lambda self:self.all_seqhis_ana.table)
    amat = property(lambda self:self.all_seqmat_ana.table)

    def make_labels(self, sel):
        """
        :param sel: selection choice argument 

        Sel can be of various types:

        *slice*
            picks seqmat or seqhis labels depending on the flv member
        *list*
            with elements which can be 
 
            *hexint* 
                eg 0x8ccd 

            *hexstring(without 0x)* 
                 eg "8ccd"

            *string* 
                 preformed labels "TO BT BT AB"  

        """ 
        if type(sel) is slice:
            if self.flv == "seqhis":
                labels = self.ahis.labels[sel] 
            elif self.flv == "seqmat":
                labels = self.amat.labels[sel] 
            else:
                assert 0, flv
            pass
        elif type(sel) is list:
            if self.flv == "seqhis":
                labels = map( lambda _:self.histype.label(_), sel)
            elif self.flv == "seqmat":
                labels = map( lambda _:self.mattype.label(_), sel)
            else:
                assert 0, flv
        elif type(sel) is str or type(sel) is int or type(sel) is np.uint64:
            if self.flv == "seqhis":
                labels = [self.histype.label(sel)]
            elif self.flv == "seqmat":
                labels = [self.mattype.label(sel)]
            else:
                assert 0, flv
            pass
        else:
            log.fatal("unhandled selection type %s %s " % (sel, type(sel)))
            assert 0

        log.debug("%s.make_labels %s ->  %s " % (self.nom, repr(sel), repr(labels)))
        return labels
 
    def make_psel_startswith(self, lab):
        if self.flv == "seqhis":
            psel = self.all_seqhis_ana.seq_startswith(lab)
        elif self.flv == "seqmat":
            psel = self.all_seqmat_ana.seq_startswith(lab)
        pass
        return psel

    def make_psel_or(self, labs):
        if self.flv == "seqhis":
            psel = self.all_seqhis_ana.seq_or(labs)
        elif self.flv == "seqmat":
            psel = self.all_seqmat_ana.seq_or(labs)
        pass
        return psel

    def make_selection_(self, labels):
        if not self.rec or len(labels) == 0:
            log.info("skip make_selection_ as no labels")
            return None
   
        log.debug("%s.make_selection_ labels %s " % (self.nom,repr(labels)))
        self._labels = labels
        if len(labels) == 1 and labels[0][-3:] == ' ..':
            log.debug("make_selection_ wildcard startswith %s " % labels[0] ) 
            lab = labels[0][:-3]  # exclude the " .."
            psel = self.make_psel_startswith(lab)
        elif len(labels) == 1 and labels[0] == 'PFLAGS_DEBUG':
            psel = self.pflags2 != self.pflags
        else:
            psel = self.make_psel_or(labels)
        pass
        return psel

    def _get_label0(self):
        nlab = len(self._labels) 
        if nlab == 1:
            lab0 = self._labels[0]
        else:
            lab0 = None
        return lab0
    label0 = property(_get_label0)
   

    def iflg(self, flg):
       """
       :param flg: eg SC
       :return iflg: zero based index of the flg within the label

       For example iflg('SC') with label 'TO BT BT SC ..' would return 3
       if a non-single line selection were active or the flg did not 
       appear None is returned.

       """ 
       lab0 = self.label0
       if lab0 is None:
           log.fatal("Evt.index0 requires single line selection active, eg sel = slice(0,1) " )
           return None
       pass
       flgs = lab0.split()
       if flgs.count(flg) == 0:
           log.fatal("Evt.index0 expects label0 %s containg flg %s " % (lab0, flg)) 
           return None
       pass
       iflg = flgs.index(flg) 
       return iflg

 
    def _get_nrec(self):
        """
        :return: number of records

        with a single label selection such as "TO BT AB" nrec would return  3
        If there is no single label selection return -1  
        """ 
        lab0 = self.label0
        if lab0 is None:
            return -1
        elab = lab0.split()
        return len(elab)
    nrec = property(_get_nrec)

    def _get_alabels(self):
        """
        :return alabels: all labels of current flv 

        NB cannot directly use with AB comparisons as the labels 
        will be somewhat different for each evt 
        """
        if self.flv == "seqhis":
            alabels = self.ahis.labels
        elif self.flv == "seqmat":
            alabels = self.amat.labels
        else:
            alabels = []
        pass
        return alabels
    alabels = property(_get_alabels)


    def nrecs(self, start=0, stop=None, step=1):
        sli = slice(start, stop, step)
        labels = self.alabels[sli] 
        nrs = np.zeros(len(labels), dtype=np.int32) 
        for ilab, lab in enumerate(labels):
            nrs[ilab] = len(lab.split())
        pass
        return nrs

    def totrec(self, start=0, stop=None, step=1):
        nrs = self.nrecs(start, stop, step)
        return int(nrs.sum())


    def _get_recs(self):
        nr = self.nrec
        if nr == -1:
            log.warning("recs slicing only works when a single label selection is active ")
            return None
        pass
        return slice(0,nr)
    recs = property(_get_recs)

    def make_selection(self, sel, not_):
        if sel is None:
            return None

        labels = self.make_labels(sel)
        psel = self.make_selection_(labels)
        if not_:
            psel = np.logical_not(psel)
        pass
        return psel


    def _init_selection(self, psel):
        """
        :param psel: photon length boolean selection array, make it with make_selection or directy with numpy 
        """
        # for first _init_selection hold on to the originals
        if self._psel is None:
            self.ox_ = self.ox
            self.c4_ = self.c4
            self.wl_ = self.wl
            self.rx_ = self.rx
        pass
        if psel is None: 
            if hasattr(self, 'ox_'):
                log.debug("_init_selection with psel None : resetting selection to original ")
                self.ox = self.ox_
                self.c4 = self.c4_
                self.wl = self.wl_
                self.rx = self.rx_
                self.seqhis_ana = SeqAna(self.seqhis, self.histype)   
                self.seqmat_ana = SeqAna(self.seqmat, self.mattype)   
                self.nsel = len(self.ox_)
            else:
                log.warning("_init_selection with psel None : no prior selection, ignoring ")
            return  

        log.debug("psel %s " % repr(psel))

        self._psel = psel 
        nsel = len(psel[psel == True])
        if nsel == 0:
            log.warning("_init_selection EMPTY nsel %s len(psel) %s " % (nsel, len(psel)))
        else:
            log.debug("_init_selection nsel %s len(psel) %s  " % (nsel, len(psel)))
        pass 
        self.nsel = nsel 

        ## always basing new selection of the originals
        self.ox = self.ox_[psel]
        self.c4 = self.c4_[psel]
        self.wl = self.wl_[psel]
        self.rx = self.rx_[psel]

        self.seqhis_ana = SeqAna(self.seqhis[psel], self.histype)   # sequence history with selection applied
        self.seqmat_ana = SeqAna(self.seqmat[psel], self.mattype)   # sequence history with selection applied


    def _get_reclab(self):
        """
        Sequence label with single record highlighted with a bracket 
        eg  TO BT [BR] BR BT SA 

        """
        nlab = len(self._labels) 
        if nlab == 1 and self._irec > -1: 
            lab = self._labels[0]
            elab= lab.split()
            if self._irec < len(elab):
                elab[self._irec] = "[%s]" % elab[self._irec]
            pass
            lab = " ".join(elab) 
        else:
            lab = ",".join(self._labels) 
        pass
        return lab 
    reclab = property(_get_reclab)


    # *irec* is convenience pointer the current record of interest 
    def _get_irec(self):
        return self._irec
    def _set_irec(self, irec):
        ## hmm could convert a string "SC" into the first occurence integer ?

        nr = self.nrec
        if nr > -1 and irec < 0:
            irec += nrec 

        self._irec = irec
    irec = property(_get_irec, _set_irec)
 

    # *psel* provides low level selection control via  boolean array 
    def _get_psel(self):
        return self._psel
    def _set_psel(self, psel):
        self._init_selection(psel)
    psel = property(_get_psel, _set_psel)
 

    # *sel* provides high level selection control using slices, labels, hexint etc
    def _get_sel(self):
        return self._sel
    def _set_sel(self, sel):
        log.debug("Evt._set_sel %s " % repr(sel))
        self._sel = sel
        psel = self.make_selection(sel, False)
        self._init_selection(psel)
    sel = property(_get_sel, _set_sel)
      
 
    def psel_dindex(self):
        """
        Return dindex option string allowing debug dumping during OKG4 running,
        see for example tconcentric-tt-pflags  
        """ 
        return "--dindex=%s" % ",".join(map(str,np.where(self.psel)[0]))


    def select_(self, label="TO RE BT BT BT BT SA"):
        """
        :param label: seqhis or seqmat label
        :return boolean selection array:
        """
        if label == "BAD_PFLAGS":
            select = self.pflags2 != self.pflags
        else:
            if label[0:2] in "TO CK SI GN NL".split():
                code = self.histype.code(label)
                select = self.seqhis == code
            else:
                code = self.mattype.code(label)
                select = self.seqmat == code
            pass
        pass
        return select 

    def dindex_(self, label="TO RE BT BT BT BT SA", limit=None, reverse=False):
        """
        :param label: seqhis or seqmat label
        :param limit:
        :return array: list of photon record_id that match the seqhis label 
        """
        select = self.select_(label)
        a = np.where(select)[0]
        if reverse:
            a = a[::-1] 
        return a[:limit]

    def dindex(self, label="TO RE BT BT BT BT SA", limit=10, reverse=False ):
        """
        :param label: seqhis label
        :param limit:
        :return string: dindex option string with record_id of photons with the selected history 

        Find record_id of photons with particular histories::

            In [27]: e1.dindex("TO AB")
            Out[27]: '--dindex=13,14,24,26,28,34,53,83,84,98'

            In [28]: e1.dindex("TO SC AB")
            Out[28]: '--dindex=496,609,698,926,1300,1346,1356,1633,1637,2376'

            In [29]: e1.dindex("TO RE SC AB")
            Out[29]: '--dindex=1472,10272,12785,17200,22028,24184,31503,32334,43509,44892'

            In [31]: e1.dindex("TO RE BT BT BT BT SA")
            Out[31]: '--dindex=63,115,124,200,225,270,307,338,342,423'

            In [36]: e1.dindex("BAD_PFLAGS")
            Out[36]: '--dindex=3352,12902,22877,23065,41882,60653,68073,69957,93373,114425'

            In [37]: e1.dindex("BAD_PFLAGS",reverse=True)
            Out[37]: '--dindex=994454,978573,976708,967547,961725,929984,929891,925473,919938,917897'

        """
        return "--dindex=%s" % ",".join(map(str,self.dindex_(label, limit, reverse))) 

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
        log.info("init_index START")
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
            rsr = rs.reshape(-1, self.maxrec, 1, 4)        
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
        pass

        log.info("init_index DONE")

 
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
            if t is not None:   # ht are often missing 
                sst.add(t)
            pass
            lines.append("%10s  %s  %s " % (name, p, t ))

        nstamp = len(sst)
        if nstamp > 1:
            log.warning("MIXED TIMESTAMP EVENT DETECTED")
            print "\n".join(lines)

        return nstamp


    def _brief(self):
        if self.valid:
             return "%s %s %s %s" % (self.label, self.stamp, pdict_(self.idomd), self.path)
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
    def compare_ana(cls, a, b, ana_ , lmx=20, c2max=None, cf=True, zero=False, cmx=0, pr=False):
        if not (a.valid and b.valid):
            log.fatal("need two valid events to compare ")
            sys.exit(1)
        pass

        a_ana = getattr(a, ana_, None)
        b_ana = getattr(b, ana_, None)

        if a_ana is None:
            log.warning("missing a_ana %s " % ana_ )  
            return None
        if b_ana is None:
            log.warning("missing b_ana  %s " % ana_ )  
            return None
             
        a_tab = a_ana.table
        b_tab = b_ana.table
        c_tab = None

        if cf:
            c_tab = a_tab.compare(b_tab)
            c_tab.title = ana_

            if len(c_tab.lines) > lmx:
                c_tab.sli = slice(0,lmx)

            if pr:
                print c_tab
            if c2max is not None:
                assert c_tab.c2p < c2max, "c2p deviation for table %s c_tab.c2p %s >= c2max %s " % ( ana_, c_tab.c2p, c2max )

        else:
            a_tab.title = "A:%s " % ana_
            if len(a_tab.lines) > lmx:
                a_tab.sli = slice(0,lmx)
            if pr:
                print a_tab

            b_tab.title = "B:%s " % ana_
            if len(b_tab.lines) > lmx:
                b_tab.sli = slice(0,lmx)
            if pr:
                print b_tab
        pass
        return c_tab


    @classmethod
    def compare_table(cls, a, b, analist='seqhis_ana seqmat_ana'.split(), lmx=20, c2max=None, cf=True, zero=False, cmx=0, pr=False):
        cft = {}
        for ana_ in analist:
            c_tab = cls.compare_ana( a, b, ana_ , lmx=lmx, c2max=c2max, cf=cf, zero=zero, cmx=cmx, pr=pr)
            cft[ana_] = c_tab 
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


    def rw(self):
        recs = self.recs
        if recs is None:
             return None  

        return self.recwavelength(recs)

    def recwavelength(self, recs):
        boundary_domain = self.fdom[2,0]

        pzwl = self.rx[:,recs,1,1]

        nwavelength = (pzwl & np.uint16(0xFF00)) >> 8

        p_wavelength = nwavelength.astype(np.float32)*boundary_domain[W]/255.0 + boundary_domain[X]

        return p_wavelength 


    def rpol(self):
        recs = self.recs
        if recs is None:
             return None  
        return self.rpolw_(recs)[:,:,:3]

    def rpol_(self, fr):
        return self.rpolw_(fr)[:,:3]

    def rpolw_(self, recs):
        """
        Unlike rpol_ this works with irec slices, 
        BUT note that the wavelength returned in 4th column is 
        not decompressed correctly.
        Due to shape shifting it is not easy to remove
        """
        return self.rx[:,recs,1,0:2].copy().view(np.uint8).astype(np.float32)/127.-1.

    def rpol_old_(self, recs):
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
        pxpy = self.rx[:,recs,1,0]
        pzwl = self.rx[:,recs,1,1]
        m1m2 = self.rx[:,recs,1,2]
        bdfl = self.rx[:,recs,1,3]

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

    def recflags(self, recs):
        m1m2 = self.rx[:,recs,1,2]
        bdfl = self.rx[:,recs,1,3]

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

    def rflgs_(self, recs):
        return self.recflags(recs) 

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
        """
        compression bins
        """
        p_center = self.fdom[0,0,:W]
        p_extent = self.fdom[0,0,W]
        log.debug(" pbins p_center %s p_extent %s " % (repr(p_center), repr(p_extent)))
        #assert p_center[0] == p_center[1] == p_center[2] == 0., p_center
        pb = np.linspace(p_center[0] - p_extent, p_center[1] + p_extent, 2*32767+1)
        return pb 

    def rpost(self):
        recs = self.recs
        if recs is None:
            log.warning("this only works on evt with single line seqs")
            return None
        return self.rpost_(recs)

    def rdir(self, fr=0, to=1, nrm=True):
        """
        :param fr:
        :param to:
        :param nrm:

        Vector between points on the propagation identified by "fr" and "to" 
        zero based point indices.
        """
        fr_ = self.rpost_(fr)
        to_ = self.rpost_(to)
        step = to_[:,:3] - fr_[:,:3] 

        if nrm:
            dir_ = norm_(step)
        else:
            dir_ = step 
        pass
        return dir_

    def rpost_(self, recs):
        """
        NB recs can be a slice, eg slice(0,5)

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
        center, extent = self.post_center_extent()
        p = self.rx[:,recs,0].astype(np.float32)*extent/32767.0 + center 
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


def deviation_plt(evt):
    dv = evt.a_deviation_angle(axis=X)
    if plt:
        plt.close()
        plt.ion()
    
        plt.hist(dv/deg, bins=360, log=True, histtype="step") 
        plt.show()


if __name__ == '__main__':
    ok = opticks_main()

    seq = "TO BT BT BT BT DR BT BT BT BT BT BT BT BT SA"
    #seq = "PFLAGS_DEBUG"

    a = Evt(tag="%s"%ok.utag, src=ok.src, det=ok.det, args=ok)
    print a.seqhis_ana.table[0:20]

    #b = Evt(tag="-%s"%ok.utag, src=ok.src, det=ok.det, args=ok, seqs=[seq])
    #b.history_table(slice(0,20))





  

