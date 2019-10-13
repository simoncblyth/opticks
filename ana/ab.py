#!/usr/bin/env python
#
# Copyright (c) 2019 Opticks Team. All Rights Reserved.
#
# This file is part of Opticks
# (see https://bitbucket.org/simoncblyth/opticks).
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and 
# limitations under the License.
#

"""

"""
from __future__ import print_function
import os, sys, logging, numpy as np
from collections import OrderedDict as odict

from opticks.ana.ctx import Ctx
from opticks.ana.cfh import CFH
from opticks.ana.base import json_save_
from opticks.ana.nbase import chi2, vnorm
from opticks.ana.nload import np_load
from opticks.ana.decompression import decompression_bins
from opticks.ana.histype import HisType
from opticks.ana.mattype import MatType
from opticks.ana.evt import Evt
from opticks.ana.abstat import ABStat
#from opticks.ana.dv import Dv, DvTab
from opticks.ana.qdv import QDV, QDVTab
from opticks.ana.make_rst_table import recarray_as_rst
from opticks.ana.metadata import CompareMetadata
from opticks.ana.abprofile import ABProfile
from opticks.ana.absmry import ABSmry

log = logging.getLogger(__name__)



ratio_ = lambda num,den:float(num)/float(den) if den != 0 else -1 


class Maligned(object):
    def __init__(self, ab):
        self.ab = ab 

        tot = len(ab.a.seqhis)
        self.tot = tot
        self.maligned = ab.maligned
        self.aligned = ab.aligned
        self.nmal = len(ab.maligned)
        self.nmut = len(ab.misutailed) 

        self.fmaligned = ratio_(self.nmal, tot)
        self.faligned = ratio_(len(ab.aligned), tot)
        self.sli = slice(0,25)

        self.migs = self.count_migtab() 

    def __getitem__(self, sli):
         self.sli = sli
         return self

    def lines(self):
        return ["aligned  %7d/%7d : %.4f : %s " % ( len(self.aligned), self.tot, self.faligned,   ",".join(map(lambda _:"%d"%_, self.aligned[self.sli])) ),
                "maligned %7d/%7d : %.4f : %s " % ( len(self.maligned), self.tot, self.fmaligned,  ",".join(map(lambda _:"%d"%_, self.maligned[self.sli])) ) ]

    def label(self, q ):
        return "%16x %30s " % ( q, self.ab.histype.label(q) )

    def count_migtab(self, dump=False):
        """
        Count occurrences of different history migration pairs 
        """
        ab = self.ab 
        mal = ab.maligned
        a = ab.a
        b = ab.b
        d = odict()
        for i,m in enumerate(mal):
            k = (a.seqhis[m],b.seqhis[m])
            if not k in d: d[k]=0 
            d[k] += 1  
            if dump:
                print( " %4d   :  %s  %s " % (i, self.label(a.seqhis[m]), self.label(b.seqhis[m])) )
            pass
        pass
        return d 

    def migtab(self):
        tab = "\n".join([" %4d   %s  %s " % ( kv[1], self.label(kv[0][0]), self.label(kv[0][1]) )  for kv in sorted(self.migs.items(), key=lambda kv:kv[1], reverse=True)])
        return tab

    def __repr__(self):
        return "\n".join( ["ab.mal"] + self.lines() + [repr(self.sli)]
                   + map(lambda iq:self.ab.recline(iq), enumerate(self.maligned[self.sli])) + ["ab.mal.migtab"]
                   + [self.migtab(), "."]
                ) 



class RC(object):
    offset = { "rpost_dv":0, "rpol_dv":1 , "ox_dv":2 } 

    def __init__(self, ab):
        rc = {}
        log.info("[ rpost_dv ")
        rc["rpost_dv"] = ab.rpost_dv.RC
        log.info("]")
        
        log.info("[ rpol_dv ")
        rc["rpol_dv"] = ab.rpol_dv.RC 
        log.info("]")

        log.info("[ ox_dv ")
        rc["ox_dv"] = ab.ox_dv.RC 
        log.info("]")


        assert max(rc.values()) <= 1 
        irc = rc["rpost_dv"] << self.offset["rpost_dv"] | rc["rpol_dv"] << self.offset["rpol_dv"] | rc["ox_dv"] << self.offset["ox_dv"]
        self.rc = irc 

        levels = map(lambda _:_.level, filter(None, [ab.rpost_dv.maxlevel,ab.ox_dv.maxlevel,ab.ox_dv.maxlevel]))
        self.level = max(levels) if len(levels) > 0 else None

    def __repr__(self):
        return "RC 0x%.2x" % self.rc




class AB(object):
    """
    AB : Event Pair comparison
    =============================

    Selection examples::

         ab.sel = ".6ccd"             
         ab.sel = "TO BT BT SC .."     # wildcard selection same as above
         ab.sel = None                 # back to default no selection

         ab.aselhis = "TO BT BT SA"    # with align-ment enabled 
                                       # (used for non-indep photon samples, such as from emitconfig)

    Subsequently check tables with::

         ab.his
         ab.flg
         ab.mat

    Histo persisting, provides random access to histos for debugging::

         ab.sel = slice(0,1)
         ab.qwn = "Z"
         ab.irec = 5

         h = ab.h      # qwn and irec used in creation of histogram, which is persisted

    For example, checking a hi chi2 contributor::

        In [12]: h = ab.h
        [2016-11-14 12:21:43,098] p12883 {/Users/blyth/opticks/ana/ab.py:473} INFO - AB.rhist qwn T irec 5 
        [2016-11-14 12:21:43,098] p12883 {/Users/blyth/opticks/ana/cfh.py:344} INFO - CFH.load from /tmp/blyth/opticks/CFH/concentric/1/TO_BT_BT_BT_BT_AB/5/T 

        In [13]: h.lhabc
        Out[13]: 
        array([[     2.9649,      6.9934,      0.    ,      0.    ,      0.    ],
               [     6.9934,     11.0218,      0.    ,      0.    ,      0.    ],
               [    11.0218,     15.0503,      0.    ,      0.    ,      0.    ],
               [    15.0503,     19.0787,      0.    ,      0.    ,      0.    ],
               [    19.0787,     23.1072,  14159.    ,  13454.    ,     17.9997],
               [    23.1072,     27.1356,  14796.    ,  15195.    ,      5.3083],
               [    27.1356,     31.164 ,      0.    ,      0.    ,      0.    ],
               [    31.164 ,     35.1925,      0.    ,      0.    ,      0.    ],
               [    35.1925,     39.2209,      0.    ,      0.    ,      0.    ],
               [    39.2209,     43.2494,      0.    ,      0.    ,      0.    ]], dtype=float32)

    """
    C2CUT = 30
    STREAM = sys.stderr

    @classmethod  
    def print_(cls, obj):
        print(str(obj), file=cls.STREAM)

    def _get_maligned(self):
        return np.where(self.a.seqhis != self.b.seqhis)[0]
    maligned = property(_get_maligned)

    def _get_aligned(self):
        return np.where(self.a.seqhis == self.b.seqhis)[0]
    aligned = property(_get_aligned)
    
    def _get_misutailed(self):
        return np.where(self.a.utail != self.b.utail)[0]
    misutailed = property(_get_misutailed)

    def _get_utailed(self):
        return np.where(self.a.utail == self.b.utail)[0]
    utailed = property(_get_utailed)


    def recline(self, iq, u=False):
        """
        :param iq: i,q 2-tuple of i:enumeration index 0,1,2,... and q:photon index
        :return line: comparing a and b seqhis labels  
        """
        i, q = iq
        al = self.histype.label(self.a.seqhis[q])
        bl = self.histype.label(self.b.seqhis[q])
        mk = " " if al == bl else "*"
        head = " %6d %6d : %s : %50s %50s " % ( i, q, mk, al, bl )
        if u==True: 
            u = self.u 
            uu = u[q].ravel()        # randoms for photon record_id q
            ua = self.a.utail[q]
            ub = self.b.utail[q]
            wa = np.where( uu == ua )[0][0]
            wb = -1 if ub == 0 else np.where( uu == ub )[0][0]
            wd = 0 if wb == -1 else wb - wa 
            tail = " ua %6.4f ub %6.4f wa %2d wb %2d wd %2d " % ( ua, ub, wa, wb, wd ) 
        else:
            tail = None
        pass
        return " ".join(filter(None, [head, tail]))

    def dumpline(self, wh, u=False):
        if type(wh) is slice:
            start = wh.start if wh.start is not None else 0
            stop = wh.stop if wh.stop is not None else 1
            step = wh.step if wh.step is not None else 1
            wh = range(start,stop, step) 
        pass
        self.print_( "\n".join( map(lambda iq:self.recline(iq, u=u), enumerate(wh))))


    def check_utaildebug(self):
        """
        notes/issues/ts-box-utaildebug-decouple-maligned-from-deviant.rst

        Looking at utail mismatches that do not correspond to seqhis mismatches. 

        Hmm can use this approach to examine the random consumption of each photon 


        Hmm for analysis of events created with a mask active the q will not 
        correspond.
        """

        if self.cfm.utaildebug == 0:
            log.info("requires both A and B to have been run with --utaildebug option")
            return 
        pass 

        mask = self.ok.mask
        if not mask is None:
            qmask = map(int, mask.split(","))
            assert len(qmask) == len(self.a.seqhis)
        else:
            qmask = None
        pass 

        u = self.u 
        w = np.where(np.logical_and( self.a.utail != self.b.utail, self.a.seqhis == self.b.seqhis ))[0]
        log.info("utail mismatch but seqhis matched u.shape:%r w.shape: %r " % (u.shape, w.shape) )
        for i,q in enumerate(w):
            q_ = q if qmask is None else qmask[q]            
            uu = u[q_].ravel()        # randoms for photon record_id q 
            ua = self.a.utail[q]
            ub = self.b.utail[q]
            wa = np.where( uu == ua )[0][0]
            wb = -1 if ub == 0 else np.where( uu == ub )[0][0]
            wd = 0 if wb == -1 else wb - wa 
            print(" i %5d q_ %7d ua %10.4f ub %10.4f  wa %3d wb %3d wd %3d : %s  " % (i, q_, ua, ub, wa, wb, wd, self.recline((i,q))  ))
        pass



    def dump(self):
        log.debug("[")
        self.print_(self.pro)  
        if self.is_comparable: 
            self.print_(self.cfm)  
            self.print_(self.mal)
            self.print_(self)
            self.print_(self.cfm)  
            self.print_(self.brief)
            self.print_(self.rpost_dv)
            self.print_(self.rpol_dv)
            self.print_(self.ox_dv)
            self.print_(self.brief)
            self.print_(self.rc_)
            self.print_(self.cfm)  
        pass
        log.debug("]")


    def __init__(self, ok, overridetag=0):
        """
        :param ok: arguments instance from opticks_main
        :param overridetag: when non-zero overrides the tags of events to be loaded NOT YET IMPLEMENTED
        """
        log.debug("[")
        self.ok = ok
        self.overridetag = overridetag
        self.histype = HisType()
        self.tabs = []
        self.dvtabs = []
        self.stream = sys.stderr
        self._sel = None
        self.dshape = None

        self.load()
        self.load_u()

        self.is_comparable = self.valid and not self.a.ph.missing and not self.b.ph.missing

        self.pro = ABProfile(self.a.tagdir, self.b.tagdir)

        if self.is_comparable: 
            self.cfm = self.compare_meta()
            self.mal = self.check_alignment()
            self.compare_shapes()
            self.compare_domains()
            self.compare()
            self.init_point()
            self.smry = ABSmry.Make(self)
            self.smry.save()  
        pass 
        log.debug("]")

    def load(self):
        """
        It takes aound 4s on Precision Workstation to load 1M full AB evt pair. 
        So avoid needing to duplicate that.
        """
        log.info("[ %s np.__version__ %s " % (self.ok.smry, np.__version__) )
        args = self.ok
 
        if args.utag is None:
            assert len(args.utags) == 2, ( "expecting 2 utags ", args.utags )
            atag = "%s" % args.utags[0]
            btag = "%s" % args.utags[1]
        else:
            atag = "%s" % args.utag
            btag = "-%s" % args.utag
        pass

        a = Evt(tag=atag, src=args.src, det=args.det, pfx=args.pfx, args=args, nom="A", smry=args.smry)
        b = Evt(tag=btag, src=args.src, det=args.det, pfx=args.pfx, args=args, nom="B", smry=args.smry)

        pass
        self.a = a
        self.b = b 

        valid = a.valid and b.valid 
        if not valid:
            log.fatal("invalid loads : a.valid:%s b.valid:%s " % (a.valid, b.valid))
        pass
        self.valid = valid 

        self.align = None
        self._dirty = False

        ## property setters
        if valid:
            self.sel = None
            self.irec = 0
            self.qwn = "X"
        pass
        log.info("] ")


    def _get_brief(self):
        abn = "AB(%s,%s,%s)  %s %s    %s " % (self.ok.tag, self.ok.src, self.ok.det, self.sel, self.irec, self.dshape )
        return abn
    brief = property(_get_brief)

    def __repr__(self):
        abn = self.brief
        abr = "A %s " % self.a.brief 
        bbr = "B %s " % self.b.brief 

        amd = ",".join(self.a.metadata.csgbnd)
        acsgp = self.a.metadata.TestCSGPath

        if self.b.valid:
            bmd = ",".join(self.b.metadata.csgbnd)
            bcsgp = self.b.metadata.TestCSGPath
            assert amd == bmd
            assert acsgp == bcsgp
        else:
            pass
        pass
        return "\n".join(filter(None,["ab", abn, abr,bbr, amd, acsgp,"." ]))

    def __str__(self):
        lmx = self.ok.lmx
        if len(self.his.lines) > lmx:
            self.his.sli = slice(0,lmx)
        if len(self.mat.lines) > lmx:
            self.mat.sli = slice(0,lmx)
        pass
        return "\n".join(map(repr, [self,self.ahis,self.flg,self.mat]))


    def load_u(self):
        """
        This array of randoms is used by below check_utaildebug.

        Default u array is (100k,16,16) containing doubles. 
        To generate an extra large array (1M,16,16)::

            TRngBuf_NI=1000000 TRngBufTest 

        Which creates a 2GB file::  

            In [6]: 16*16*1000000*8/1e9
            Out[6]: 2.048

            [blyth@localhost sysrap]$ ls -l "$TMP/TRngBufTest_0_1000000.npy"
            -rw-rw-r--. 1 blyth blyth 2048000096 Jul 28 12:02 /home/blyth/local/opticks/tmp/TRngBufTest_0_1000000.npy

        """
        upath0 = "$TMP/TRngBufTest_0.npy"
        upath1 = "$TMP/TRngBufTest_0_1000000.npy"

        if os.path.exists(os.path.expandvars(upath1)):
            upath = upath1 
            log.info("using the extra large random array %s " % upath ) 
        else: 
            upath = upath0
        pass
        u = np_load(upath)
        u = None if u is None else u.astype(np.float32)
        self.u = u 


    def compare_domains(self):
        assert np.all( self.a.fdom == self.b.fdom )
        self.fdom = self.a.fdom

        isli = slice(2,None)  ## permit rngmax difference 
        assert np.all( self.a.idom[isli] == self.b.idom[isli] )
        self.idom = self.a.idom

    def compare_shapes(self):
        assert self.a.dshape == self.b.dshape, (self.a.dshape, self.b.dshape)   
        self.dshape = self.a.dshape  

        # description of the load_slice
        assert self.a.dsli == self.b.dsli, ( self.a.dsli, self.b.dsli )
        self.dsli = self.a.dsli 


 
    def compare_meta(self):
        cfm = CompareMetadata(self.a.metadata, self.b.metadata)

        non_aligned_skips = "SC AB RE" 
        #non_aligned_skips = "RE" 

        self.dvskips = "" if cfm.align == 1 else non_aligned_skips   

        return cfm 

    def check_alignment(self):
        log.debug("[")
        mal = Maligned(self)
        log.debug("]")
        return mal 
        
    def compare(self):
        """
        * Taking around 5s for 1M events, all in RC 
        * prohis and promat are progressively masked tables, not enabled by default
        """
        log.info("[")

        self.ahis = self._get_cf("all_seqhis_ana", "ab.ahis")
        self.amat = self._get_cf("all_seqmat_ana", "ab.amat")

        if self.ok.prohis:self.prohis()
        if self.ok.promat:self.promat()

        self.rc_ = RC(self) 

        log.info("]")

    def _get_RC(self):
        return self.rc_.rc if self.is_comparable else 255
    RC = property(_get_RC) 

    def _get_level(self):
        return self.rc_.level if self.is_comparable else 255
    level = property(_get_level) 


    numPhotons = property(lambda self:self.cfm.numPhotons) 


    def init_point(self):
        log.debug("[")
        self.point = self.make_point()
        log.debug("]")

    def point_dtype(self):
        dtype=[
                ('irl', int), 
                ('isel', int), 
                ('irec', int), 
                ('nrec', int), 
                ('reclab', "|S64")
              ]
        return dtype

    def _get_recpoint(self):
        """
        Returns None when single line selection is not active
        otherwise returns selected recpoint

        ::

            In [2]: ab.sel = "TO BT BT RE BT BT [SA]"

            In [3]: ab.recpoint
            Out[3]: (85, 12, 6, 7, 'TO BT BT RE BT BT [SA]')

            In [4]: ab.recpoint.isel
            Out[4]: 12

            In [5]: ab.recpoint.nrec
            Out[5]: 7

            In [6]: ab.recpoint.irec
            Out[6]: 6

        """
        rl = self.reclab
        if rl is None:
            return None
        pass 
        rp = self.point[self.point.reclab == rl]
        assert len(rp) == 1, rp
        return rp[0]
    recpoint = property(_get_recpoint)

    def make_point(self):
        """
        :return point: recarray for holding point level metadata

        ::

            In [5]: print "\n".join(map(repr, ab.point[:8]))
            (0, 0, 0, '[TO] BT BT BT BT SA')
            (1, 0, 1, 'TO [BT] BT BT BT SA')
            (2, 0, 2, 'TO BT [BT] BT BT SA')
            (3, 0, 3, 'TO BT BT [BT] BT SA')
            (4, 0, 4, 'TO BT BT BT [BT] SA')
            (5, 0, 5, 'TO BT BT BT BT [SA]')
            (6, 1, 0, '[TO] AB')
            (7, 1, 1, 'TO [AB]')

            In [6]: print "\n".join(map(repr, ab.point[-8:]))
            (64257, 4847, 8, 'TO RE RE RE RE BT BT BT [SC] BR BR BR BR BR BR BR')
            (64258, 4847, 9, 'TO RE RE RE RE BT BT BT SC [BR] BR BR BR BR BR BR')
            (64259, 4847, 10, 'TO RE RE RE RE BT BT BT SC BR [BR] BR BR BR BR BR')
            (64260, 4847, 11, 'TO RE RE RE RE BT BT BT SC BR BR [BR] BR BR BR BR')
            (64261, 4847, 12, 'TO RE RE RE RE BT BT BT SC BR BR BR [BR] BR BR BR')
            (64262, 4847, 13, 'TO RE RE RE RE BT BT BT SC BR BR BR BR [BR] BR BR')
            (64263, 4847, 14, 'TO RE RE RE RE BT BT BT SC BR BR BR BR BR [BR] BR')
            (64264, 4847, 15, 'TO RE RE RE RE BT BT BT SC BR BR BR BR BR BR [BR]')

        """ 
        rls = self.reclabs(0,None)
        nrls = len(rls)

        point = np.recarray((nrls,), dtype=self.point_dtype())
        point.reclab = rls
        point.irl = np.arange(0, nrls)

        offset = 0 
        for isel,clab in enumerate(self.clabels[0:None]):
            sls = Ctx.reclabs_(clab)
            nsls = len(sls)
            point.isel[offset:offset+nsls] = np.repeat( isel, nsls)
            point.irec[offset:offset+nsls] = np.arange( 0   , nsls)
            point.nrec[offset:offset+nsls] = np.repeat( nsls, nsls)
            offset += nsls 
        pass
        return point



    def prohis(self, rng=range(1,8)):
        log.info("[")
        for imsk in rng:
            setattr(self, "his_%d" % imsk, self.cf("seqhis_ana_%d" % imsk)) 
        pass
        log.info("]")
    def promat(self, rng=range(1,8)):
        log.info("[")
        for imsk in rng:
            setattr(self, "mat_%d" % imsk, self.cf("seqmat_ana_%d" % imsk)) 
        pass
        log.info("]")

    def tabname(self, ana):
        if ana.endswith("_dv"):
            tn = ana + "tab"
        else:
            tn = ana.replace("_ana", "_tab")
        pass
        return tn

    def _make_cf(self, ana, shortname):
        """
        all_ tables have no selection applied so they are not dirtied by changing selection
        """
        ordering = self.ok.cfordering 
        assert ordering in ["max","self","other"] 
        c_tab = Evt.compare_ana( self.a, self.b, ana, lmx=self.ok.lmx, cmx=self.ok.cmx, c2max=None, cf=True, ordering=ordering, shortname=shortname )
        if not ana[0:3] == "all":
            self.tabs.append(c_tab)
        pass 
        tabname = self.tabname(ana)
        setattr(self, tabname, c_tab)
        return c_tab 



    



    def _make_dv(self, ana):
        """
        :param ana: ox_dv, rpost_dv, rpol_dv
        """  
        log.debug("[ %s " % ana )
        we = self.warn_empty 
        self.warn_empty = False
        seqtab = self.ahis

        #dv_tab = DvTab(ana, seqtab, self, skips=self.dvskips, selbase="ALIGN" ) 

        use_utaildebug = self.cfm.utaildebug   
        log.info("_make_dv ana %s use_utaildebug %s " % (ana, use_utaildebug)) 
        dv_tab = QDVTab(ana, self, use_utaildebug=use_utaildebug ) 
        self.dvtabs.append(dv_tab)

        self.warn_empty = we
        log.debug("] %s " % ana )
        return dv_tab 

    def _get_dv(self, ana):
        log.debug("[ %s " % ana )
        tabname = self.tabname(ana)
        dv_tab = getattr(self, tabname, None)
        if dv_tab is None:
            dv_tab = self._make_dv(ana) 
        elif dv_tab.dirty:
            dv_tab = self._make_dv(ana) 
        else:
            pass 
        pass
        setattr(self, tabname, dv_tab) 
        log.debug("] %s " % ana )
        return dv_tab


    def _get_ox_dv(self):
        return self._get_dv("ox_dv")
    ox_dv = property(_get_ox_dv)

    def _get_rpost_dv(self):
        return self._get_dv("rpost_dv")
    rpost_dv = property(_get_rpost_dv)

    def _get_rpol_dv(self):
        return self._get_dv("rpol_dv")
    rpol_dv = property(_get_rpol_dv)


    def _set_dirty(self, dirty):
        for tab in self.tabs:
            tab.dirty = dirty
        pass
    def _get_dirty(self):
        dtabs = filter(lambda tab:tab.dirty, self.tabs)
        return len(dtabs) > 0 
    dirty = property(_get_dirty, _set_dirty)




    mxs = property(lambda self:dict(map(lambda _:[_.name,_.maxdvmax], self.dvtabs )))
    mxs_max = property(lambda self:max(self.mxs.values()))

    rmxs = property(lambda self:dict(map(lambda _:[_.name,_.maxdvmax], filter(lambda _:_.name[0] == 'r',self.dvtabs ))))
    rmxs_max = property(lambda self:max(self.rmxs.values()))

    pmxs = property(lambda self:dict(map(lambda _:[_.name,_.maxdvmax], filter(lambda _:_.name[0] != 'r',self.dvtabs ))))
    pmxs_max = property(lambda self:max(self.pmxs.values()))



    fds = property(lambda self:dict(map(lambda _:[_.name,_.fdiscmax], self.dvtabs )))
    fds_max = property(lambda self:max(self.fds.values()))


    c2p = property(lambda self:dict(map(lambda _:[_.title,_.c2p], self.tabs )))
    c2p_max = property(lambda self:max(self.c2p.values()))


    def _get_cf(self, ana, shortname):
        """
        Changing *sel* property invokes _set_sel 
        resulting in a change to the SeqAna in the A B Evt,
        thus all AB comparison tables are marked dirty, causing 
        them to be recreated at next access.
        """
        log.debug("[ %s " % shortname ) 
        tabname = self.tabname(ana)
        tab = getattr(self, tabname, None)
        if tab is None:
            tab = self._make_cf(ana, shortname) 
        elif tab.dirty:
            tab = self._make_cf(ana, shortname) 
        else:
            pass 
        log.debug("] %s " % shortname ) 
        return tab

    def _get_his(self):
        return self._get_cf("seqhis_ana", "ab.his")
    def _get_mat(self):
        return self._get_cf("seqmat_ana", "ab.mat")
    def _get_flg(self):
        return self._get_cf("pflags_ana", "ab.flg")

    his = property(_get_his)
    mat = property(_get_mat)
    flg = property(_get_flg)

    # high level *sel* selection only, for lower level *psel* selections 
    # apply individually to evt a and b 

    def _get_sel(self):
        return self._sel
    def _set_sel(self, sel, nom="sel"):
        """
        NB slice selection at AB level must be
        converted into seq string selection at evt level
        as the labels can diverge, expecially out in the tail

        slice selection forces into seqhis selection

        ::

            ab.sel = "[TO] BT BT BT BT SA" 
            hh = ab.hh

        """
        log.debug("[ %s " % repr(sel))

        if type(sel) is slice:
            clabels = self.clabels
            seqs = clabels[sel]  
            assert len(seqs) == 1, seqs
            log.debug("AB._set_set convert slice selection %r into common label seq selection  %s " % (sel, seqs[0]))
            sel = seqs[0]
            self.flv = "seqhis"
        pass

        if nom == "sel":
            self.align = None
            self.a.sel = sel
            self.b.sel = sel
        elif nom == "selhis":
            self.align = None
            self.a.selhis = sel
            self.b.selhis = sel
        elif nom == "aselhis":
            self.align = "seqhis"
            self.a.selhis = sel
            self.b.selhis = sel
        elif nom == "selmat":
            self.align = None
            self.a.selmat = sel
            self.b.selmat = sel
        elif nom == "selflg":
            self.align = None
            self.a.selflg = sel
            self.b.selflg = sel
        else:
            assert 0, nom 
        pass

        self._sel = sel 
        self.dirty = True  
        log.debug("] %s " % repr(sel))

    sel = property(_get_sel, _set_sel)
    
    def _set_selmat(self, sel):
        self._set_sel( sel, nom="selmat")
    def _set_selhis(self, sel):
        self._set_sel( sel, nom="selhis")
    def _set_aselhis(self, sel):
        self._set_sel( sel, nom="aselhis")
    def _set_selflg(self, sel):
        self._set_sel( sel, nom="selflg")

    selmat = property(_get_sel, _set_selmat)
    selhis = property(_get_sel, _set_selhis)
    selflg = property(_get_sel, _set_selflg)
    aselhis = property(_get_sel, _set_aselhis)




    def _get_align(self):
        return self._align 
    def _set_align(self, align):
        """
        CAUTION: currently need to set sel after align for it to have an effect
                 unless use aselhis
        """
        if align is None:
            _align = None
        elif type(align) is str:
            if align == "seqhis":
                 _align = self.a.seqhis == self.b.seqhis 
            else:
                 assert False, (align, "not implemented" ) 
        else:
            _align = align
        pass

        self._align = _align
        self.a.align = _align
        self.b.align = _align
        self.dirty = True  

    align = property(_get_align, _set_align)    



    def _set_flv(self, flv):
        self.a.flv = flv
        self.b.flv = flv
    def _get_flv(self):
        a_flv = self.a.flv    
        b_flv = self.b.flv    
        assert a_flv == b_flv
        return a_flv
    flv = property(_get_flv, _set_flv)

    def _set_irec(self, irec):
        self.a.irec = irec 
        self.b.irec = irec 
    def _get_irec(self):
        a_irec = self.a.irec
        b_irec = self.b.irec
        assert a_irec == b_irec
        return a_irec
    irec = property(_get_irec, _set_irec)

    def _get_reclab(self):
        a_reclab = self.a.reclab
        b_reclab = self.b.reclab
        assert a_reclab == b_reclab
        return a_reclab
    reclab = property(_get_reclab)

    def _get_label0(self):
        a_label0 = self.a.label0
        b_label0 = self.b.label0
        assert a_label0 == b_label0
        return a_label0
    label0 = property(_get_label0)

    def _get_seq0(self):
        lab0 = self.label0
        if lab0 is None:
            return None
        return lab0.replace(" ","_")
    seq0 = property(_get_seq0)

    def count(self, line=0, col=1):
        """standard selects have only one sequence line""" 
        cu = self.seq.cu 
        if cu is None:
            return None
        pass
        return cu[line,col]

    def _get_aseq(self):
        """
        aseq is not changed by the current selection 
        """
        flv = self.flv
        if flv == "seqhis":
            return self.ahis
        elif flv == "seqmat":
            return self.amat
        else:
            pass
        return None
    aseq = property(_get_aseq)

    def _get_seq(self):
        """
        seq is changed by current selection
        """
        flv = self.flv
        if flv == "seqhis":
            return self.his
        elif flv == "seqmat":
            return self.mat
        else:
            pass
        return None
    seq = property(_get_seq)

    def _get_achi2(self):
        aseq = self.aseq 
        assert not aseq is None
        return aseq.c2
    achi2 = property(_get_achi2)

    def a_count(self, line=0):
        return self.count(line,1)

    def b_count(self, line=0):
        return self.count(line,2)

    def ab_count(self, line=0):
        ac = self.a_count(line)
        bc = self.b_count(line)
        return "%d/%d" % (ac,bc)

    def _get_suptitle(self):
        abc = self.ab_count()
        title = "%s/%s/%s : %s  :  %s " % (self.ok.tag, self.ok.det, self.ok.src, abc, self.reclab )
        return title
    suptitle = property(_get_suptitle)

    def _get_ctx(self):
        """
        hmm at some point seq0 lost its underscores
        """
        return Ctx({'det':self.ok.det, 'tag':self.ok.tag, 'src':self.ok.src, 'seq0':self.label0, 'lab':self.reclab, 'irec':self.irec, 'qwn':self.qwn })
    ctx = property(_get_ctx)

    pctx = property(lambda self:self.ctx.pctx)
    qctx = property(lambda self:self.ctx.qctx)


    def _get_nrec(self):
        """
        :return: number of record points, when a single seq selection is active
        """ 
        a_nr = self.a.nrec
        b_nr = self.b.nrec
        assert a_nr == b_nr, (a_nr, b_nr, "A and B should have same nrec ?")
        if a_nr == b_nr and a_nr != -1:
            nr = a_nr
        else:
            nr = -1
        pass
        log.debug(" a_nr:%d b_nr:%d nr:%d " % (a_nr, b_nr, nr) )
        return nr 
    nrec = property(_get_nrec)


    def iflg(self, flg):
        """
        :return: number of record points, when a single seq selection is active
        """ 
        a_ifl = self.a.iflg(flg)
        b_ifl = self.b.iflg(flg)
        assert a_ifl == b_ifl, (a_ifl, b_ifl, "A and B should have same iflg ?")
        if a_ifl == b_ifl and a_ifl != None:
            ifl = a_ifl
        else:
            ifl = None
        pass
        log.info(" a_ifl:%s b_ifl:%s ifl:%s " % (a_ifl, b_ifl, ifl) )
        return ifl 

    def nrecs(self, start=0, stop=None, step=1):
        """
        """
        sli = slice(start, stop, step)
        labels = self.clabels[sli] 
        nrs = np.zeros(len(labels), dtype=np.int32) 
        for ilab, lab in enumerate(labels):
            nrs[ilab] = len(lab.split())
        pass
        return nrs

    def stats_dtype(self, qwns=None):
        if qwns is None:
            qwns = self.ok.qwns
        pass
        dtype = [(i,np.int32) for i in "iv is na nb".split()] 
        dtype += [(s,"|S64") for s in "reclab".split()]
        dtype += [(q,np.float32) for q in list(qwns.replace(",",""))]
        dtype += [(q,np.float32) for q in "seqc2 distc2".split()]
        return dtype


    def reclabs(self, start=0, stop=5):
        """
        :param start: common label slice starting index
        :param stop: common label slice stop index, can be None
        :return rls: list of reclabels

        Full reclabs is very large list::

            In [5]: rl = ab.reclabs(0,None)

            In [6]: len(rl)
            Out[6]: 64265

            In [7]: len(ab.clabels)
            Out[7]: 4848

            In [8]: 64265./4848.        ## average seq0 length is 13 points 
            Out[8]: 13.255981848184819

            In [9]: rl[:5]
            Out[9]: 
            ['[TO] BT BT BT BT SA',
             'TO [BT] BT BT BT SA',
             'TO BT [BT] BT BT SA',
             'TO BT BT [BT] BT SA',
             'TO BT BT BT [BT] SA']

        """
        l = []
        for clab in self.clabels[start:stop]:
            l.extend(Ctx.reclabs_(clab))
        pass
        return l


    def stats_(self, reclabs, qwns=None, rehist=False, c2shape=True):
        """
        ::

            st = ab.stats_(ab.point.reclab[:10])

        """
        if qwns is None:
            qwns = self.ok.qwns
        pass

        nrl = len(reclabs)
        stat = np.recarray((nrl,), dtype=self.stats_dtype(qwns))

        for ival, rl in enumerate(reclabs):

            self.sel = rl     ## setting single line selection

            rp = self.recpoint

            isel = rp.isel

            # NB OrderedDict setting order must match dtype name order
            od = odict()
            od["iv"] = rp.irl
            od["is"] = isel
            od["na"] = self.aseq.cu[isel,1] 
            od["nb"] = self.aseq.cu[isel,2]
            od["reclab"] = rl

            hh = self.rhist(rehist=rehist, c2shape=c2shape)
            qd = odict(map(lambda h:(h.qwn,h.c2p), hh))
            od.update(qd)

            od["seqc2"] = self.seq.c2[0]
            od["distc2"] = CFH.c2per_(hh)

            stat[ival] = tuple(od.values())
        pass

        st = ABStat(stat) 
        st.save()
        return st 

 
    def stats(self, start=0, stop=5, qwns=None, rehist=False, c2shape=True):
        """
        slice selection in AB has to be converted into common seq selection 
        in the evt 
        """
        if qwns is None:
            qwns = self.ok.qwns
        pass
        qwns = qwns.replace(",","")

        dtype = self.stats_dtype(qwns)

        nrs = self.nrecs(start, stop)
        trs = nrs.sum()

        log.info("AB.stats : %s :  trs %d nrs %s " % ( qwns, trs, repr(nrs)) )
        
        stat = np.recarray((trs,), dtype=dtype)
        ival = 0
        for i,isel in enumerate(range(start, stop)):

            self.sel = slice(isel, isel+1)    ## changing single seq line selection 

            nr = self.nrec
            assert nrs[i] == nr, (i, nrs[i], nr )  

            for irec in range(nr):

                self.irec = irec 

                self.qwn = qwns   # for collective qctx 

                qctx = self.qctx

                key = self.suptitle

                log.debug("stats irec %d nrec %d ival %d key %s qctx %s " % (irec, nr, ival, key, qctx))


                # NB OrderedDict setting order must match dtype name order
                od = odict()
                od["iv"] = ival
                od["is"] = isel
                od["na"] = self.a_count(0)    # single line selection so line=0
                od["nb"] = self.b_count(0)    # single line selection so line=0
                od["reclab"] = self.reclab 

                hh = self.rhist(qwns, irec, rehist=rehist, c2shape=c2shape)
                qd = odict(map(lambda h:(h.qwn,h.c2p), hh))
                od.update(qd)

                od["seqc2"] = self.seq.c2[0]
                od["distc2"] = CFH.c2per_(hh)

                stat[ival] = tuple(od.values())
                ival += 1
            pass
        pass
        log.info("AB.stats histogramming done")
        assert ival == trs, (ival, trs )

        st = ABStat(self.ok, stat) 
        st.save()
        return st 

    def totrec(self, start=0, stop=None, step=1):
        """
        :param sli:

        ::

            In [2]: ab.totrec()    # union of labels brings in a lot more of them
            Out[2]: 64265

            In [3]: ab.a.totrec()
            Out[3]: 43177

            In [4]: ab.b.totrec()
            Out[4]: 43085

        """
        nrs = self.nrecs(start, stop, step)
        return int(nrs.sum())

    def _get_clabels(self):
        """
        :return clabels: all labels of current flv 
        """
        flv = self.flv
        if flv == "seqhis":
            clabels = self.ahis.labels
        elif flv == "seqmat":
            clabels = self.amat.labels
        else:
            clabels = []
        pass
        return clabels
    clabels = property(_get_clabels)

    def _get_rposta_dv(self):
        """
        absolute a.rposta - b.rposta covering entire record array 
        """
        return np.abs( self.a.rposta - self.b.rposta ) 
    rposta_dv = property(_get_rposta_dv) 

    def rpost(self):
        """
        Sliced decompression within a selection 
        works via the rx replacement with rx[psel] 
        done in Evt.init_selection.
        """
        self.checkrec()
        av = self.a.rpost()
        bv = self.b.rpost()
        return av, bv

    def rpost_dv_max(self):
        """
        :return max a-b item deviations: 

        The np.amax with axis=(1,2) tuple of ints argument 
        acts to collapse those dimensions in the aggregation
        """ 
        av = self.a.rpost()
        bv = self.b.rpost()
        dv = np.abs( av - bv )
        return dv.max(axis=(1,2)) 
         
    def rpost_dv_where(self, cut):
        """
        :return photon indices with item deviations exceeding the cut: 
        """
        av = self.a.rpost()
        bv = self.b.rpost()
        dv = np.abs( av - bv )
        return self.a.where[np.where(dv.max(axis=(1,2)) > cut) ]  


    def rpol_dv_max(self):
        """
        :return max a-b item deviations: 
        """ 
        av = self.a.rpol()
        bv = self.b.rpol()
        dv = np.abs( av - bv )
        return dv.max(axis=(1,2)) 


    def rpol_dv_where_(self, cut):
        """
        :return photon indices with item deviations exceeding the cut: 
        """
        av = self.a.rpol()
        bv = self.b.rpol()
        dv = np.abs( av - bv )
        return np.where(dv.max(axis=(1,2)) > cut)   

    def rpol_dv_where(self, cut):
        """
        :return photon indices with item deviations exceeding the cut: 
        """
        av = self.a.rpol()
        bv = self.b.rpol()
        dv = np.abs( av - bv )
        return self.a.where[np.where(dv.max(axis=(1,2)) > cut) ]  


    def _set_warn_empty(self, we):
        self.a.warn_empty = we
        self.b.warn_empty = we
    def _get_warn_empty(self):
        a_we = self.a.warn_empty
        b_we = self.b.warn_empty
        assert a_we == b_we
        return a_we
    warn_empty = property(_get_warn_empty, _set_warn_empty) 


    def checkrec(self,fr=0,to=0):
        nr = self.nrec
        if nr > -1 and fr < nr and to < nr:
            return True 
        pass
        log.fatal("checkrec requires a single label selection nr %d fr %d to %d" % (nr,fr,to))
        return False

    def rdir(self, fr=0, to=1):
        if not self.checkrec(fr,to):
            return None
        aval = self.a.rdir(fr,to)
        bval = self.b.rdir(fr,to)
        return aval, bval

    def rpol_(self, fr):
        if not self.checkrec(fr):
            return None
        aval = self.a.rpol_(fr)
        bval = self.b.rpol_(fr)
        return aval, bval

    def rpol(self):
        if not self.checkrec():
            return None
        aval = self.a.rpol()
        bval = self.b.rpol()
        return aval, bval

    def rw(self):
        if not self.checkrec():
            return None
        aval = self.a.rw()
        bval = self.b.rw()
        return aval, bval


    @classmethod
    def rrandhist(cls):
        bn = np.linspace(-4,4,200)
        av = np.random.standard_normal(8000)
        bv = np.random.standard_normal(8000)
        la = ["A rand", "B rand"]

        ctx = {}
        ctx["det"] = "dummy"
        ctx["tag"] = "1"
        ctx["seq"] = "dummy"
        ctx["qwn"] = "U"
        ctx["irec"] = "0"   

        cfh = CFH(ctx)
        cfh(bn,av,bv,la,c2cut=cls.C2CUT)
        return cfh
 
    def _set_qwn(self, qwn):
        self._qwn = qwn 
    def _get_qwn(self):
        return self._qwn
    qwn = property(_get_qwn, _set_qwn)

    def _get_h(self):
        hh = self.rhist( self.qwn, self.irec)
        assert len(hh) == 1
        return hh[0]        
    h = property(_get_h)

    def _get_hh(self):
        hh = self.rhist()
        return hh
    hh = property(_get_hh)


    def rhist_(self, ctxs, rehist=False, c2shape=False):
        """
        :param ctxs: list of Ctx 
        :param rehist: bool, when True recreate histograms, otherwise pre-persisted histos are loaded


        Collects unique seq0 and then interactes 

        """
        seq0s = list(set(map(lambda ctx:ctx.seq0,ctxs)))
        hh = []
        for seq0 in seq0s:
            sel = seq0.replace("_", " ")
            log.info("AB.rhist_ setting sel to \"%s\" rehist %d  " % (sel, rehist) )

            self.sel = sel   # adjust selection 

            for ctx in filter(lambda ctx:ctx.seq0 == seq0, ctxs):
                hs = self.rhist(qwn=ctx["qwn"], irec=int(ctx["irec"]), rehist=rehist, c2shape=c2shape)
                assert len(hs) == 1
                hh.append(hs[0])
            pass
        pass
        return hh

    def rhist(self, qwn=None, irec=None, rehist=False, log_=False, c2shape=True ): 

        if qwn is None:
            qwn = self.ok.qwns
        assert type(qwn) is str

        if irec is None:
            irec = self.irec
        assert type(irec) is int

        hh = []

        srec = CFH.srec_(irec)

        log.debug("AB.rhist qwn %s irec %s srec %s " % (qwn, irec, srec))

        for ir in CFH.irec_(srec):

            log.debug(" ir %s ir 0x%x " % (ir,ir))

            self.irec = ir

            for q in str(qwn):

                self.qwn = q 

                ctx=self.ctx

                h = CFH(ctx)

                h.log = log_

                if h.exists() and not rehist:
                    h.load()
                else:
                    bn, av, bv, la = self.rqwn(q, ir)

                    h(bn,av,bv,la, c2cut=self.C2CUT, c2shape=c2shape )

                    h.save()
                pass
                hh.append(h)
            pass
        pass
        return hh
 




    def rqwn(self, qwn, irec): 
        """
        :param qwn: X,Y,Z,W,T,A,B,C or R  
        :param irec: step index 0,1,...
        :return binx, aval, bval, labels

        ::

            bi,a,b,l = cf.rqwn("T",4)

        """
        a = self.a
        b = self.b
        lval = "%s[%s]" % (qwn.lower(), irec)
        labels = ["Op : %s" % lval, "G4 : %s" % lval]
 
        if qwn == "R":
            apost = a.rpost_(irec)
            bpost = b.rpost_(irec)
            aval = vnorm(apost[:,:2])
            bval = vnorm(bpost[:,:2])
            cbins = a.pbins()
        elif qwn == "W":
            aval = a.recwavelength(irec)
            bval = b.recwavelength(irec)
            cbins = a.wbins()
        elif qwn in Evt.RPOST:
            q = Evt.RPOST[qwn]
            aval = a.rpost_(irec)[:,q]
            bval = b.rpost_(irec)[:,q]
            if qwn == "T":
                cbins = a.tbins()
            else:
                cbins = a.pbins()
            pass
        elif qwn in Evt.RPOL:
            q = Evt.RPOL[qwn]
            aval = a.rpol_(irec)[:,q]
            bval = b.rpol_(irec)[:,q]
            cbins = a.rpol_bins()
        else:
            assert 0, "qwn %s unknown " % qwn 
        pass

        binscale = Evt.RQWN_BINSCALE[qwn]
        bins = cbins[::binscale]   

        # hmm should arrange scaling to retain extreme bins to avoid clipping perhaps ??
        # formerly tried clever binning, but certainty of simple binning turns out easier to interpret
        #    bins = decompression_bins(cbins, [aval, bval], label=lval, binscale=binscale )
        #  

        if len(bins) == 0:
            raise Exception("no bins")

        return bins, aval, bval, labels



if __name__ == '__main__':
    from opticks.ana.main import opticks_main
    ok = opticks_main()
    ab = AB(ok)
    ab.dump()

    
