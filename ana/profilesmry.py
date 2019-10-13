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
profilesmry.py
===============

::

    an ; ip profilesmry.py    
      ## loads times from scans, after manual adjustment of pfx and cat startswith in __main__

"""

from __future__ import print_function
import os, sys, re, logging, numpy as np, argparse, textwrap
from collections import OrderedDict as odict
log = logging.getLogger(__name__)

from opticks.ana.num import Num
from opticks.ana.base import findfile
from opticks.ana.profile import Profile
from opticks.ana.bashnotes import BashNotes

class ProfileSmry(object):
    """
    ProfileSmry finds and loads profiles and holds values obtained 
    """

    BASE = "$OPTICKS_EVENT_BASE" 

    @classmethod
    def LoadHit_(cls, htpath):
        return np.load(htpath) if os.path.exists(htpath) else None


    @classmethod
    def Load_(cls, pfx, base, startswith=None):
        """
        :param pfx: prefix, see scan-vi, eg scan-ph-0
        :param base: directory 
        :param startswith: used to select the *cat* category of runs
                           the cat is the path element after evt, 
                           an example of *cat* for scan-ph is cvd_0_rtx_1_100M

        :return s: odict keyed by cat with Profile instances 

        Finds all persisted profiles with selected prefix that meet the startswith selection, 
        collecting them into an odict which is returned.
        """ 
        assert not base is None
        if startswith is None:
            select_ = lambda n:True
        else:  
            select_ = lambda n:n.find(startswith) == 0   
        pass

        relp = findfile(base, Profile.NAME )   # list of relative paths beneath base
        #log.info("base %s relp %d : NAME %s startswith %s " % (base, len(relp), Profile.NAME, startswith ))

        s = odict() 
        for rp in relp:
            path = os.path.join(base, rp)
            elem = path.split("/")
            cat = elem[elem.index("evt")+1]  
            sel = select_(cat)

            #log.info("path %s cat %s sel %s " % (path, cat, sel) )

            if not sel: continue
            name = cat 
            ecat = cat.split("_")
            npho = Num.Int( ecat[-1] )  

            dir_ = os.path.dirname(path)

            prof = Profile(dir_, name)
            prof.npho = npho 

            htpath1 = os.path.join(dir_, "1", "ht.npy")
            ht = cls.LoadHit_(htpath1)
            nhit = ht.shape[0] if not ht is None else -1
            prof.ht = ht
            prof.nhit = nhit

            ihit = prof.npho/prof.nhit

            #print("car %20s npho %9d nhit %9d ihit %5d     path %s " % (cat, prof.npho, prof.nhit, ihit,  path))
            s[cat] = prof
        pass
        return s  

    @classmethod
    def Base(cls, pfx, base=None):
        if base is None:
            base = cls.BASE  
        pass
        base = os.path.expandvars(os.path.join(base,pfx))
        return base


    CATPTN = re.compile("^cvd_(?P<cvd>\d)_rtx_(?P<rtx>\d)_(?P<M>\d*)M$")

    @classmethod
    def ExamineCats(cls, pfx, base=None):
        base = cls.Base(pfx, base)
        evtdir = os.path.join(base, "evt")
        cats = os.listdir(evtdir)
        c = {}
        for cat in cats:
            m = cls.CATPTN.match(cat)
            if not m:
                log.error("failed to match %s " % cat )
                continue 
            pass
            c[cat] = m.groupdict()
        pass
        return c 

    @classmethod
    def UCVD(cls, c ):
        ucvd = list(set(map(lambda d:d["cvd"], c.values())))
        return ucvd  

    @classmethod
    def Load(cls, pfx, base=None, startswith=None, gpufallback=None):
        base = cls.Base(pfx, base)
        s = cls.Load_(pfx, base, startswith)
        ps = cls.FromDict(s, pfx, startswith)
        ps.base = base
        ps.gpufallback = gpufallback
        return ps 
        
    @classmethod
    def FromDict(cls, s, pfx, startswith):
        """
        :param s: raw odict keyed with cat with Profile instance values

        Creates ProfileSmry instance comprising arrays populated
        from the Profile instances 
        """
        launch = np.zeros( [len(s), 10], dtype=np.float32 )  
        alaunch = np.zeros( [len(s) ], dtype=np.float32 )  
        interval = np.zeros( [len(s), 9], dtype=np.float32 )  
        ainterval = np.zeros( [len(s)], dtype=np.float32 )  
        npho = np.zeros( len(s), dtype=np.int32 ) 
        nhit = np.zeros( len(s), dtype=np.int32 ) 
        meta = np.zeros( len(s), dtype=np.object )

        for i, kv in enumerate(sorted(s.items(), key=lambda kv:kv[1].npho )): 
            prof = kv[1]  

            npho[i] = prof.npho
            nhit[i] = prof.nhit
            launch[i] = prof.launch
            meta[i] = prof.meta

            alaunch[i] = np.average( launch[i][1:] )
            interval[i] = prof.start_interval
            ainterval[i] = np.average( interval[i][1:] )
        pass

        ps = cls(s)
        ps.launch = launch  
        ps.alaunch = alaunch  
        ps.interval = interval 
        ps.ainterval = ainterval 
        ps.npho = npho
        ps.nhit = nhit
        ps.creator = "FromDict:%s:%s" % (pfx,startswith )
        ps.meta = meta
        ps.postinit()  

        return ps 


    @classmethod
    def FromExtrapolation(cls, npho, time_for_1M=0. ):
        """
        See notes/issues/geant4-beamOn-profiling.rst

        100 s : for tboolean-box scan-ph
        239 s : for full JUNO  scan-pf before alignment shakedown
        """
        assert time_for_1M > 0, time_for_1M

        s = odict()
        ps = cls(s)
        ps.npho = npho
        xtim =  (npho/1e6)*time_for_1M 
        ps.alaunch = xtim
        ps.ainterval = xtim
        ps.creator = "FromExtrapolation" 
        return ps  

    @classmethod
    def FromAB(cls, a, b, att="ainterval"):
        s = odict()
        ps = cls(s)
        assert np.all( a.npho == b.npho )
        ps.npho = a.npho
        ps.ratio = getattr(b,att)/getattr(a, att)
        ps.creator = "FromAB" 
        return ps  

    @classmethod
    def FromAtt(cls, a, num_att="ainterval", den_att="alaunch" ):
        s = odict()
        ps = cls(s)
        ps.npho = a.npho
        ps.ratio = getattr(a,num_att)/getattr(a, den_att)
        ps.creator = "FromAtt" 
        return ps  


    def __init__(self, s):
        self.s = s 


    COMMON = r"""
    CDeviceBriefAll
    CDeviceBriefVis
    RTXMode
    NVIDIA_DRIVER_VERSION
    """
    def commonk(self):
        return filter(None,textwrap.dedent(self.COMMON).split("\n"))

    def postinit(self):
        d = odict()
        for k in self.commonk():
            d[k] = self.metacommon(k)
        pass
        self.d = d

    def descmeta(self):
        return "\n".join(["%25s : %s " % (k, v) for k,v in self.d.items()])

    def _get_gpu(self):
        return self.d.get('CDeviceBriefVis',self.gpufallback) 
    gpu = property(_get_gpu)

    def _get_rtx(self):
        RTXMode = self.d.get('RTXMode', None)
        assert RTXMode in [None,0,1]
        e = { None:"?", 0:"OFF", 1:"ON", 2:"ON.Tri" } 
        return "RTX %s" % e[RTXMode] 
    rtx = property(_get_rtx)

    def _get_autolabel(self):
        return "%s, %s" % (self.gpu, self.rtx) 
    autolabel = property(_get_autolabel)

    def metacommon(self, k):
        vv = list(set(map( lambda m:m.get(k, None), self.meta )))
        assert len(vv) in [0,1], vv
        return vv[0] if len(vv) == 1 else None

    def desc(self):
        return "ProfileSmry %s %s %s " % (self.creator, getattr(self, 'base',""), self.d['CDeviceBriefVis'] ) 

    def __repr__(self):
        return "\n".join([self.desc(), self.autolabel, self.descmeta(), Profile.Labels()]+map(lambda kv:repr(kv[1]), sorted(self.s.items(), key=lambda kv:kv[1].npho )  ))



class ProfileMain(object):
    @classmethod
    def ParseArgs(cls, doc):
        parser = argparse.ArgumentParser(__doc__)
        default_cvd = os.environ.get("OPTICKS_DEFAULT_INTEROP_CVD", "0")  ## hmm this is broken by scan-rsync when looking as scans from another machine
        parser.add_argument( "--pfx", default="scan-ph", help="Start of prefix to be appended with a hyphen and integer, beneath which to search for OpticksProfile.npy" )
        parser.add_argument( "vers", nargs="*", default=[10], type=int, help="Prefix beneath which to search for OpticksProfile.npy" )
        parser.add_argument( "--cvd", default=default_cvd, help="CUDA_VISIBLE_DEVICE for the named GPU" )
        parser.add_argument( "--gpufallback", default="Quadro_RTX_8000", help="Fallback GPU Name for older scans without this metadata, eg TITAN_RTX" )
        args = parser.parse_args()
        return cls(args)

    def get_pfx(self, v):
        return "%s-%s" % ( self.args.pfx, v) 

    def get_cvd(self, pfx):
        """
        When only one cvd in the cats return it, 
        otherwise return the argument
        """ 
        c = ProfileSmry.ExamineCats(pfx)
        ucvd = ProfileSmry.UCVD(c)         
        if len(ucvd) == 1:
            cvd = ucvd[0]
        else:
            log.info("mixed cvd using argument %s " % pm.cvd ) 
            cvd = pm.cvd
        pass 
        return cvd 

    def _get_bashcmd(self):
        pfx = self.args.pfx  # without version tail -0 -1 
        elem = pfx.split("-")
        assert len(elem) == 2 
        return "%s-;%s-notes" % (elem[0], pfx)
    bashcmd = property(_get_bashcmd)

    def __init__(self, args):
        self.args = args 
        self.vers = args.vers
        self.pfx0 = self.get_pfx(self.vers[0])
        self.cvd = args.cvd 
        self.gpufallback = args.gpufallback 

        bashcmd = self.bashcmd
        log.info("lookup BashNotes from %s " % bashcmd )
        self.bnote = BashNotes(bashcmd)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(precision=4, linewidth=200)

    pm = ProfileMain.ParseArgs(__doc__) 

    ps = {}
    for v in pm.vers:
        pfx = pm.get_pfx(v) 
        cvd = pm.get_cvd(pfx) 

        print(" v %d  pfx %s " % (v, pfx))
        print(" %s " % (pm.bnote(v)))

        ps[0] = ProfileSmry.Load(pfx, startswith="cvd_%s_rtx_0" % cvd)
        ps[1] = ProfileSmry.Load(pfx, startswith="cvd_%s_rtx_1" % cvd)

        #ps[9] = ProfileSmry.FromExtrapolation( ps[0].npho,  time_for_1M=239. )

        print("\n")
        print(ps[0])
        print("\n")
        print(ps[1])
        #print(ps[9])
    pass



