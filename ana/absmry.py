#!/usr/bin/env python
"""
ABSmry
========

Canonical instance ab.smry collects summary info 
from bi-simulation event comparisons.

"""
from __future__ import print_function
import os, sys, logging, numpy as np
from collections import OrderedDict as odict
from opticks.ana.base import json_save_, json_load_, findfile
from opticks.ana.num import Num
from opticks.ana.level import Level

log = logging.getLogger(__name__)


class ABSmryTab(object):
    def __init__(self, base="$TMP"):
        base = os.path.expandvars(base)
        LV=os.environ.get("LV", None)
        log.info("base %s LV %s " % (base, LV))
        relp = findfile(base, ABSmry.NAME )
        self.base = base
        s = odict() 
        for rp in relp:
            path = os.path.join(base, rp)
            smry = ABSmry.Load(os.path.dirname(path))

            if LV is not None and smry.key != LV: continue

            try:
                fkey = float(smry.key)
            except ValueError:
                fkey = 1000.0+len(s)    
            pass
            log.debug("rp:%s smry.key:%s fkey:%s" % (rp, smry.key, fkey))

            while fkey in s:
                fkey += 0.001
            pass 
            s[fkey] = smry   
        pass
        self.s = s
    def __repr__(self):
        return "\n".join(["ABSmryTab", ABSmry.Labels()]+map(lambda kv:repr(kv[1]), sorted(self.s.items(), key=lambda kv:float(kv[0]))  ))
             


class ABSmry(odict):
    """

    After adding keys can usually update the summary just via redoing analysis
    with scan-tp    


    """
    NAME = "ABSmry.json"
    DVS = "rpost_dv rpol_dv ox_dv".split()
    DVSQ = "maxdvmax RC level ndvp".split()

    @classmethod
    def Path(cls, dir_):
        return os.path.join(dir_, cls.NAME) 

    @classmethod
    def Pfx(cls, dir_):
        e = dir_.split("/")
        return e[e.index("evt")-1]


    KEYS = r"""
    ab.a.tagdir
    ab.a.metadata.lv
    ab.a.metadata.solid
    ab.dsli
    ab.RC
    ab.level
    ab.cfm.numPhotons 
    ab.mal.fmaligned
    ab.mal.nmal
    ab.pro.ap.tim
    ab.pro.bp.tim
    ab.pro.boa
    """

    @classmethod
    def Make(cls, ab):
        s = cls()
        s.dir_ = ab.a.tagdir

        for q in filter(None, map(str.strip,cls.KEYS.split("\n"))):
            v = eval(q)
            #log.info("eval(%s) -> %s  %r " % (q, v, type(v)) )
            if type(v) is np.float32:
                v = float(v)
            pass 
            s[q] = v
        pass
        for dv in cls.DVS:
            for q in cls.DVSQ:
                key = "ab.%s.%s" % (dv, q)
                s[key] = eval(key)
            pass
        pass 
        s.init()
        return s 

    @classmethod
    def Load(cls, dir_):
        log.debug("dir_ %s " % dir_ )
        d = json_load_(cls.Path(dir_))
        s = cls()
        for k, v in d.items():
            s[k] = v  
        pass
        s.init()
        return s 

    pfx = property(lambda self:self.Pfx(self.tdir)) 
    tdir = property(lambda self:self["ab.a.tagdir"]) 
    key = property(lambda self:self["ab.a.metadata.lv"]) 
    dsli = property(lambda self:self.get("ab.dsli","."))
    RC = property(lambda self:self["ab.RC"])
    level = property(lambda self:self["ab.level"])
    fmal = property(lambda self:self["ab.mal.fmaligned"])
    nmal = property(lambda self:self.get("ab.mal.nmal",-1))
    npho = property(lambda self:self.get("ab.cfm.numPhotons",-1))
    solid = property(lambda self:self["ab.a.metadata.solid"])

    ati = property(lambda self:self.get("ab.pro.ap.tim",-1))
    bti = property(lambda self:self.get("ab.pro.bp.tim",-1))
    boa = property(lambda self:self.get("ab.pro.boa",-1))


    def __init__(self):
        odict.__init__(self)

    def init(self):
        self.lev = None if self.level is None else Level.FromLevel(self.level)

    def save(self, dir_=None):
        if dir_ is None:
            dir_ = self.dir_
        pass
        path = self.Path(dir_) 
        log.info("saving to %s " % path)
        json_save_(path, self )


    HEAD2 = False


    @classmethod
    def Labels(cls):
        """
        level uses ansi codes in table, so has 9 invisible characters that take no screen space
        """
        head = " %5s %10s %4s %7s %4s %4s %10s %5s " % ("pfx", "dsli", "LV", "level", "RC", "npho", "fmal(%)", "nmal" ) 
        head2 = " %7s %7s %7s " % ( "ati", "bti", "boa" )  if cls.HEAD2 else None  
        space = "   "
        body = cls.label_dv()
        tail = "solid" 
        return " ".join(filter(None,[head,head2,space,body,space, tail])) 

    def __repr__(self):
        head = " %5s %10s %4s %16s 0x%.2x %4s %10.3f %5d " % ( self.pfx[-5:], self.dsli, self.key, self.lev.fn_(self.lev.nam) , self.RC, Num.String(self.npho), self.fmal*100.0, self.nmal )
        head2 = " %7.4f %7.1f %7.1f " % (self.ati, self.bti, self.boa ) if self.HEAD2 else None
        space = "   "
        body = self.desc_dv()
        tail = "%s"  % self.solid
        return " ".join(filter(None,[head,head2,space,body,space,tail])) 
 

    @classmethod
    def dvk(cls, n, k ):
        return "ab.%s.%s" % (n,k) 
    def dvlev(self, dvn):
        levk = self.dvk(dvn, "level") 
        levn = self[levk]
        return Level.FromName(levn)
    def desc_dv_(self, dvn):
        lev = self.dvlev(dvn)      

        mxk = self.dvk(dvn, "maxdvmax") 
        mxv = self[mxk] 
        mxs = "%10.4f" % mxv

        ndk = self.dvk(dvn, "ndvp") 
        ndv = self.get(ndk, -1)

        return " %16s %15s %5s " % ( lev.fn_(lev.nam), lev.fn_(mxs), ndv ) 

    @classmethod
    def label_dv_(cls, dvn):
        mxk = cls.dvk(dvn, "maxdvmax") 
        # 16+15+3-18 = 16    
        # 16+15+3-18+6 = 22 
        return " %22s " % ( mxk[3:-5] ) 
    @classmethod
    def label_dv(cls):
        return " ".join([cls.label_dv_(dvn) for dvn in cls.DVS])

    def desc_dv(self):
        return " ".join([self.desc_dv_(dvn) for dvn in self.DVS])

     


def test_MakeLoad(ok):
    ab = AB(ok)
    ab.dump()

    dir_ = "/tmp"
    s = ABSmry.Make(ab)  
    s.save(dir_)

    s2 = ABSmry.Load(dir_)



if __name__ == '__main__':
    from opticks.ana.main import opticks_main

    from opticks.ana.ab import AB 
    ok = opticks_main()

    #test_MakeLoad(ok)
  
    st = ABSmryTab()
    print(st)





