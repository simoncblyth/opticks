#!/usr/bin/env python
"""
profile.py
=============
"""

from __future__ import print_function
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.nload import np_load
from opticks.ana.nload import tagdir_, stmp_, time_



class Prof(object):
    def __init__(self, ok, name ):
        self.ok = ok
        self.name = name
        self.fmt = "%Y%m%d-%H%M"

        g4 = name.find("g4") > -1      

        self.tagdir = ok.ntagdir if g4 else ok.tagdir
        self.sli = slice(0,10)
        self.loadProfile() 
        self.loadAcc()

        if g4:  
            g4r, g40, g41 = self.deltaT("CRunAction::BeginOfRunAction","CRunAction::EndOfRunAction")   ## CG4::propagate includes significant initialization
            stt = self.setupTrancheTime()
            tim = g4r - stt 
            idx = [g40, g41]  
        else:
            okp, ok0, ok1 = self.deltaT("OPropagator::launch")   
            tim = okp
            idx = [ok0, ok1]  
        pass
        self.tim = tim
        self.idx = idx
        self.sli = slice(idx[0],idx[1]+1)


    def pfmt(self, path1, path2, path3=None):
        t_path1 = time_(path1)
        t_path2 = time_(path2)

        if path3 is not None:
            t_path3 = time_(path3)
            adt3 = abs(t_path1 - t_path3)
            assert adt3.seconds < 1  
        pass 

        adt = abs(t_path1 - t_path2)
        assert adt.seconds < 1  
        return "  %-90s  %s " % ( path1, t_path1.strftime(self.fmt) )

    def path(self, sfx=""):
        return os.path.join( self.tagdir, "OpticksProfile%s.npy" % sfx )   # quads 

    def loadProfile(self):
        path = self.path("")
        lpath = self.path("Labels")  
        self.prfmt = self.pfmt(path, lpath)

        a = np.load(path)
        assert a.ndim == 2

        l = np.load(lpath)
        assert l.ndim == 2 and l.shape[1] == 64
        assert len(a) == len(l) 

        ll = list(l.view("|S64")[:,0])
        lll = np.array(ll)

        t,dt,v,dv = a[:,0], a[:,1], a[:,2], a[:,3]

        self.l = lll 
        self.a = a 

        self.t = t
        self.dt = dt
        self.v = v
        self.dv = dv


    def loadAcc(self):
        path = self.path("")
        acpath = self.path("Acc")
        lacpath = self.path("AccLabels")
        self.acfmt = self.pfmt(acpath, lacpath, path)

        ac = np.load(acpath)
        assert ac.ndim == 2

        lac = np.load(lacpath)
        assert lac.ndim == 2 and lac.shape[1] == 64
        assert len(ac) == len(lac) 

        self.ac = ac
        self.lac = np.array( lac.view("|S64")[:,0] )


    def acc(self, label):
        return self.ac[np.where( self.lac == label )]

    def setupTrancheTime(self):
        stt_acc = self.acc("CRandomEngine::setupTranche")
        assert stt_acc.ndim == 2 and stt_acc.shape[1] == 4 
        assert len(stt_acc) == 1 or len(stt_acc) == 0 
        stt = stt_acc[0,1] if len(stt_acc) == 1 else 0.  
        return stt 

    def delta(self, arg0, arg1=None ):
        """
        :param arg0: start label 
        :param arg1: end label 
        :return dt, dv:  deltaTime, deltaVM between two OK_PROFILE code points 

        If arg1 is not provided assume arg0 specifies a pair 
        using the underscore convention. For example with arg0 
        as "CG4::propagate" the pair becomes ("_CG4::propagate","CG4::propagate") 
        """
        if arg1 is None:
            l0 = "_%s" % arg0 
            l1 = "%s" % arg0 
        else:
            l0 = arg0 
            l1 = arg1 
        pass
        p0 = np.where( self.l == l0 )[0][0]
        p1 = np.where( self.l == l1 )[0][0]
        v = self.v 
        t = self.t 

        dv = v[p1]-v[p0]
        dt = t[p1]-t[p0]

        log.debug(" l0:%30s l1:%30s p0:%3d p1:%3d  (v0:%10.1f v1:%10.1f dv:%10.1f )  ( t0:%10.4f t1:%10.4f dt:%10.4f )  " % ( l0, l1, p0, p1, v[p0],v[p1], dv, t[p0],t[p1], dt  )) 
        return dt, dv, p0, p1  

    def deltaT(self, arg0, arg1=None):
        dt, dv, p0, p1 = self.delta(arg0, arg1)
        return dt, p0, p1 

    def line(self, i):
        return " %6d : %50s : %10.4f %10.4f %10.4f %10.4f   " % ( i, self.l[i], self.t[i], self.v[i], self.dt[i], self.dv[i] )

    def labels(self):
        return " %6s : %50s : %10s %10s %10s %10s   " % ( "idx", "label", "t", "v", "dt", "dv" )

    def bodylines(self):
        nli = self.sli.stop - self.sli.start
        if nli < 10:
            ll = map(lambda i:self.line(i), np.arange(self.sli.start, self.sli.stop)) 
        else:
            ll = map(lambda i:self.line(i), np.arange(self.sli.start, self.sli.start+5) ) 
            ll += [" ..."]
            ll += map(lambda i:self.line(i), np.arange(self.sli.stop - 5, self.sli.stop) )   
        pass
        return ll

    def lines(self):
        return [self.name, self.prfmt, self.acfmt, "%r" % self.sli, self.labels()] + self.bodylines() + [self.labels()] 

    def __repr__(self):
        return "\n".join(self.lines())

    def __getitem__(self, sli):
        self.sli = sli
        return self 



class Profile(object):
    def __init__(self, ok):
        self.okp = Prof(ok, "ab.pro.okp" ) 
        self.g4p = Prof(ok, "ab.pro.g4p" ) 
        g4p_okp = self.g4p.tim/self.okp.tim if self.okp.tim > 0 else -1  
        self.g4p_okp = g4p_okp
  
    def brief(self): 
        return "      okp %-10.4f     g4p %-10.4f      g4p/okp %-10.4f    " % (self.okp.tim, self.g4p.tim, self.g4p_okp )   

    def __repr__(self):
        return "\n".join(["ab.pro", self.brief()] + self.okp.lines() + self.g4p.lines() )



if __name__ == '__main__':
    from opticks.ana.main import opticks_main
    import matplotlib.pyplot as plt

    ok = opticks_main(doc=__doc__)  
    log.debug(ok.brief)

    op = Profile(ok) 
    print(op)

    okp = op.okp
    g4p = op.g4p

    plt.plot( okp.t, okp.v, 'o' )
    plt.plot( g4p.t, g4p.v, 'o' )

    plt.axvline( okp.t[okp.idx[0]], c="b" )
    plt.axvline( okp.t[okp.idx[1]], c="b" )

    plt.axvline( g4p.t[g4p.idx[0]], c="r" )
    plt.axvline( g4p.t[g4p.idx[1]], c="r" )

    plt.ion()
    plt.show()



