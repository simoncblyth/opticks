#!/usr/bin/env python
"""
profile.py
=============
"""

from __future__ import print_function
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.nload import np_load
from opticks.ana.nload import tagdir_, stamp_


class Profile(object):
    def __init__(self, ok):
        self.tagdir = ok.tagdir 
        self.ok = ok 
        self.sli = slice(0,10)
        self.init()

    def path(self, sfx=""):
        return os.path.join( self.tagdir, "OpticksProfile%s.npy" % sfx )   # quads 
        
    def init(self):
        self.loadProfile() 
        self.loadAcc() 

        okp, ok0, ok1 = self.deltaT("OPropagator::launch")   
        g4r, g40, g41 = self.deltaT("CRunAction::BeginOfRunAction","CRunAction::EndOfRunAction")   ## CG4::propagate includes significant initialization
        stt = self.setupTrancheTime()

        g4p = g4r - stt 
        g4p_okp = -1.0 if okp == 0. else g4p/okp 

        self.okp = okp
        self.g4r = g4r
        self.g4p = g4p 
        self.stt = stt
        self.g4p_okp = g4p_okp

        self.oki = [ok0, ok1]
        self.g4i = [g40, g41]



    def loadProfile(self):
        path = self.path("")
        lpath = self.path("Labels")  
        print("path:%s stamp:%s " % (path, stamp_(path) ))
        print("lpath:%s stamp:%s " % (lpath, stamp_(lpath) ))

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
        acpath = self.path("Acc")
        lacpath = self.path("AccLabels")

        print("acpath:%s stamp:%s " % (acpath, stamp_(acpath) ))
        print("lacpath:%s stamp:%s " % (lacpath, stamp_(lacpath) ))

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

        log.info(" l0:%30s l1:%30s p0:%3d p1:%3d  (v0:%10.1f v1:%10.1f dv:%10.1f )  ( t0:%10.4f t1:%10.4f dt:%10.4f )  " % ( l0, l1, p0, p1, v[p0],v[p1], dv, t[p0],t[p1], dt  )) 
        return dt, dv, p0, p1  


    def deltaT(self, arg0, arg1=None):
        dt, dv, p0, p1 = self.delta(arg0, arg1)
        return dt, p0, p1 
   
    def brief(self): 
        return "      okp %10.4f     g4r %-10.4f stt %-10.4f g4p %-10.4f        g4p/okp %-10.4f    " % (self.okp, self.g4r, self.stt, self.g4p, self.g4p_okp )   

    def labels(self):
        return " %3s : %50s : %10s %10s %10s %10s   " % ( "idx", "label", "t", "v", "dt", "dv" )

    def line(self, i):
        return " %3d : %50s : %10.4f %10.4f %10.4f %10.4f   " % ( i, self.l[i], self.t[i], self.v[i], self.dt[i], self.dv[i] )

    def __repr__(self):
        return "\n".join(["ab.pro",  self.brief(), "%r" % self.sli, self.labels()] + map(lambda i:self.line(i), np.arange(len(self.l))[self.sli]) + [self.labels()]  )

    def __getitem__(self, sli):
        self.sli = sli
        return self 



if __name__ == '__main__':
    from opticks.ana.main import opticks_main
    import matplotlib.pyplot as plt

    ok = opticks_main(doc=__doc__)  
    log.info(ok.brief)

    op = Profile(ok) 
    print(op)


    a = op.a  
    l = op.l 

    plt.plot( op.t, op.v, 'o' )

    plt.axvline( op.t[op.oki[0]], c="b" )
    plt.axvline( op.t[op.oki[1]], c="b" )

    plt.axvline( op.t[op.g4i[0]], c="r" )
    plt.axvline( op.t[op.g4i[1]], c="r" )


    plt.ion()
    plt.show()



