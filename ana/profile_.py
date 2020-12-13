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
profile_.py
=============

Renamed from profile.py due to notes/issues/ipython-ipdb-issue.rst

::

    TEST=OpticksRunTest ipython -i ~/opticks/ana/profile_.py -- --tag 0


    LV=box profile_.py 
    LV=box python2.7 profile_.py 


    LV=box ipython -i profile_.py 

    LV=0 ip profile_.py 

    LV=box ip profile_.py --cat cvd_1_rtx_1_1M --pfx scan-ph --tag 0

    ip(){ local arg1=${1:-evt.py}; shift; ipython -i -- $(which $arg1) $* ; }


    ip profile_.py --cat cvd_1_rtx_0_1M --pfx scan-pf-0 --tag 0
         OKG4Test run  


scan-pf-0 OKG4Test 239s for 1M::

    In [17]: ap.times("CRunAction::BeginOfRunAction")
    Out[17]: array([1206.6406, 1464.2812, 1708.2578, 1950.6406, 2191.8984, 2439.336 , 2681.1562, 2916.8828, 3153.2656, 3389.5   ], dtype=float32)

    In [18]: ap.times("CRunAction::EndOfRunAction")
    Out[18]: array([1460.2266, 1706.0625, 1948.4453, 2189.6719, 2436.9219, 2678.9219, 2914.6875, 3151.0625, 3387.3281, 3622.4453], dtype=float32)

    In [19]: ap.times("CRunAction::EndOfRunAction") - ap.times("CRunAction::BeginOfRunAction")
    Out[19]: array([253.5859, 241.7812, 240.1875, 239.0312, 245.0234, 239.5859, 233.5312, 234.1797, 234.0625, 232.9453], dtype=float32)

    In [25]: np.average( ap.times("CRunAction::EndOfRunAction") - ap.times("CRunAction::BeginOfRunAction") )
    Out[25]: 239.3914


    In [22]: ap.times("OPropagator::launch")
    Out[22]: array([1463.7578, 1707.7422, 1950.1172, 2191.3594, 2438.7422, 2680.625 , 2916.3516, 3152.7344, 3388.9844, 3624.125 ], dtype=float32)

    In [24]: ap.times("OPropagator::launch") - ap.times("_OPropagator::launch")
    Out[24]: array([0.6875, 0.6953, 0.6875, 0.6953, 0.8125, 0.7188, 0.6953, 0.6953, 0.6953, 0.6875], dtype=float32)


::

    ip profile_.py --cat cvd_1_rtx_1_1M --pfx scan-pf-0 --tag 0


    In [2]: pr.q
    Out[2]: array([0.0204, 0.0182, 0.0208, 0.0191, 0.0179, 0.02  , 0.0193, 0.018 , 0.0178, 0.0196])

    In [1]: pr.q
    Out[1]: array([0.0214, 0.0189, 0.019 , 0.0186, 0.0187, 0.0206, 0.0183, 0.0183, 0.0191, 0.0188])


"""

from __future__ import print_function
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)
lmap = lambda *args:list(map(*args))


#from opticks.ana.debug import MyPdb
"""
# not working with py3
try:
    from IPython.core.debugger import Pdb as MyPdb
except ImportError:
    class MyPdb(object):
        def set_trace(self):
            log.error("IPython is required for ipdb.set_trace() " )
        pass  
    pass
pass
ipdb = MyPdb()
"""
ipdb = None

from opticks.ana.log import bold_, blink_
from opticks.ana.base import json_load_
from opticks.ana.base import u_,b_,d_  
from opticks.ana.nload import np_load
from opticks.ana.nload import tagdir_, stmp_, time_, tfmt_ 


def fmtarray_(a):
    if a is None:
        return "-"
    pass 
    return " ".join(map(lambda v:"%8.4f" % v, a)) 

class Profile(object):
    """
    Opticks executables can record time and virtual memory 
    and codepoint strings at various points throughout operations. 
    These are recorded into NPY files  such as OpticksProfile.npy  
    which this Profile class loads.  
    """
    NAME = "OpticksProfile.npy"
    G4DT = ("CRunAction::BeginOfRunAction","CRunAction::EndOfRunAction",)
    OKDT = ("_OPropagator::launch", "OPropagator::launch",)
    PARM = "parameters.json"

    def __init__(self, pdir, name, g4=False ):
        """
        :param pdir: directory from which to load OpticksProfile.npy and siblings, 
                       
             This was formerly tagdir with expected to be the tag integer string, 
             negative for G4 ... hmm but its now one up at the torch dir ?

        :param name: informational name for outputs
        """  
        self.pdir = pdir
        self.name = name
        self.g4 = g4

        self.sli = slice(0,0)  # nowt
        self.fmt = "%Y%m%d-%H%M"

        if os.path.exists(self.pdir):
            self.valid = True
            self.init()
            self.initLaunch()
        else:
            self.valid = False
            self.tim = -1 
            self.prfmt = "INVALID" 
            self.acfmt = "INVALID" 
            log.fatal("pdir %s DOES NOT EXIST " % self.pdir) 
        pass  

    def init(self):
        log.info("[")
        self.loadProfile() 
        self.loadAcc()    ## Accumulated timings 
        self.loadMeta()

        if self.g4:  
            g4r, g40, g41 = self.deltaT(*self.G4DT)   ## CG4::propagate includes significant initialization
            stt = self.setupTrancheTime()
            tim = g4r - stt 
            idx = [g40, g41]  
        else:
            okp, ok0, ok1 = self.deltaT(*self.OKDT)   
            tim = okp
            idx = [ok0, ok1]  
        pass
        self.tim = tim
        self.idx = idx
        self.sli = slice(idx[0],idx[1]+1)
        log.info("]")


    def pfmt(self, path1, path2, path3=None):
        """
        Check that timestamps of related files are close to each other
        """
        t_path1 = time_(path1)
        t_path2 = time_(path2)
        t_path3 = time_(path3) if path3 is not None else None

        flag = 0 

        if t_path3 is not None:
            adt3 = abs(t_path1 - t_path3)
            if adt3.seconds > 1:
                flag += 1  
            pass
            #assert adt3.seconds < 1  
        pass 

        adt = abs(t_path1 - t_path2)
        if adt.seconds > 1:
            flag += 1 
        pass

        if flag > 0:
            log.warning("pfmt related files timestamps differ by more than 1s 1-3: %10.3f 1-2 %10.3f" % ( adt3.seconds, adt.seconds ))
            log.info("path1 %s : %s " % (path1, tfmt_(t_path1)))
            log.info("path2 %s : %s " % (path2, tfmt_(t_path2)))
            log.info("path3 %s : %s " % (path3, tfmt_(t_path3)))
        pass     
        #assert adt.seconds < 1  
        return "  %-90s  %s " % ( path1, t_path1.strftime(self.fmt) )

    def path(self, sfx=""):
        return os.path.join( self.pdir, "OpticksProfile%s.npy" % sfx )   # quads 

    def metapath(self):
        #return os.path.join( self.pdir, "0", self.PARM ) 
        return os.path.join( self.pdir, self.PARM ) 

    def loadMeta(self):
        path = self.metapath()
        exists = os.path.exists(path) 
        if not exists:
            log.info("path %s does not exist " % path ) 
        pass
        meta = json_load_(path) if exists else {}
        self.meta = meta
        log.debug("loaded %s keys %s " % (path, len(self.meta))) 


    def loadProfile(self):
        """
        ::

            In [8]: pr.a                                                                                                                                                                                         
            Out[8]: 
            array([[    0.    , 59888.504 ,     0.    ,  4625.252 ],
                   [    0.    ,     0.    ,     0.    ,     0.    ],
                   [    0.    ,     0.    ,     0.    ,     0.    ],
                   [    0.    ,     0.    ,     0.    ,     0.    ],
                   [    3.8281,     3.8281,   155.98  ,   155.98  ],
                   [    3.8281,     0.    ,   155.98  ,     0.    ],
                   [    3.8281,     0.    ,   155.98  ,     0.    ],
                   [    3.8281,     0.    ,   156.2988,     0.3188],
                   [    3.832 ,     0.0039,   156.8149,     0.5161],

            In [10]: pr.l[:10]                                                                                                                                                                                   
            Out[10]: 
            array([b'OpticksRun::OpticksRun', b'Opticks::Opticks', b'_OKG4Mgr::OKG4Mgr', b'_OpticksHub::init', b'_GMergedMesh::Create', b'GMergedMesh::Create::Count', b'_GMergedMesh::Create::Allocate',
                   b'GMergedMesh::Create::Allocate', b'GMergedMesh::Create::Merge', b'GMergedMesh::Create::Bounds'], dtype='|S45')

        """
        log.info("[")
        path = self.path("")
        lpath = self.path("Labels")
        qpath = self.path("Lis")

        self.prfmt = self.pfmt(path, lpath)
        a = np.load(path)
        assert a.ndim == 2
        log.info("path:%s a.shape:%r" % (path,a.shape)) 

        l = np.load(lpath)
        assert l.ndim == 2 and l.shape[1] == 64
        assert len(a) == len(l) 
        log.info("lpath:%s l.shape:%r" % (lpath,l.shape)) 

        ll = list(l.view("|S64")[:,0])
        lll = np.array(ll)

        t,dt,v,dv = a[:,0], a[:,1], a[:,2], a[:,3]


        q = np.load(qpath) if os.path.exists(qpath) else None
        log.info("qpath:%s q.shape:%r" % (qpath,q.shape)) 

     
        self.l = lll 
        self.a = a 

        self.t = t
        self.dt = dt
        self.v = v
        self.dv = dv

        self.q = q  


        self.apath = path
        self.lpath = lpath

        log.info("]")


    def __len__(self):
        return len(self.a) if self.valid else 0

    def loadAcc(self):
        """
        Acc are accumulated timings 
        """
        log.info("[")
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
        log.info("]")


    def acc(self, label):
        return self.ac[np.where( self.lac == label )]

    def setupTrancheTime(self):
        stt_acc = self.acc("CRandomEngine::setupTranche")
        assert stt_acc.ndim == 2 and stt_acc.shape[1] == 4 
        assert len(stt_acc) == 1 or len(stt_acc) == 0 
        stt = stt_acc[0,1] if len(stt_acc) == 1 else 0.  
        return stt 


    def delta_(self, arg0, arg1=None ):
        """
        :param arg0: start label 
        :param arg1: end label 
        :return w0,w1: arrays of indices of matching labels 

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

        w0 = np.where( self.l == b_(l0) )[0]   # array of matching idx, empty if not found
        w1 = np.where( self.l == b_(l1) )[0]

        return w0, w1

    def dv_(self, i0, i1 ):
        return self.v[i1]-self.v[i0]

    def dt_(self, i0, i1 ):
        return self.t[i1]-self.t[i0]


    def delta(self, arg0, arg1=None):
        """ 
        :param arg0: start label 
        :param arg1: end label 
        :return dt, dv:  deltaTime, deltaVM between two OK_PROFILE code points 
        """
        w0, w1 = self.delta_(arg0,arg1 )  

        i0 = w0[0] if len(w0) == 1 else 0
        i1 = w1[0] if len(w1) == 1 else 0 

        v = self.v 
        t = self.t 

        dv = self.dv_(i0,i1)
        dt = self.dt_(i0,i1)

        log.debug(" i0:%3d i1:%3d  (v0:%10.1f v1:%10.1f dv:%10.1f )  ( t0:%10.4f t1:%10.4f dt:%10.4f )  " % ( i0, i1, self.v[i0],self.v[i1], dv, self.t[i0],self.t[i1], dt  )) 
        return dt, dv, i0, i1  

    def times(self, l0="_OPropagator::launch"):
        """
        :param l0: label of the profile stamp
        :return array of all times matching the label:
        """ 
        pr = self
        tt = pr.t[np.where(pr.l == b_(l0) )] 
        return tt 

    def plt_axvline(self, ax):
        """
        """
        #w0 = np.where( self.l == "_OEvent::uploadSource")[0]
        w1, w2 = self.delta_(*self.OKDT)

        #assert len(w0) == len(w1)
        assert len(w1) == len(w2) 

        for i in range(len(w1)):         
            #i0 = w0[i]
            i1 = w1[i]
            i2 = w2[i]
            #ax.axvline( self.t[i0], c="r" )
            ax.axvline( self.t[i1], c="g" )
            ax.axvline( self.t[i2], c="b" )
        pass

    def deltaT(self, arg0, arg1=None):
        """
        :param arg0:
        :param arg1: 
        :return dt,p0,p1: time between labels and indices of the labels
        """
        dt, dv, p0, p1 = self.delta(arg0, arg1)
        return dt, p0, p1 

    def line(self, i):
        li = d_(self.l[i])
        if li in self.OKDT:
            fn_ = bold_
        elif li in self.G4DT:
            fn_ = blink_
        else:
            fn_ = lambda _:_
        pass
        return " %6d : %50s : %10.4f %10.4f %10.4f %10.4f   " % ( i, fn_(li), self.t[i], self.v[i], self.dt[i], self.dv[i] )

    def labels(self):
        return " %6s : %50s : %10s %10s %10s %10s   " % ( "idx", "label", "t", "v", "dt", "dv" )


    launch_start = property(lambda self:self.times(self.OKDT[0]))
    launch_stop = property(lambda self:self.times(self.OKDT[1]))
    launch = property(lambda self:self.launch_stop - self.launch_start)

    start_interval = property(lambda self:np.diff(self.launch_start))
    stop_interval = property(lambda self:np.diff(self.launch_stop))

    def initLaunch(self):
        """
        intervals only defined with "--multievent 2+" running 
        """
        log.info("[ %r" % self.launch)
        self.avg_launch = np.average(self.launch)
        nsta = len(self.start_interval)  
        nsto = len(self.stop_interval)
        assert nsta == nsto
        if nsta > 0:
            self.multievent = True
            self.avg_start_interval = np.average(self.start_interval)
            self.avg_stop_interval = np.average(self.stop_interval)
            self.overhead_ratio = self.avg_start_interval/self.avg_launch
        else:
            self.avg_start_interval = -1
            self.avg_stop_interval = -1
            self.overhead_ratio = -1
            self.multievent = False
        pass
        #ipdb.set_trace()       
        log.info("]")

    def overheads(self):

        print(".launch_start %r ", self.launch_start )   
        print(".launch_stop  %r ", self.launch_stop  )   
        print(".launch           avg %10.4f     %r " % (  self.avg_launch         , self.launch  ))   

        if self.multievent:
            print(".start_interval   avg %10.4f     %r " % (  self.avg_start_interval , self.start_interval) )   
            print(".stop_interval    avg %10.4f     %r " % (  self.avg_stop_interval  , self.stop_interval ) )   
        pass

    def _get_note(self):
        return "MULTIEVT" if self.multievent else "not-multi"
    note = property(_get_note)


    @classmethod
    def Labels(cls):
        return " %20s %10s %10s %10s %10s : %50s : %50s " % ( "name", "note", "av.interv","av.launch","av.overhd", "launch", "q" )

    def brief(self):
        return " %20s %10s %10.4f %10.4f %10.4f : %50s : %50s " % (self.name, self.note, self.avg_start_interval, self.avg_launch, self.overhead_ratio, fmtarray_(self.launch), fmtarray_(self.q) )


    def bodylines(self):
        start = self.sli.start
        stop = self.sli.stop
        if start is None: start = 0 
        if stop is None: stop = len(self) 

        stop = min(len(self), stop)

        nli = stop - start
        if nli < 10:
            ll = lmap(lambda i:self.line(i), np.arange(start, stop)) 
        else:
            ll = lmap(lambda i:self.line(i), np.arange(start, start+5) ) 
            ll += [" ..."]
            ll += lmap(lambda i:self.line(i), np.arange(stop - 5, stop) )   
        pass
        return ll

    def lines(self):
        return ["name:",self.name, "prfmt:",self.prfmt, "acfmt:",self.acfmt, "sli:","%r" % self.sli, "labels:",self.labels(),"bodylines:"] + self.bodylines() + ["labels:",self.labels()] 

    def __str__(self):
        return "\n".join(self.lines())

    def __repr__(self):
        return self.brief()

    def __getitem__(self, sli):
        self.sli = sli
        return self 



def multievent_plot( pr, plt ):
    """
    TODO: split off the plotting 

    * plotting should never live together with generally usable code, as plotting machinery 
      has much more dependencies and also prone to being swapped for other plotting machinery 
    """
    w0 = np.where(pr.l == b_("_OpticksRun::createEvent"))[0]
    w1 = np.where(pr.l == b_("OpticksRun::resetEvent"))[0]
    assert len(w0) == len(w1)

    w10 = w1 - w0 
    uw10 = np.unique(w10[1:])   # exclude the first, as initialization stamps may be different 
    assert len(uw10) == 1       # number of profile stamps should be the same for all events 
    uw = uw10[0]

    ## each event after the 1st has the same set of labels 
    for j in range(2, len(w0)): assert np.all( pr.l[w0[1]:w1[1]] == pr.l[w0[j]:w1[j]] )

    tt = np.zeros( [9, uw] )
    for j in range(1, len(w0)): tt[j-1] = pr.dt[w0[j]:w1[j]] 

    print("pr[w0[1]:w1[1]]")
    print(pr[w0[1]:w1[1]])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)


    #ax.set_xlim( [pr.t[w0[0]], pr.t[w1[-1]] + 0.5 ])

    ax.set_xlim( 4.9, 5.3 )


    plt.plot( pr.t, pr.v, 'o' )

    pr.plt_axvline(ax) 

    ax.set_ylabel("Process Virtual Memory (MB)")
    ax.set_xlabel("Time since executable start (s)")


    plt.ion()
    fig.show()


def simple_plot( pr, plt):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot( pr.t, pr.v, 'o' )

    ax.set_ylabel("Process Virtual Memory (MB)")
    ax.set_xlabel("Time since executable start (s)")

    plt.ion()
    fig.show()





if __name__ == '__main__':
    from opticks.ana.main import opticks_main
    from opticks.ana.plot import init_rcParams
    import matplotlib.pyplot as plt
    init_rcParams(plt)

    ok = opticks_main(doc=__doc__)  
    log.debug(ok.brief)

    #test_ABProfile( ok, plt )
    log.info("tagdir: %s " % ok.tagdir)

    pr = Profile(ok.tagdir, "pro") 
    log.info("[pr")
    print(pr)
    log.info("]pr")

    if not pr.valid: 
        log.fatal(" pr not valid exiting ")
        sys.exit(0)   
    pass 

    if pr.multievent:
        multievent_plot(pr, plt)  
    pass

    simple_plot( pr, plt )



