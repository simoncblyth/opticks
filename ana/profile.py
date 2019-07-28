#!/usr/bin/env python
"""
profile.py
=============

::

    LV=0 ip profile.py 

    LV=box ip profile.py --cat cvd_1_rtx_1_1M --pfx scan-ph --tag 0


"""

from __future__ import print_function
import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.log import bold_, blink_
from opticks.ana.nload import np_load
from opticks.ana.nload import tagdir_, stmp_, time_

from IPython.core.debugger import Pdb
ipdb = Pdb()


class Profile(object):

    NAME = "OpticksProfile.npy"
    G4DT = ("CRunAction::BeginOfRunAction","CRunAction::EndOfRunAction",)
    OKDT = ("_OPropagator::launch", "OPropagator::launch",)

    def __init__(self, tagdir, name ):
        """
        :param tagdir: directory from which to load OpticksProfile.npy and siblings
        :param name: informational name for outputs
        """  
        self.tagdir = tagdir
        tag = os.path.basename(tagdir)
        g4 = tag[0] == "-"
        log.debug(" tagdir:%s name:%s tag:%s g4:%s " % (tagdir, name, tag, g4))

        self.name = name
        self.tag = tag
        self.g4 = g4

        self.sli = slice(0,0)  # nowt
        self.fmt = "%Y%m%d-%H%M"

        if os.path.exists(self.tagdir):
            self.valid = True
            self.init()
            self.initLaunch()
        else:
            self.valid = False
            self.tim = -1 
            self.prfmt = "INVALID" 
            self.acfmt = "INVALID" 
            log.fatal("tagdir %s DOES NOT EXIST " % self.tagdir) 
        pass  

    def init(self):
        self.loadProfile() 
        self.loadAcc()

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


    def __len__(self):
        return len(self.a) if self.valid else 0

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

        w0 = np.where( self.l == l0 )[0]   # array of matching idx, empty if not found
        w1 = np.where( self.l == l1 )[0]

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
        """ 
        pr = self
        tt = pr.t[np.where(pr.l == l0 )] 
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
        dt, dv, p0, p1 = self.delta(arg0, arg1)
        return dt, p0, p1 

    def line(self, i):
        li = self.l[i]
        if li in self.OKDT:
            fn_ = bold_
        elif li in self.G4DT:
            fn_ = blink_
        else:
            fn_ = lambda _:_
        pass
        return " %6d : %50s : %10.4f %10.4f %10.4f %10.4f   " % ( i, fn_(self.l[i]), self.t[i], self.v[i], self.dt[i], self.dv[i] )

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
        log.debug("[")
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
        log.debug("]")

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
        return " %20s %10s %15s %15s %15s %50s " % ( "name", "note", "avg interval","avg launch","avg overhead", "launch" )

    def brief(self):
        return " %20s %10s %10.4f %10.4f %10.4f %50s " % (self.name, self.note, self.avg_start_interval, self.avg_launch, self.overhead_ratio, repr(self.launch) )


    def bodylines(self):
        start = self.sli.start
        stop = self.sli.stop
        if start is None: start = 0 
        if stop is None: stop = len(self) 

        stop = min(len(self), stop)

        nli = stop - start
        if nli < 10:
            ll = map(lambda i:self.line(i), np.arange(start, stop)) 
        else:
            ll = map(lambda i:self.line(i), np.arange(start, start+5) ) 
            ll += [" ..."]
            ll += map(lambda i:self.line(i), np.arange(stop - 5, stop) )   
        pass
        return ll

    def lines(self):
        return [self.name, self.prfmt, self.acfmt, "%r" % self.sli, self.labels()] + self.bodylines() + [self.labels()] 


    def __str__(self):
        return "\n".join(self.lines())

    def __repr__(self):
        return self.brief()

    def __getitem__(self, sli):
        self.sli = sli
        return self 



class ABProfile(object):
    def __init__(self, atagdir, btagdir):
        self.ap = Profile(atagdir, "ab.pro.ap" ) 
        self.bp = Profile(btagdir, "ab.pro.bp" ) 
        valid = self.ap.valid and self.bp.valid 
        if valid:
            boa = self.bp.tim/self.ap.tim if self.ap.tim > 0 else -1  
        else:
            boa = -2 
        pass 
        self.boa = boa
  
    def brief(self): 
        return "      ap.tim %-10.4f     bp.tim %-10.4f      bp.tim/ap.tim %-10.4f    " % (self.ap.tim, self.bp.tim, self.boa )   

    def __repr__(self):
        return "\n".join(["ab.pro", self.brief()] + self.ap.lines() + self.bp.lines() )






def test_ABProfile(ok, plt):

    op = ABProfile(ok.tagdir, ok.ntagdir)   # assumes ok/g4 
    print(op)

    ap = op.ap
    bp = op.bp

    plt.plot( ap.t, ap.v, 'o' )
    plt.plot( bp.t, bp.v, 'o' )

    plt.axvline( ap.t[ap.idx[0]], c="b" )
    plt.axvline( ap.t[ap.idx[1]], c="b" )

    plt.axvline( bp.t[bp.idx[0]], c="r" )
    plt.axvline( bp.t[bp.idx[1]], c="r" )

    plt.ion()
    plt.show()



def multievent_plot( pr, plt ):

    w0 = np.where(pr.l == "_OpticksRun::createEvent")[0]
    w1 = np.where(pr.l == "OpticksRun::resetEvent")[0]
    assert len(w0) == len(w1)

    w10 = w1 - w0 
    uw10 = np.unique(w10[1:])
    assert len(uw10) == 1
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
    print(pr)

    if not pr.valid: 
        log.fatal(" pr not valid exiting ")
        sys.exit(0)   
    pass 

    if pr.multievent:
        multievent_plot(pr, plt)  
    pass



