#!/usr/bin/env python
"""
tpmt_distrib.py : PmtInBox Opticks vs Geant4 distributions
================================================================

Usage
-------

As this can create many tens of plot windows, a way of wading through them 
without getting finger stain is to resize the invoking ipython window very 
small and then repeatedly run::

   plt.close()

To close each window in turn.

See Also
----------

:doc:`tpmt` 
       history comparison and how to create the events

:doc:`tpmt_debug` 
       development notes debugging simulation to achieve *pmt_test.py* matching

"""
import os, logging, numpy as np
from collections import OrderedDict as odict
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 18,10.2   # plt.gcf().get_size_inches()   after maximize
import matplotlib.gridspec as gridspec

from opticks.ana.base import opticks_environment, opticks_args
from opticks.ana.evt import Evt
from opticks.ana.nbase import chi2, decompression_bins
from opticks.ana.cfplot import cfplot
from env.doc.make_rst_table import recarray_as_rst


class CF(object):
    def __init__(self, tag="4", src="torch", det="PmtInBox", seqs=[], select=None ):

        self.tag = tag
        self.src = src
        self.det = det
        self.seqs = seqs
        self.select = select

        a = Evt(tag="%s" % tag, src=src, det=det, seqs=seqs)
        b = Evt(tag="-%s" % tag , src=src, det=det, seqs=seqs)

        log.info("CF a %s " % (a.brief )) 
        log.info("CF b %s " % (b.brief )) 

        his = a.history.table.compare(b.history.table)
        mat = a.material.table.compare(b.material.table)

        self.a = a
        self.b = b 
        self.his = his
        self.mat = mat

        self.ss = []
        if select is not None:
            self.init_select(select)

    def a_count(self, line=0):
        """subselects usually have only one sequence line""" 
        return self.his.cu[line,1]

    def b_count(self, line=0):
        return self.his.cu[line,2]

    def ab_count(self, line=0):
        ac = self.a_count(line)
        bc = self.b_count(line)
        return "%d/%d" % (ac,bc)

    def suptitle(self, irec=-1):
        abc = self.ab_count()
        lab = self.seqlab(irec)
        title = "%s/%s/%s : %s  :  %s " % (self.tag, self.det, self.src, abc, lab )
        return title

    def nrec(self):
        nseq = len(self.seqs) 
        if nseq != 1:return -1
        seq = self.seqs[0]
        eseq = seq.split()
        return len(eseq)

    def seqlab(self, irec=1):
        """
        Sequence label with single record highlighted with a bracket 
        eg  TO BT [BR] BR BT SA 

        """
        nseq = len(self.seqs) 
        if nseq == 1 and irec > -1: 
            seq = self.seqs[0]
            eseq = seq.split()
            if irec < len(eseq):
                eseq[irec] = "[%s]" % eseq[irec]
            pass
            lab = " ".join(eseq) 
        else:
            lab = ",".join(seqs) 
        pass
        return lab 

    def init_select(self, sli):
        """
        Spawn CF for each of the selections, according to 
        slices of the history sequences.
        """
        totrec = 0 
        for label in self.his.labels[sli]:
            seqs = [label]
            scf = self.spawn(seqs)
            totrec += scf.nrec() 
            self.ss.append(scf) 
        pass
        self.totrec = totrec

    def __repr__(self):
        return "CF(%s,%s,%s,%s) " % (self.tag, self.src, self.det, repr(self.seqs))

    def spawn(self, seqs):
        return CF(self.tag, self.src, self.det, seqs)

    def dump_ranges(self, i):
        log.info("%s : dump_ranges %s " % (repr(self), i) )

        a = self.a
        b = self.b

        ap = a.rpost_(i)
        ar = np.linalg.norm(ap[:,:2],2,1)
        if len(ar)>0:
            print " ".join(map(lambda _:"%6.3f" % _, (ar.min(),ar.max())))

        bp = b.rpost_(0)
        br = np.linalg.norm(bp[:,:2],2,1)
        if len(br)>0:
            print " ".join(map(lambda _:"%6.3f" % _, (br.min(),br.max())))

    def dump_histories(self, lmx=20):
        if len(self.his.lines) > lmx:
            self.his.sli = slice(0,lmx)
        if len(self.mat.lines) > lmx:
            self.mat.sli = slice(0,lmx)

        print "\n",self.his
        print "\n",self.mat

    def dump(self):
        self.dump_ranges(0)
        self.dump_histories()

    def rqwn(self, qwn, irec): 
        a = self.a
        b = self.b
        lval = "%s[%d]" % (qwn.lower(), irec)
        labels = ["Op : %s" % lval, "G4 : %s" % lval]
 
        if qwn == "R":
            apost = a.rpost_(irec)
            bpost = b.rpost_(irec)
            aval = np.linalg.norm(apost[:,:2],2,1)
            bval = np.linalg.norm(bpost[:,:2],2,1)
            cbins = a.pbins()
        elif qwn in Evt.RPOST:
            q = Evt.RPOST[qwn]
            aval = a.rpost_(irec)[:,q]
            bval = b.rpost_(irec)[:,q]
            if qwn == "T":
                cbins = a.tbins()
            else:
                cbins = a.pbins()
        elif qwn in Evt.RPOL:
            q = Evt.RPOL[qwn]
            aval = a.rpol_(irec)[:,q]
            bval = b.rpol_(irec)[:,q]
            cbins = a.rpol_bins()
        else:
            assert 0, "qwn %s unknown " % qwn 
        pass

        if qwn in "ABC":
            # polarization is char compressed so have to use primordial bins
            bins = cbins
        else: 
            rbins = decompression_bins(cbins, aval, bval)
            binscale = Evt.RQWN_BINSCALE[qwn]
            if len(rbins) > binscale:
                bins = rbins[::binscale]
            else:
                bins = rbins
            pass

        return bins, aval, bval, labels


def qwns_plot(scf, qwns, irec, log_=False, c2_cut=30):

    fig = plt.figure()

    if irec < 0:
         irec += scf.nrec() 

    fig.suptitle(scf.suptitle(irec))

    nx = len(qwns)

    gs = gridspec.GridSpec(2, nx, height_ratios=[3,1])

    c2ps = []
    for ix in range(nx):

        gss = [gs[ix], gs[nx+ix]]

        qwn = qwns[ix]

        bins, aval, bval, labels = scf.rqwn(qwn, irec)

        log.info("%s %s " % (qwn, repr(labels) ))

        c2p = cfplot(fig, gss, bins, aval, bval, labels=labels, log_=log_, c2_cut=c2_cut )

        c2ps.append(c2p) 
    pass

    qd = odict(zip(list(qwns),c2ps))
    return qd


def qwn_plot(scf, qwn, irec, log_=False, c2_cut=30, c2_ymax=10):

    if irec < 0:
         irec += scf.nrec() 

    bins, aval, bval, labels = scf.rqwn(qwn, irec)

    fig = plt.figure()
    fig.suptitle(scf.suptitle(irec))

    nx,ix = 1,0
    gs = gridspec.GridSpec(2, nx, height_ratios=[3,1])
    gss = [gs[ix], gs[nx+ix]]

    c2p = cfplot(fig, gss, bins, aval, bval, labels=labels, log_=log_, c2_cut=c2_cut, c2_ymax=c2_ymax)
    c2ps = [c2p]

    #print "c2p", c2p

    qd = odict(zip(list(qwn),c2ps))
    return qd




def multiplot(cf, pages=["XYZT","ABCR"]):

    qwns = "".join(pages)
    dtype = [("key","|S64")] + [(q,np.float32) for q in list(qwns)]

    log_ = False
    c2_cut = 0.

    stat = np.recarray((cf.totrec,), dtype=dtype)

    ival = 0 
    for scf in cf.ss:
        nrec = scf.nrec()
        for irec in range(nrec):
            key = scf.suptitle(irec)

            od = odict()
            od.update(key=key) 

            for page in pages:
                qd = qwns_plot( scf, page, irec, log_, c2_cut)
                od.update(qd)
            pass

            stat[ival] = tuple(od.values())
            ival += 1
        pass
    pass

    np.save(os.path.expandvars("$TMP/stat.npy"),stat)

    rst = recarray_as_rst(stat)
    print rst 




if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=200)
    opticks_environment()
    args = opticks_args(tag="4", src="torch", det="PmtInBox")

    plt.ion()
    plt.close()

    select = slice(1,2)
    #select = slice(0,8)

    cf = CF(tag=args.tag, src=args.src, det=args.det, select=select )
    cf.dump()
    
    multiplot(cf, pages=["XYZT","ABCR"])
  
    #qwn_plot( cf.ss[0], "T", -1, c2_ymax=2000)

    #qwn_plot( scf, "R", irec)
    #qwns_plot( scf, "XYZT", irec)
    #qwns_plot( scf, "ABCR", irec)








