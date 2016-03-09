#!/usr/bin/env python
"""
PmtInBox Opticks vs cfg4 distributions
==========================================

Without and with cfg4 runs::

   ggv-;ggv-pmt-test 
   ggv-;ggv-pmt-test --cfg4 

Visualize the cfg4 created evt in interop mode viewer::

   ggv-;ggv-pmt-test --cfg4 --load

Issues
-------

See pmt_test.py for the history of getting flags and materials into agreement.

TODO
-----

* XY radius plotting
* polx,y,z wavelength  
* push out to more sequences
* auto-handling records for the sequence
* creating multiple pages...


"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from env.numerics.npy.evt import Evt
from env.numerics.npy.nbase import chi2, decompression_bins
from env.numerics.npy.cfplot import cfplot


class CF(object):
    def __init__(self, tag="4", src="torch", det="PmtInBox", seqs=[], subselect=None ):

        self.tag = tag
        self.src = src
        self.det = det
        self.seqs = seqs


        a = Evt(tag="%s" % tag, src=src, det=det, seqs=seqs)
        b = Evt(tag="-%s" % tag , src=src, det=det, seqs=seqs)

        log.info("CF a %s b %s " % (a.label, b.label )) 

        his = a.history.table.compare(b.history.table)
        mat = a.material.table.compare(b.material.table)

        self.a = a
        self.b = b 
        self.his = his
        self.mat = mat

        self.ss = []
        if subselect is not None:
            self.init_subselect(subselect)

    def suptitle(self, irec=-1):
        lab = self.seqlab(irec)
        title = "(%s) %s/%s  :  %s " % (self.tag, self.det, self.src, lab )
        return title

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

    def init_subselect(self, sli):
        """
        Spawn CF for each of the subselections, according to 
        slices of the history sequences.
        """
        for label in self.his.labels[sli]:
            seqs = [label]
            ss = self.spawn(seqs)
            self.ss.append(ss) 

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

    def dump_histories(self):
        print self.his
        #print self.mat

    def dump(self):
        self.dump_ranges(0)
        self.dump_histories()

    def rqwn(self, qwn, irec): 
        a = self.a
        b = self.b
        lval = "%s[%d]" % (qwn.lower(), irec)
        labels = ["Op : %s" % lval, "G4 : %s" % lval]

        if qwn in Evt.RPOST:
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


def qwns_plot(scf, qwns, irec, log_=False):

    fig = plt.figure()

    fig.suptitle(scf.suptitle(irec))

    nx = len(qwns)

    gs = gridspec.GridSpec(2, nx, height_ratios=[3,1])

    for ix in range(nx):

        gss = [gs[ix], gs[nx+ix]]

        qwn = qwns[ix]

        bins, aval, bval, labels = scf.rqwn(qwn, irec)

        log.info("%s %s " % (qwn, repr(labels) ))

        cfplot(fig, gss, bins, aval, bval, labels=labels, log_=log_ )
    pass




def qwn_plot(scf, qwn, irec, log_=False):

    bins, aval, bval, labels = scf.rqwn(qwn, irec)

    fig = plt.figure()
    fig.suptitle(scf.suptitle(irec))

    nx,ix = 1,0
    gs = gridspec.GridSpec(2, nx, height_ratios=[3,1])
    gss = [gs[ix], gs[nx+ix]]

    cfplot(fig, gss, bins, aval, bval, labels=labels, log_=log_ )





if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    np.set_printoptions(precision=4, linewidth=200)

    plt.ion()
    plt.close()

    cf = CF(tag="4", src="torch", det="PmtInBox", subselect=slice(0,3) )
    cf.dump()

    iss = 1   # selection index
    irec = 1

    scf = cf.ss[iss] 

    #qwn_plot( scf, "A", irec)
    #qwns_plot( scf, ["X","Y","Z","T"], irec)
    qwns_plot( scf, ["A","B","C"], irec)


