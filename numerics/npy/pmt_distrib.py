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

* push out to more sequences
* auto-handling records for the sequence
* creating multiple pages...
* polx,y,z wavelength  

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

        seqlab = ",".join(seqs) 
        suptitle = "(%s) %s %s %s " % (tag, det, src, seqlab )

        a = Evt(tag="%s" % tag, src=src, det=det, seqs=seqs)
        b = Evt(tag="-%s" % tag , src=src, det=det, seqs=seqs)

        log.info("CF a %s b %s " % (a.label, b.label )) 

        his = a.history.table.compare(b.history.table)
        mat = a.material.table.compare(b.material.table)

        self.suptitle = suptitle
        self.a = a
        self.b = b 
        self.his = his
        self.mat = mat

        self.ss = []
        if subselect is not None:
            self.init_subselect(subselect)

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

    def rpost(self, qwn, irec): 
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
        else:
            assert 0, "qwn %s unknown " % qwn 
        pass
        return cbins, aval, bval, labels
 


def qwns_plot(scf, qwns, irec, log_=False):

    fig = plt.figure()
    fig.suptitle(scf.suptitle)

    nx = len(qwns)
    gs = gridspec.GridSpec(2, nx, height_ratios=[3,1])

    for ix in range(nx):

        gss = [gs[ix], gs[nx+ix]]

        qwn = qwns[ix]

        binscale = Evt.RPOST_BINSCALE[qwn]

        cbins, aval, bval, labels = scf.rpost(qwn, irec)

        log.info("%s %s " % (qwns[ix], repr(labels) ))

        rbins = decompression_bins(cbins, aval, bval)
        if len(rbins) > binscale:
            bins = rbins[::binscale]
        else:
            bins = rbins

        cfplot(fig, gss, bins, aval, bval, labels=labels, log_=log_ )
    pass


def qwn_plot(scf, qwn, irec, log_=False):

    cbins, aval, bval, labels = scf.rpost(qwn, irec)

    rbins = decompression_bins(cbins, aval, bval)
    binscale = Evt.RPOST_BINSCALE[qwn]

    if len(rbins) > binscale:
        bins = rbins[::binscale]
    else:
        bins = rbins

    fig = plt.figure()
    fig.suptitle(scf.suptitle)

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

    scf = cf.ss[iss] 
    irec = 2

    #qwn_plot( scf, "T", irec)
    qwns_plot( scf, ["X","Y","Z","T"], irec)


