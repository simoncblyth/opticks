#!/usr/bin/env python

import os, sys, logging, numpy as np
from opticks.ana.base import opticks_main
from opticks.ana.nbase import chi2, vnorm, decompression_bins
from opticks.ana.evt import Evt
log = logging.getLogger(__name__)


class CF(object):
    def __init__(self, args, select_slice=None):

        self.args = args 
        self.seqs = []
        self.compare()

        self.ss = []
        if select_slice is not None:
            self.init_select(select_slice)

    def compare(self):
        try:
            a = Evt(tag="%s" % self.args.tag, src=self.args.src, det=self.args.det, args=self.args)
            b = Evt(tag="-%s" % self.args.tag , src=self.args.src, det=self.args.det, args=self.args)
        except IOError as err:
            log.fatal(err)
            sys.exit(args.mrc)
      
        print "CF a %s " % a.brief 
        print "CF b %s " % b.brief 

        self.a = a
        self.b = b 

        tables = []

        tables += ["seqhis_ana", "pflags_ana"] 
        if self.args.prohis:
            tables += ["seqhis_ana_%d" % imsk for imsk in range(1,8)] 

        tables += ["seqmat_ana"]       
        if self.args.promat:
            tables += ["seqmat_ana_%d" % imsk for imsk in range(1,8)] 

        tables2 = ["hflags_ana"]
        cft = Evt.compare_table(a,b, tables, lmx=self.args.lmx, cmx=self.args.cmx, c2max=None, cf=True)
        cft2 = Evt.compare_table(a,b, tables2, lmx=self.args.lmx, cmx=self.args.cmx, c2max=None, cf=True)

        self.his = cft["seqhis_ana"]
        self.mat = cft["seqmat_ana"]


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
        title = "%s/%s/%s : %s  :  %s " % (self.args.tag, self.args.det, self.args.src, abc, lab )
        return title

    def nrec(self):
        """
        :return: number of steps, when a single sequence is selected
        """ 
        nseq = len(self.seqs) 
        if nseq != 1:return -1
        seq = self.args.seqs[0]
        eseq = seq.split()
        return len(eseq)

    def seqlab(self, irec=1):
        """
        Sequence label with single record highlighted with a bracket 
        eg  TO BT [BR] BR BT SA 

        """
        nseq = len(self.seqs) 
        if nseq == 1 and irec > -1: 
            seq = self.args.seqs[0]
            eseq = seq.split()
            if irec < len(eseq):
                eseq[irec] = "[%s]" % eseq[irec]
            pass
            lab = " ".join(eseq) 
        else:
            lab = ",".join(self.seqs) 
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
        return "CF(%s,%s,%s,%s) " % (self.args.tag, self.args.src, self.args.det, repr(self.seqs))

    def spawn(self, seqs):
        return CF(self.args.tag, self.args.src, self.args.det, seqs)

    def dump_ranges(self, i):
        log.info("%s : dump_ranges %s " % (repr(self), i) )

        a = self.a
        b = self.b

        ap = a.rpost_(i)
        ar = vnorm(ap[:,:2])
        if len(ar)>0:
            print " ".join(map(lambda _:"%6.3f" % _, (ar.min(),ar.max())))

        bp = b.rpost_(0)
        br = vnorm(bp[:,:2])
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
        """
        :param qwn: X,Y,Z,W,T,A,B,C or R  
        :param irec: step index 0,1,...
        :return binx, aval, bval, labels

        ::

            bi,a,b,l = cf.rqwn("T",4)


        """
        a = self.a
        b = self.b
        lval = "%s[%d]" % (qwn.lower(), irec)
        labels = ["Op : %s" % lval, "G4 : %s" % lval]
 
        if qwn == "R":
            apost = a.rpost_(irec)
            bpost = b.rpost_(irec)
            aval = vnorm(apost[:,:2])
            bval = vnorm(bpost[:,:2])
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
            if rbins is None:
                bins = None
            else: 
                binscale = Evt.RQWN_BINSCALE[qwn]
                if len(rbins) > binscale:
                    bins = rbins[::binscale]
                else:
                    bins = rbins
                pass

        return bins, aval, bval, labels



if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=200)

    args = opticks_main(tag="1", src="torch", det="default")
    log.info(" args %s " % repr(args))

    seqhis_select = slice(1,2)
    #seqhis_select = slice(0,8)
    try:
        cf = CF(tag=args.tag, src=args.src, det=args.det, select=seqhis_select )
    except IOError as err:
        log.fatal(err)
        sys.exit(args.mrc)

    cf.dump()
 
