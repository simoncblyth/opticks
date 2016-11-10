#!/usr/bin/env python

import os, sys, logging, numpy as np
from opticks.ana.base import opticks_main
from opticks.ana.cfh import CFH
from opticks.ana.nbase import chi2, vnorm
from opticks.ana.decompression import decompression_bins
from opticks.ana.histype import HisType
from opticks.ana.mattype import MatType
from opticks.ana.evt import Evt
log = logging.getLogger(__name__)


class CF(object):
    def __init__(self, args, seqs=[], spawn=None, top=True):
        """
        :param args:
        :param seqs: used beneath top level 
        :param spawn:  only used from top level cf
        """
        self.args = args 
        self.seqs = seqs
        self.top = top
        self.af = HisType()
        self.mt = MatType()

        self.compare(seqs)

        self.ss = []
        self.init_spawn(spawn)

    def init_spawn(self, spawn, flv="seqhis"):
        """
        Spawn CF for each of the selections, according to 
        slices of the history sequences.

        ::

            In [29]: cf.his.labels[:10]
            Out[29]: 
            ['TO BT BT BT BT SA',
             'TO AB',
             'TO SC BT BT BT BT SA',
             'TO BT BT BT BT AB',
             'TO BT BT AB',
             'TO RE BT BT BT BT SA',
             'TO BT BT SC BT BT SA',
             'TO BT BT BT BT SC SA',
             'TO BT BT BT BT DR SA',
             'TO RE RE BT BT BT BT SA']

        """
        if spawn is None:
            return 

        assert self.top == True, "spawn is only allowed at top level "

        totrec = 0 

        if type(spawn) is slice:
            if flv == "seqhis":
                labels = self.his.labels[spawn] 
            elif flv == "seqmat":
                labels = self.mat.labels[spawn] 
            else:
                assert 0, flv
            pass
        elif type(spawn) is list:
            # elements of spawn can be hexint, hexstring(without 0x), or preformed labels  
            # a single wildcarded label also supported eg "TO BT BT SC .."
            if flv == "seqhis":
                labels = map( lambda _:self.af.label(_), spawn)
            elif flv == "seqmat":
                labels = map( lambda _:self.mt.label(_), spawn)
            else:
                assert 0, flv
        else:
            log.fatal("spawn argument must be a slice or list of seqs") 
            assert 0
        pass

        for label in labels:
            seqs = [label]
            scf = self.spawn(seqs)
            totrec += scf.nrec() 
            self.ss.append(scf) 
        pass
        self.totrec = totrec

    def spawn(self, seqs):
        scf = CF(self.args, seqs, spawn=None, top=False)
        scf.parent = self
        scf.his = self.his
        scf.mat = self.mat
        return scf

    def compare(self, seqs=[]):
        try:
            a = Evt(tag="%s" % self.args.tag, src=self.args.src, det=self.args.det, args=self.args, seqs=seqs)
            b = Evt(tag="-%s" % self.args.tag , src=self.args.src, det=self.args.det, args=self.args, seqs=seqs)
        except IOError as err:
            log.fatal(err)
            sys.exit(args.mrc)
      
        self.a = a
        self.b = b 

        if self.top:
            self.fullcompare() 
        else:
            log.info("spawned seqs %s psel A %d B %d " % (repr(seqs), a.nsel, b.nsel ))
            self.fullcompare() 
        pass

    def fullcompare(self):
        a = self.a
        b = self.b
        print "CF a %s " % a.brief 
        print "CF b %s " % b.brief 

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

    def cf(self, ana="seqhis_ana"):
        a = self.a
        b = self.b
        print "CF a %s " % a.brief 
        print "CF b %s " % b.brief 
        c_tab = Evt.compare_ana( a, b, ana, lmx=self.args.lmx, cmx=self.args.cmx, c2max=None, cf=True)
        return c_tab


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

    def ctx(self, **kwa):
        irec = kwa.get('irec',"0")
        lab = self.seqlab(irec)
        d = {'det':self.args.det, 'tag':self.args.tag, 'src':self.args.src, 'seq':self.seqs[0].replace(" ","_"), 'lab':lab }
        d.update(kwa)
        return d 

    def nrec(self):
        """
        :return: number of steps, when a single sequence is selected
        """ 
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
            lab = ",".join(self.seqs) 
        pass
        return lab 

    def __repr__(self):
        return "CF(%s,%s,%s,%s) " % (self.args.tag, self.args.src, self.args.det, repr(self.seqs))
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

    def rpost(self):
        """
        Sliced decompression within a selection 
        works via the rx replacement with rx[psel] 
        done in Evt.init_selection.

        ::

            In [2]: scf = cf.ss[0]

            In [3]: a,b=scf.xyzt()

            In [4]: a
            Out[4]: 
            A([    [[    0.    ,     0.    ,     0.    ,     0.0998],
                    [ 2995.0267,     0.    ,     0.    ,    15.0472],
                    [ 3004.9551,     0.    ,     0.    ,    15.0975],
                    [ 3995.0491,     0.    ,     0.    ,    20.0378],
                    [ 4004.9776,     0.    ,     0.    ,    20.0882],
                    [ 4995.0716,     0.    ,     0.    ,    24.9544]],

                   [[    0.    ,     0.    ,     0.    ,     0.0998],
                    [ 2995.0267,     0.    ,     0.    ,    15.0472],
                    [ 3004.9551,     0.    ,     0.    ,    15.0975],
                    [ 3995.0491,     0.    ,     0.    ,    20.0378],
                    [ 4004.9776,     0.    ,     0.    ,    20.0882],
                    [ 4995.0716,     0.    ,     0.    ,    24.9544]],

            In [5]: b
            Out[5]: 
            A([    [[    0.    ,     0.    ,     0.    ,     0.0998],
                    [ 2995.0267,     0.    ,     0.    ,    15.4775],
                    [ 3004.9551,     0.    ,     0.    ,    15.5296],
                    [ 3995.0491,     0.    ,     0.    ,    20.6687],
                    [ 4004.9776,     0.    ,     0.    ,    20.7199],
                    [ 4995.0716,     0.    ,     0.    ,    25.8589]],

                   [[    0.    ,     0.    ,     0.    ,     0.0998],
                    [ 2995.0267,     0.    ,     0.    ,    15.4775],
                    [ 3004.9551,     0.    ,     0.    ,    15.5296],
                    [ 3995.0491,     0.    ,     0.    ,    20.6687],
                    [ 4004.9776,     0.    ,     0.    ,    20.7199],
                    [ 4995.0716,     0.    ,     0.    ,    25.8589]],

            In [6]: a.shape
            Out[6]: (669843, 6, 4)

            In [7]: b.shape
            Out[7]: (670752, 6, 4)

            In [8]: scf.a.seqs
            Out[8]: ['TO BT BT BT BT SA']

            In [9]: scf.a.psel
            Out[9]: array([ True,  True, False, ...,  True,  True, False], dtype=bool)

            In [10]: scf.a.psel.shape
            Out[10]: (1000000,)

        """
        self.checkstep()
        aval = self.a.rpost()
        bval = self.b.rpost()
        return aval, bval


    def checkstep(self):
        nstep = self.a.nstep()
        nstep2 = self.b.nstep()
        assert nstep == nstep2
        if nstep == -1:
            log.fatal("fixed step slicing only works on cf with single line seqs eg cf.ss[0]")
            assert 0

    def rdir(self, fr=0, to=1):
        aval = self.a.rdir(fr,to)
        bval = self.b.rdir(fr,to)
        return aval, bval

    def rpol_(self, fr):
        aval = self.a.rpol_(fr)
        bval = self.b.rpol_(fr)
        return aval, bval

    def rpol(self):
        self.checkstep()
        aval = self.a.rpol()
        bval = self.b.rpol()
        return aval, bval


    def rw(self):
        self.checkstep()
        aval = self.a.rw()
        bval = self.b.rw()
        return aval, bval

    def rhist(self, qwn, irec, cut=30): 
        bn, av, bv, la = self.rqwn(qwn, irec)
        ctx=self.ctx(qwn=qwn,irec=irec)
        cfh = CFH(ctx)
        cfh(bn,av,bv,la,cut=cut)
        return cfh
 
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
        lval = "%s[%s]" % (qwn.lower(), irec)
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
            pass
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
            bins = decompression_bins(cbins, [aval, bval], label=lval, binscale=Evt.RQWN_BINSCALE[qwn] )


        if len(bins) == 0:
            raise Exception("no bins")

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
 
