#!/usr/bin/env python 
"""
evts.py
========

Try concatenation of event arrays::

   ipython --pdb -i evts.py -- --pfx tds3gun --src natural 

   OR

   ./evts.sh 

"""
import json
from opticks.ana.nload import A
from opticks.ana.evt import Evt 

class EvtConcatenate(object):
    mtems = ["fdom", "idom"]
    stems = ["ox", "rx", "bn", "ph"]

    def __init__(self, ok):

        self.ok = ok

        self.aa = {}   # content arrays : ox, rx, bn, ph
        self.bb = {}
        self.am = {}   # metadata arrays : idom, fdom
        self.bm = {} 
        self.aj = {}   # json metadata
        self.bj = {} 

        for stem in self.stems: 
            self.aa[stem] = []
            self.bb[stem] = []
        pass

    def load_all(self, n):
        for i in range(n):
            itag = 1+i  
            self.load_one(itag)
        pass

    def load_one(self, itag):

        ok = self.ok
        aname = "a%d" % itag
        bname = "b%d" % itag

        atag = "%s"% itag
        btag = "-%s"% itag

        a = Evt(tag=atag, src=ok.src, det=ok.det, pfx=ok.pfx, args=ok)
        b = Evt(tag=btag, src=ok.src, det=ok.det, pfx=ok.pfx, args=ok)

        ap = A.path_("ox", ok.src, atag, ok.det, ok.pfx)
        bp = A.path_("ox", ok.src, btag, ok.det, ok.pfx)

        print(ap) 
        print(a.ox.path) 

        print(bp) 
        print(b.ox.path) 

        globals()[aname] = a  
        globals()[bname] = b  

        if itag == 1:  
            # TODO: should really assert-compare and add 
            self.aj["parameters.json"] = a.metadata.parameters 
            self.bj["parameters.json"] = b.metadata.parameters 
        pass

        ## necessary metadata arrays : fdom, idom
        for mtem in self.mtems:
            a_ = getattr(a, mtem)
            b_ = getattr(b, mtem)

            if not mtem in self.am:
                self.am[mtem] = a_
            else:
                assert np.all( self.am[mtem] == a_ )
            pass
            if not mtem in self.bm:
                self.bm[mtem] = b_
            else:
                assert np.all( self.bm[mtem] == b_ )
            pass
        pass

        ## content arrays : ox, rx, ph, ...
        for stem in self.stems:

            a_ = getattr(a, stem)
            b_ = getattr(b, stem)

            if not a_.missing and len(a_) > 0:
                self.aa[stem].append(a_)
            pass
            if not b_.missing and len(b_) > 0:
                self.bb[stem].append(b_)
            pass
        pass
    pass

    def save_concat(self, jtag):
        log.info("save_concat jtag:%d" % jtag)
        ok = self.ok
        atag_c = "%s"%jtag 
        btag_c = "-%s"%jtag 

        for mtem in self.mtems:
            ap = A.path_(mtem, ok.src, atag_c, ok.det, ok.pfx)
            bp = A.path_(mtem, ok.src, btag_c, ok.det, ok.pfx)
            ad = os.path.dirname(ap)
            bd = os.path.dirname(bp)
            if not os.path.isdir(ad):
                os.makedirs(ad)
            pass
            if not os.path.isdir(bd):
                os.makedirs(bd)
            pass
            log.info("save %s " % ap)
            np.save(ap, self.am[mtem] )
            log.info("save %s " % bp)
            np.save(bp, self.bm[mtem] )

            last =  mtem == self.mtems[-1]
            if last:
                for k in self.aj.keys():  ## keys are file names eg parameters.json
                    json.dump(self.aj[k], open(os.path.join(ad, k), "w"))
                    json.dump(self.bj[k], open(os.path.join(bd, k), "w"))
                pass
            pass
        pass

        for stem in self.stems:
            ap = A.path_(stem, ok.src, atag_c, ok.det, ok.pfx)
            bp = A.path_(stem, ok.src, btag_c, ok.det, ok.pfx)

            aa_ = tuple(self.aa[stem])
            if len(aa_) > 0:
                ac =  np.concatenate(aa_) 
                log.info("save %s " % ap)
                np.save(ap, ac)
                globals()["a_"+stem] = ac
            pass

            bb_ = tuple(self.bb[stem])
            if len(bb_) > 0:
                bc =  np.concatenate(bb_) 
                log.info("save %s " % bp)
                np.save(bp, bc)
                globals()["b_"+stem] = bc
            pass
        pass


if __name__ == '__main__':
    from opticks.ana.main import opticks_main
    ok = opticks_main()
    ec = EvtConcatenate(ok)
    ec.load_all(7)           # load tags 1 to 7 
    ec.save_concat(100)      # save into new tag 100

