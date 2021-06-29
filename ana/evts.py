#!/usr/bin/env python 
"""
evts.py
========

Try concatenation of event arrays::

   ipython --pdb -i evts.py -- --pfx tds3gun --src natural 

"""
import json
from opticks.ana.nload import A
from opticks.ana.evt import Evt 

if __name__ == '__main__':
    from opticks.ana.main import opticks_main
    ok = opticks_main()


    mtems = ["fdom", "idom"]
    stems = ["ox", "rx", "bn", "ph"]

    aa = {} 
    bb = {}
    am = {}
    bm = {}
    aj = {}
    bj = {}
 
    for stem in stems: 
        aa[stem] = []
        bb[stem] = []
    pass

    for i in range(7):
        itag = 1+i 

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

        if i == 0:  
            # TODO: should really assert-compare and add 
            aj["parameters.json"] = a.metadata.parameters 
            bj["parameters.json"] = b.metadata.parameters 
        pass

        for mtem in mtems:
            a_ = getattr(a, mtem)
            b_ = getattr(b, mtem)

            if not mtem in am:
                am[mtem] = a_
            else:
                assert np.all( am[mtem] == a_ )
            pass
            if not mtem in bm:
                bm[mtem] = b_
            else:
                assert np.all( bm[mtem] == b_ )
            pass
        pass

        for stem in stems:

            a_ = getattr(a, stem)
            b_ = getattr(b, stem)

            if not a_.missing and len(a_) > 0:
                aa[stem].append(a_)
            pass
            if not b_.missing and len(b_) > 0:
                bb[stem].append(b_)
            pass
        pass
    pass

    jtag = 100
    atag_c = "%s"%jtag 
    btag_c = "-%s"%jtag 

    for mtem in mtems:
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
        np.save(ap, am[mtem] )
        np.save(bp, bm[mtem] )

        last =  mtem == mtems[-1]
        if last:
            for k in aj.keys():
                json.dump(aj[k], open(os.path.join(ad, k), "w"))
                json.dump(bj[k], open(os.path.join(bd, k), "w"))
            pass
        pass
    pass

    pass
    for stem in stems:
        ap = A.path_(stem, ok.src, atag_c, ok.det, ok.pfx)
        bp = A.path_(stem, ok.src, btag_c, ok.det, ok.pfx)

        aa_ = tuple(aa[stem])
        if len(aa_) > 0:
            ac =  np.concatenate(aa_) 
            np.save(ap, ac)
            globals()["a_"+stem] = ac
        pass

        bb_ = tuple(bb[stem])
        if len(bb_) > 0:
            bc =  np.concatenate(bb_) 
            np.save(bp, bc)
            globals()["b_"+stem] = bc
        pass
    pass


