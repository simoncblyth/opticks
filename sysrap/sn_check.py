#!/usr/bin/env python
"""
sn_check.py
============


"""

import numpy as np

class snd_check(object):
    def __init__(self, f, symbol="c"):
        lvn = f.soname_names
        snd = f.csg.node
        snd_fields = snd.shape[1]
        assert snd_fields == 17
        self.snd = snd


class sn_check(object):
    def __init__(self, f, symbol="c"):
        lvn = f.soname_names
        sn = f._csg.sn
        sn_fields = sn.shape[1]
        assert sn_fields in [17,19]
        with_child = sn_fields == 19

        s_pa = f._csg.s_pa
        s_tv = f._csg.s_tv
        s_bb = f._csg.s_bb

        self.sn = sn
        self.sn_fields = sn_fields
        self.with_child = with_child

        self.s_pa = s_pa
        self.s_tv = s_tv
        self.s_bb = s_bb


        lv = sn[:,2]
        tv = sn[:,3]
        pa = sn[:,4]
        bb = sn[:,5]
        parent = sn[:,6]

        if with_child:
            pass
        else:
            left = sn[:,7]
            right = sn[:,8]

            self.left = left
            self.right = right
            self.ucheck("left")
            self.ucheck("right")
            self.check_left()
            self.check_right()
        pass

        self.lvn = lvn
        self.symbol = symbol

        self.lv = lv
        self.tv = tv
        self.pa = pa
        self.bb = bb
        self.parent = parent

        for field in "lv tv pa bb parent".split():
            self.ucheck(field)
        pass
        self.check_ulv()
        self.check_utv()
        self.check_upa()
        self.check_ubb()
        self.check_parent()

    def ucheck(self, field="lv"):
        uchk = np.unique(getattr(self,field), return_counts=True)
        setattr(self, "u%(field)s" % locals(), uchk[0] )
        setattr(self, "n%(field)s" % locals(), uchk[1] )

    def check_ulv(self):
        assert np.all( self.ulv == np.arange(len(self.ulv)) )
    def check_utv(self):
        assert self.utv.max() < len(self.s_tv), "integrity of transform refs"
    def check_upa(self):
        assert self.upa.max() < len(self.s_pa), "integrity of param refs"
    def check_ubb(self):
        assert self.ubb.max() < len(self.s_bb), "integrity of bbox refs"
    def check_parent(self):
        assert self.parent.max() < len(self.sn), "integrity of parent node refs"
        # assert np.all( self.nparent[1:] == 1 )
        # not fulfilled by the virtual mask polycone

    def check_left(self):
        assert self.left.max() < len(self.sn), "integrity of left node refs"
        assert np.all( self.nleft[1:] == 1 )
    def check_right(self):
        assert self.right.max() < len(self.sn), "integrity of right node refs"
        assert np.all( self.nright[1:] == 1 )


    def __repr__(self):
        sym = self.symbol
        #locals()[sym] = self   ## curious locals worked with older python, now need globals
        globals()[sym] = self
        expr = "np.c_[%(sym)s.ulv, %(sym)s.nlv, %(sym)s.lvn[%(sym)s.ulv]][np.where(%(sym)s.nlv>5)]" % locals()
        lines = []
        lines.append("opticks.sysrap.sn_check")
        lines.append(expr)
        lines.append(str(eval(expr)))
        lines.append("# NB raw nodes from G4VSolid, not complete binary tree counts")
        return "\n".join(lines)

if __name__ == '__main__':
    print("klop")
pass


