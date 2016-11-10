#!/usr/bin/env python

import os, sys, json, logging, numpy as np
from opticks.ana.base import opticks_main, json_
from opticks.ana.nbase import chi2
log = logging.getLogger(__name__)


class CFH(object):
    """
    Persistable comparison histograms and chi2
    The members are numpy arrays and a single ctx dict
    allowing simple load/save.
    """
    NAMES = "bins ahis bhis chi2".split()
    BASE = "$TMP/CFH"

    @classmethod
    def dir_(cls, ctx):
        return os.path.expandvars(os.path.join(cls.BASE,ctx["det"],ctx["tag"],ctx["seq"],ctx["irec"],ctx["qwn"]))

    @classmethod
    def path_(cls, ctx, name):
        dir_ = cls.dir_(ctx)
        return os.path.join(dir_, name)

    @classmethod
    def load_(cls, ctx):
         h = CFH(ctx)  
         h.load()
         return h 

    def path(self, name):
        return self.path_(self.ctx, name)

    def __init__(self, ctx={}):
        self.ctx = ctx 

    def __call__(self, bn, av, bv, lab, cut=30):
        self.bins = bn
        self.ahis,_ = np.histogram(av, bins=bn)
        self.bhis,_ = np.histogram(bv, bins=bn)
        c2, c2n, c2c = chi2(self.ahis.astype(np.float32), self.bhis.astype(np.float32), cut=cut)
        self.chi2 = c2

        meta = {}
        meta['cut'] = cut  
        meta['c2n'] = c2n  
        meta['c2c'] = c2c 
        meta['la'] = lab[0] 
        meta['lb'] = lab[1] 

        self.ctx.update(meta)

    def __repr__(self):
        return "%s[%s]" % (self.ctx['qwn'],self.ctx['irec'])

    def ctxpath(self):
        return self.path("ctx.json") 

    def save(self):
        dir_ = self.dir_(self.ctx)
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        log.info("saving to %s " % dir_)
        json.dump(self.ctx, file(self.ctxpath(),"w") )
        for name in self.NAMES:
            np.save(self.path(name+".npy"), getattr(self, name))
        pass
 

    def load(self):
        dir_ = self.dir_(self.ctx)
        assert os.path.exists(dir_)
        js = json_(self.ctxpath())
        k = map(str, js.keys())
        v = map(str, js.values())
        self.ctx = dict(zip(k,v))
        for name in self.NAMES:
            setattr(self, name, np.load(self.path(name+".npy")))
        pass


if __name__ == '__main__':
    ok = opticks_main(tag="1", src="torch", det="concentric")
    from opticks.ana.cf import CF

    spawn = slice(8,9)  # pluck top line of seqhis table, needed for multiplot
    try:
        cf = CF(ok, spawn=spawn)
    except IOError as err:
        log.fatal(err)
        sys.exit(ok.mrc)

    cf.dump()

    scf = cf.ss[0]

    h = scf.rhist("X", "5")
    h.save()
    hh = CFH.load_(h.ctx) 

 

