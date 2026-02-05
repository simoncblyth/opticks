#!/usr/bin/env python

import os, logging
log = logging.getLogger(__name__)

import numpy as np
from opticks.ana.npmeta import NPMeta



class sfr:
    @classmethod
    def Load(cls, fold=os.environ.get("FOLD",""), name="sfr.npy"):
        path = fold if fold.endswith(name) else os.path.join(fold, name)
        return cls(path)

    def __init__(self, path):
        if path is None:
            return
        pass

        metapath = path.replace(".npy", "_meta.txt")
        if os.path.exists(metapath):
            meta = NPMeta.Load(metapath)
        else:
            meta = None
        pass
        self.meta = meta

        a = np.load(path)
        i = a.view(np.uint64)
        assert a.shape == (4,4,4)
        assert i.shape == (4,4,4)

        ce = a[0,0]
        m2w = a[1].copy()
        w2m = a[2].copy()
        _bb = a[3].ravel()

        bbmn = _bb[:3]
        bbmx = _bb[3:6]

        m2w[0,3] = 0.
        m2w[1,3] = 0.
        m2w[2,3] = 0.
        m2w[3,3] = 1.

        w2m[0,3] = 0.
        w2m[1,3] = 0.
        w2m[2,3] = 0.
        w2m[3,3] = 1.

        self.path = path
        self.a = a
        self.i = i
        self.shape = a.shape  # for fold to treat like np.array

        self.ce = ce

        self.m2w = m2w
        self.w2m = w2m
        self.id = np.dot( m2w, w2m )

        self.bbmn = bbmn
        self.bbmx = bbmx


    def __repr__(self):

        l_ = lambda k,v:"%-12s : %s" % (k, v)

        return "\n".join(
                  [
                    l_("sframe",""),
                    l_("path",self.path),
                    l_("meta",repr(self.meta)),
                    l_("ce", repr(self.ce)),
                    l_("bbmn", repr(self.bbmn)),
                    l_("bbmx", repr(self.bbmx)),
                    l_("m2w",""), repr(self.m2w), "",
                    l_("w2m",""), repr(self.w2m), "",
                    l_("id",""),  repr(self.id)
                   ])



if __name__ == '__main__':

    fr = sfr.Load()
    print(fr)




