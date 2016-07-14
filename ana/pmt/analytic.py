#!/usr/bin/env python

import logging
import numpy as np
from dd import Dddb
from tree import Tree

log = logging.getLogger(__name__)

if __name__ == '__main__':
    format_ = "[%(filename)s +%(lineno)3s %(funcName)20s ] %(message)s" 
    logging.basicConfig(level=logging.INFO, format=format_)
    np.set_printoptions(precision=2) 

    g = Dddb.parse("$PMT_DIR/hemi-pmt.xml")

    lv = g.logvol_("lvPmtHemi")

    tr = Tree(lv)

    parts = tr.parts()

    for pt in parts:
        print pt

    #if hasattr(parts, 'csg') and len(parts.csg) > 0:
    #    for c in parts.csg:
    #        print c  

    buf = tr.convert(parts)

    path = "$IDPATH/GPmt/0/GPmt.npy"
    #path = "$IDPATH/GPmt/0/GPmt_check.npy"

    tr.dump()

    tr.save(path, buf)


