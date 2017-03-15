#!/usr/bin/env python

import os, logging
log = logging.getLogger(__name__)
import numpy as np
from opticks.ana.base import opticks_main, Buf

from dd import Dddb
from tree import Tree
from GPmt import GPmt

if __name__ == '__main__':
    args = opticks_main(apmtpath="$IDPATH/GPmt/0/GPmt.npy")

    xmlpath = "$PMT_DIR/hemi-pmt.xml"
    log.info("parsing %s -> %s " % (xmlpath, os.path.expandvars(xmlpath)))

    g = Dddb.parse(xmlpath)

    lv = g.logvol_("lvPmtHemi")

    tr = Tree(lv)

    parts = tr.parts()

    for pt in parts:
        print pt

    assert hasattr(parts, 'csg') and len(parts.csg) > 0

    buf = tr.convert(parts)
  
    tr.dump()

    #path = "$IDPATH/GPmt/0/GPmt.npy"
    #path = "$IDPATH/GPmt/0/GPmt_check.npy"
    #path = "$TMP/GPmt/0/GPmt.npy"
    path = args.apmtpath

    gp = GPmt(path, buf )
    gp.save()



