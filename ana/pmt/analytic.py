#!/usr/bin/env python

import os, logging, sys
log = logging.getLogger(__name__)
import numpy as np
from opticks.ana.base import opticks_main, Buf

from dd import Dddb
from tree import Tree
from GPmt import GPmt

if __name__ == '__main__':

    #apmtpathtmpl_default = "$TMP/GPmt/%(apmtidx)s/GPmt.npy"
    #apmtpathtmpl_default = "$IDPATH/GPmt/%(apmtidx)s/GPmt.npy"
    #apmtpathtmpl_default = "$OPTICKS_INSTALL_PREFIX/opticksdata/export/DayaBay/GPmt/%(apmtidx)s/GPmt.npy" 
    #args = opticks_main(apmtpathtmpl=apmtpathtmpl_default, apmtidx=2)

    args = opticks_main(apmtidx=2)

    apmtpath = args.apmtpath

    print "\nAiming to write serialized analytic PMT to below apmtpath\n%s\n" % apmtpath 
    proceed = raw_input("Enter YES to proceed... ") 
    if proceed != "YES": sys.exit(1)
 

    xmlpath = "$PMT_DIR/hemi-pmt.xml"
    log.info("parsing %s -> %s " % (xmlpath, os.path.expandvars(xmlpath)))

    g = Dddb.parse(xmlpath)

    lv = g.logvol_("lvPmtHemi")

    tr = Tree(lv)

    parts = tr.parts()

    for pt in parts:
        print pt

    assert hasattr(parts, 'csg') and len(parts.csg) > 0

    buf = tr.convert(parts, analytic_version=args.apmtidx)
  
    tr.dump()


    gp = GPmt(apmtpath, buf )
    gp.save()



