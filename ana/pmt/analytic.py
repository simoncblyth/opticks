#!/usr/bin/env python

import os, logging, sys
log = logging.getLogger(__name__)
import numpy as np
from opticks.ana.base import opticks_main, Buf

from ddbase import Dddb
from ddpart import partitioner_manual_mixin

from tree import Tree
from GPmt import GPmt

if __name__ == '__main__':

    #apmtpathtmpl_default = "$TMP/GPmt/%(apmtidx)s/GPmt.npy"
    #apmtpathtmpl_default = "$IDPATH/GPmt/%(apmtidx)s/GPmt.npy"
    #apmtpathtmpl_default = "$OPTICKS_INSTALL_PREFIX/opticksdata/export/DayaBay/GPmt/%(apmtidx)s/GPmt.npy" 
    #args = opticks_main(apmtpathtmpl=apmtpathtmpl_default, apmtidx=2)

    args = opticks_main(apmtidx=2)

    partitioner_manual_mixin()  # add partitioner methods to Tubs, Sphere, Elem and Primitive


    apmtpath = args.apmtpath

    print "\nAiming to write serialized analytic PMT to below apmtpath\n%s\n" % apmtpath 

    if args.yes:
        print "proceeding without asking"
    else:
        proceed = raw_input("Enter YES to proceed...  (use eg \"--apmtidx 3\" to write to different index whilst testing) ... ") 
        if proceed != "YES": sys.exit(1)
    pass 
     
    xmlpath = args.apmtddpath 
    
    log.info("parsing %s -> %s " % (xmlpath, os.path.expandvars(xmlpath)))

    g = Dddb.parse(xmlpath)

    lv = g.logvol_("lvPmtHemi")

    tr = Tree(lv)

    parts = tr.parts()

    for pt in parts:
        print pt

    assert hasattr(parts, 'gcsg') and len(parts.gcsg) > 0
    buf = tr.convert(parts)

    tr.dump()

    assert type(buf) is Buf 

    gp = GPmt(apmtpath, buf ) 
    gp.save()   # to apmtpath and sidecars



