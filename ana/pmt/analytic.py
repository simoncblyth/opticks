#!/usr/bin/env python

import os, logging, sys
log = logging.getLogger(__name__)
import numpy as np

from opticks.analytic.treebase import Tree

from opticks.ana.base import opticks_main, Buf
from opticks.ana.pmt.ddbase import Dddb
from opticks.ana.pmt.ddpart import ddpart_manual_mixin
from opticks.ana.pmt.treepart import treepart_manual_mixin


from GPmt import GPmt

if __name__ == '__main__':

    #apmtpathtmpl_default = "$TMP/GPmt/%(apmtidx)s/GPmt.npy"
    #apmtpathtmpl_default = "$IDPATH/GPmt/%(apmtidx)s/GPmt.npy"
    #apmtpathtmpl_default = "$OPTICKS_INSTALL_PREFIX/opticksdata/export/DayaBay/GPmt/%(apmtidx)s/GPmt.npy" 
    #args = opticks_main(apmtpathtmpl=apmtpathtmpl_default, apmtidx=2)

    args = opticks_main(apmtidx=2)

    ddpart_manual_mixin()  # add partitioner methods to Tubs, Sphere, Elem and Primitive
    treepart_manual_mixin() # add partitioner methods to Node and Tree


    apmtpath = args.apmtpath

    print "\nAiming to write serialized analytic PMT to below apmtpath\n%s\n" % apmtpath 

    if args.yes:
        print "proceeding without asking"
    else:
        proceed = raw_input("Enter YES to proceed...  (use eg \"--apmtidx 3\" to write to different index whilst testing, skip dialog with --yes) ... ") 
        if proceed != "YES": sys.exit(1)
    pass 
     
    xmlpath = args.apmtddpath 
    
    log.info("\n\nparsing %s -> %s " % (xmlpath, os.path.expandvars(xmlpath)))

    log.info("\n\nDddb.parse xml \n")
    g = Dddb.parse(xmlpath)

    log.info("\n\ng.logvol \n")
    lv = g.logvol_("lvPmtHemi")

    log.info("\n\nTree(lv) \n")
    tr = Tree(lv)

    parts = tr.parts()

    log.info("\n\nDump parts \n")
    for pt in parts:
        print pt

    assert hasattr(parts, 'gcsg') and len(parts.gcsg) > 0
    log.info("\n\nConvert parts to Buf \n")
    buf = tr.convert(parts)

    log.info("\n\nDump Tree \n")
    tr.dump()

    assert type(buf) is Buf 

    log.info("\n\nmake GPmt from Buf \n")

    gp = GPmt(apmtpath, buf ) 

    log.info("\n\nsave GPmt\n")
    gp.save()   # to apmtpath and sidecars



