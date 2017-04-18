#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.pmt.ddbase import Dddb
from opticks.ana.pmt.treebase import Tree
from opticks.ana.pmt.ncsgtranslator import NCSGTranslator

from opticks.dev.csg.csg import CSG  


if __name__ == '__main__':


    args = opticks_main()

    g = Dddb.parse(args.apmtddpath)

    lv = g.logvol_("lvPmtHemi")
    tr = Tree(lv)

    container = CSG("box", param=[0,0,0,1000], boundary="dummy", poly="IM", resolution="20")

    objs = []
    objs.append(container)

    nn = tr.num_nodes()
    assert nn == 5

    im = dict(poly="IM", resolution="30")
    mc = dict(poly="MC", nx="30")
    dcs = dict(poly="DCS", nominal="7", coarse="6", threshold="1", verbosity="0")
    poly = im


    ii = range(nn)
    #ii = range(2,3)

    for i in ii:
        root = tr.get(i)
        log.info("\ntranslating ..........................  %r " % root )

        obj = NCSGTranslator.TranslateLV( root.lv )
        obj.boundary = "dummy"
        obj.meta.update(poly)  
        objs.append(obj)
        CSG.Dump(obj)
    pass


    CSG.Serialize(objs, "$TMP/ncsgtranslator_test" )








