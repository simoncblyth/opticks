#!/usr/bin/env python
"""
"""
import logging, os, numpy as np
from collections import OrderedDict
from opticks.ana.dae import DAE, tag_, array_

class XMatLib(dict):
    def __init__(self, daepath="$OPTICKS_DAEPATH"):
        dict.__init__(self)
        self.dae = DAE(os.path.expandvars(daepath))
        self.init()

    def init(self):
        xms = self.dae.elems_("material")
        for xm in xms:
            matname = DAE.deref(xm.attrib['id'])
            props = xm.findall(".//%s" % tag_("matrix"))
            pd = OrderedDict()
            for prop in props:
                propname = DAE.deref(prop.attrib['name'])
                pd[propname] = array_(prop).reshape(-1,2)
            pass
            self[matname] = pd 



if __name__ == '__main__':

    #xma = XMatLib("/tmp/test.dae")    
    xma = XMatLib()    
    print xma['MineralOil']['GROUPVEL']






