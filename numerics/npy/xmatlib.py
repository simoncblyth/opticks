#!/usr/bin/env python
"""
"""

from collections import OrderedDict
import numpy as np
from env.numerics.npy.dae import DAE, tag_, array_

class XMatLib(dict):
    def __init__(self, path):
        dict.__init__(self)
        self.dae = DAE(path)
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
    xma = XMatLib("/tmp/test.dae")    
    print xma['MineralOil']['GROUPVEL']






