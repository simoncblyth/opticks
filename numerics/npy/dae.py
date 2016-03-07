#!/usr/bin/env python
"""



"""
import os, sys, logging
import numpy as np
from StringIO import StringIO
import lxml.etree as ET

import matplotlib.pyplot as plt
 
log = logging.getLogger(__name__)

COLLADA_NS='http://www.collada.org/2005/11/COLLADASchema'
tag = lambda _:str(ET.QName(COLLADA_NS,_))
xmlparse_ = lambda _:ET.parse(os.path.expandvars(_)).getroot()
tostring_ = lambda _:ET.tostring(_)

class DAE(object):
    def __init__(self, path):
        self.x = xmlparse_(path)

    def elem_(self, elem, id_):
        q = ".//%s[@id='%s']" % ((tag(elem)), id_)
        e = self.x.find(q)
        return e

    def float_array(self, id_):
        e = self.elem_("float_array", id_ )
        s = StringIO(e.text)
        a = np.loadtxt(s) 
        return a










