#!/usr/bin/env python
#
# Copyright (c) 2019 Opticks Team. All Rights Reserved.
#
# This file is part of Opticks
# (see https://bitbucket.org/simoncblyth/opticks).
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and 
# limitations under the License.
#

"""
DAE
=====

Simple XML parsing of COLLADA, for debug access to XML elements.  


"""
import os, sys, logging
import numpy as np
from StringIO import StringIO
import lxml.etree as ET

import opticks.ana.base
 
log = logging.getLogger(__name__)

COLLADA_NS='http://www.collada.org/2005/11/COLLADASchema'
tag_ = lambda _:str(ET.QName(COLLADA_NS,_))
xmlparse_ = lambda _:ET.parse(os.path.expandvars(_)).getroot()
tostring_ = lambda _:ET.tostring(_)
array_ = lambda e:np.loadtxt(StringIO(e.text))


class DAE(object):
    @classmethod
    def iddir(cls):
        return os.path.dirname(os.path.expandvars("$IDPATH"))

    @classmethod
    def idfold(cls):
        return os.path.dirname(os.path.dirname(os.path.expandvars("$IDPATH")))

    @classmethod
    def standardpath(cls, name="g4_00.dae"):
        path_0 = os.path.join(cls.iddir(), name)
        return path_0

    @classmethod
    def path(cls, fold="dpib", name="cfg4.dae"):
        path_1 = os.path.join(cls.idfold(), fold, name)
        return path_1

    @classmethod
    def deref(cls, id_):
        ox = id_.find('0x')
        return id_[0:ox] if ox > -1 else id_

    def __init__(self, path):
        self.x = xmlparse_(path)

    def elem_(self, elem, id_):
        q = ".//%s[@id='%s']" % ((tag_(elem)), id_)
        e = self.x.find(q)
        return e

    def elems_(self, elem):
        q = ".//%s" % tag_(elem)
        es = self.x.findall(q)
        return es

    def float_array(self, id_):
        e = self.elem_("float_array", id_ )
        s = StringIO(e.text)
        a = np.loadtxt(s) 
        return a

    def material(self, id_):
        e = self.elem_("material", id_ )
        return e


if __name__ == '__main__':
    pass
    logging.basicConfig(level=logging.INFO)

    








