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






