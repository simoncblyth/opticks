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
Check analytic and triangulates LVNames, PVNames lists 

::

    simon:ana blyth$ ./nodelib.py 
    INFO:opticks.ana.base:translating
    INFO:opticks.ana.base:translating
    INFO:__main__:lv ItemLists names  12230 name2code    249 code2name  12230 offset     0 npath $IDPATH/GItemList/LVNames.txt  
    INFO:__main__:pv ItemLists names  12230 name2code   5643 code2name  12230 offset     0 npath $IDPATH/GItemList/PVNames.txt  
    INFO:__main__:lv ItemLists names   1660 name2code    171 code2name   1660 offset     0 npath $IDPATH/analytic/GScene/GNodeLib/LVNames.txt  
    INFO:__main__:pv ItemLists names   1660 name2code    704 code2name   1660 offset     0 npath $IDPATH/analytic/GScene/GNodeLib/PVNames.txt  
    INFO:__main__:ana is partial list tri.size 1660 ana.size 12230 
    INFO:__main__:tri indices of first ana PV : pv_f /dd/Geometry/Pool/lvNearPoolIWS#pvNearADE10xc2cf528 idx_f 3153 
    INFO:__main__:tri indices of last  ana PV : pv_l /dd/Geometry/AdDetails/lvMOOverflowTankE#pvMOFTTopCover0xc20cf40 idx_l 4804 
    simon:ana blyth$ 


"""

import os, logging, numpy as np
from opticks.ana.base import ItemList, translate_xml_identifier_, manual_mixin

log = logging.getLogger(__name__)


#class MyItemList(object):
#    def find_index(self, name):
#        return self.names.index(name)
#
#manual_mixin(ItemList, MyItemList)


class VolumeNames(object):
    def __init__(self, reldir=None, offset=0, translate_=None):
        lv = ItemList(txt="LVNames", offset=offset, translate_=translate_ , reldir=reldir)
        pv = ItemList(txt="PVNames", offset=offset, translate_=translate_ , reldir=reldir)
        log.info( "lv %r " % lv )
        log.info( "pv %r " % pv )

        assert len(lv.names) == len(pv.names)

        self.size = len(lv.names)
        self.num_unique_pv = len(set(pv.names))
        self.lv = lv
        self.pv = pv
   


class CfVolumeNames(object):
    """
    PV names are not unique, from instancing

        In [17]: len(cfv.tri.pv.names)
        Out[17]: 12230

        In [18]: len(set(cfv.tri.pv.names))
        Out[18]: 5643

    """
    def __init__(self):
        tri = VolumeNames(reldir=None, offset=0, translate_=translate_xml_identifier_ )
        ana = VolumeNames(reldir="analytic/GScene/GNodeLib", offset=0, translate_=None)
    
        if tri.size == ana.size:
            log.info("same size %s " % tri.size)
            assert tri.pv.names == ana.pv.names
            assert tri.lv.names == ana.lv.names
        else:
            assert ana.size < tri.size, (ana.size, tri.size)
            log.info("ana is partial list tri.size %s ana.size %s " % (ana.size, tri.size))

            # PV names are not unique for instances, so the below reconstruction of the offset index 
            # from the ana pv names will only work for cases where the first found name corresponds to 
            # the correct volume indes.
            #
            # This is handled C++ side with gltftarget (config) and NScene targetnode GLTF asset metadata

            pv_f = ana.pv.names[0]
            pv_l = ana.pv.names[-1]
            idx_f = tri.pv.find_index(pv_f) 
            idx_l = tri.pv.find_index(pv_l) 
            log.info("tri indices of first ana PV : pv_f %s idx_f %d " % (pv_f,idx_f))
            log.info("tri indices of last  ana PV : pv_l %s idx_l %d " % (pv_l, idx_l))

            assert tri.pv.names[idx_f:idx_f+len(ana.pv.names)] == ana.pv.names
            assert tri.lv.names[idx_f:idx_f+len(ana.lv.names)] == ana.lv.names
        pass

        self.tri = tri
        self.ana = ana
  






if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    cfv = CfVolumeNames()



    








