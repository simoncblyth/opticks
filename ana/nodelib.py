#!/usr/bin/env python
"""
Check analytic and triangulates LVNames, PVNames lists 

"""

import os, logging, numpy as np
from opticks.ana.base import ItemList, translate_xml_identifier_, manual_mixin

log = logging.getLogger(__name__)


class MyItemList(object):
    def find_index(self, name):
        return self.names.index(name)

manual_mixin(ItemList, MyItemList)


class VolumeNames(object):
    def __init__(self, prefix="", offset=0, translate_=None):

        lvn = "%sLVNames" % (prefix)
        pvn = "%sPVNames" % (prefix)

        self.lv = ItemList(txt=lvn, offset=offset, translate_=translate_)
        self.pv = ItemList(txt=pvn, offset=offset, translate_=translate_ )
        assert len(self.lv.names) == len(self.pv.names)
        self.size = len(self.lv.names)
        self.num_unique_pv = len(set(self.pv.names))


class CfVolumeNames(object):
    """
    PV names are not unique, from instancing

        In [17]: len(cfv.tri.pv.names)
        Out[17]: 12230

        In [18]: len(set(cfv.tri.pv.names))
        Out[18]: 5643

    """
    def __init__(self):
        tri = VolumeNames(prefix="", offset=0, translate_=translate_xml_identifier_ )
        ana = VolumeNames(prefix="GNodeLib_", offset=0, translate_=None)
    
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
            # Need some top level metadata in the GLTF to identify the starting (full geometry)
            # node index to be specific.

            pv_f = ana.pv.names[0]
            pv_l = ana.pv.names[-1]
            idx_f = tri.pv.find_index(pv_f) 
            idx_l = tri.pv.find_index(pv_l) 
            log.info("tri indices of first and last ana PV : pv_f %s idx_f %d      pv_l %s idx_l %d  " % (pv_f,idx_f, pv_l, idx_l ))

            assert tri.pv.names[idx_f:idx_f+len(ana.pv.names)] == ana.pv.names
            assert tri.lv.names[idx_f:idx_f+len(ana.lv.names)] == ana.lv.names
        pass

        self.tri = tri
        self.ana = ana
  






if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    cfv = CfVolumeNames()



    








