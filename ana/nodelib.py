#!/usr/bin/env python
"""
Check analytic and triangulates LVNames, PVNames lists 

"""

import os, logging, numpy as np
from opticks.ana.base import ItemList, translate_xml_identifier_

log = logging.getLogger(__name__)


class VolumeNames(object):
    def compare(self, other):
        assert len(self.pv.names) == len(other.pv.names)
        assert len(self.lv.names) == len(other.lv.names)
        log.info("comparing %s pv names and %s lv names " % (len(self.pv.names), len(self.lv.names)))

        assert self.pv.names == other.pv.names
        assert self.lv.names == other.lv.names

    def __init__(self, prefix="", offset=0, translate_=None):

        lvn = "%sLVNames" % (prefix)
        pvn = "%sPVNames" % (prefix)

        self.lv = ItemList(txt=lvn, offset=offset, translate_=translate_)
        self.pv = ItemList(txt=pvn, offset=offset, translate_=translate_ )




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    tri = VolumeNames(prefix="", offset=0, translate_=translate_xml_identifier_ )
    ana = VolumeNames(prefix="GNodeLib_", offset=0, translate_=None)
    tri.compare(ana)










