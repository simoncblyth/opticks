#!/usr/bin/env python

import os, logging 
log = logging.getLogger(__name__)

from opticks.ana.prim import Dir

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    os.environ["IDPATH"] = "/home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/528f4cefdac670fffe846377973af10a/1"

    d = Dir(os.path.expandvars("$IDPATH/GParts/0"))

    print d     



