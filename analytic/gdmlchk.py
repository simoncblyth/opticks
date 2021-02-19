#!/usr/bin/env python
import os, sys, logging
log = logging.getLogger(__name__)

import numpy as np
import lxml.etree as ET
import lxml.html as HT

from collections import OrderedDict as odict 

tostring_ = lambda _:ET.tostring(_)
exists_ = lambda _:os.path.exists(os.path.expandvars(_))
parse_ = lambda _:ET.parse(os.path.expandvars(_)).getroot()
fparse_ = lambda _:HT.fragments_fromstring(file(os.path.expandvars(_)).read())
pp_ = lambda d:"\n".join([" %30s : %f " % (k,d[k]) for k in sorted(d.keys())])

values_ = lambda _:np.fromstring(_, sep=" ")

if __name__ == '__main__':

   #path = sys.argv[1]
   path = "/usr/local/opticks/opticksaux/export/juno2102/tds_ngt_pcnk_sycg_202102_v0.gdml"
   g = parse_(path) 


   mm = odict()
   vv = []

   es = None 
   ev = None 

   for m in g.xpath("//matrix"):
       coldim = m.attrib["coldim"]
       assert coldim == "2"
       name = m.attrib["name"]
       v = values_(m.attrib["values"])
       if len(v) % 2 == 0:
           v = v.reshape(-1,2)
           msg = ""
       else:
           msg = "UNEXPECTED MATRIX SHAPE"
           es = m.attrib["values"]
           ev = v
       pass
       print("%40s : %s : %10s : %s " % (name, coldim, str(v.shape), msg)) 
       mm[name] = v
       vv.append(v)
   pass



