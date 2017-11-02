#!/usr/bin/env python
"""
so.py: Load CPU emitsource "input photons"
================================================

::

    so.py --det PmtInBox --tag 10 --src torch 
    so.py --det dayabay  --tag 1  --src torch 
    so.py --det tboolean-torus  --tag 1  --src torch 

Jump into interactive::

    ipython -i $(which so.py) -- --det tboolean-sphere --tag 1 --src torch 

"""
import logging, sys
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.nload import A
from opticks.ana.nbase import vnorm


if __name__ == '__main__':
     args = opticks_main(src="torch", tag="1", det="tboolean-sphere")

     try:
         so = A.load_("so",args.src,args.tag,args.det)
     except IOError as err:
         log.fatal(err) 
         sys.exit(args.mrc)

     log.info("loaded so %s %s shape %s " %  (so.path, so.stamp, repr(so.shape)))

     print so

     v = so[:,0,:3]

     print "v",v
     print "vnorm(v)", vnorm(v)


