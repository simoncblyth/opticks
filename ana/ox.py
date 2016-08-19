#!/usr/bin/env python
"""
ox.py: Load final photons
===========================

::

    ox.py --det PmtInBox --tag 10 --src torch 
    ox.py --det dayabay  --tag 1  --src torch 

Jump into interactive::

    ipython -i $(which ox.py) -- --det PmtInBox --tag 10 --src torch 

"""
import logging, sys
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.nload import A

if __name__ == '__main__':
     args = opticks_main(src="torch", tag="10", det="PmtInBox")

     try:
         ox = A.load_("ox",args.src,args.tag,args.det)
     except IOError as err:
         log.fatal(err) 
         sys.exit(args.mrc)

     log.info("loaded ox %s %s shape %s " %  (ox.path, ox.stamp, repr(ox.shape)))

