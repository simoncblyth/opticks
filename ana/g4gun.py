#!/usr/bin/env python
"""
g4gun.py: loads G4Gun event
===============================

To create the event use::

   ggv-
   ggv-g4gun


"""
import os, logging, numpy as np
log = logging.getLogger(__name__)


from opticks.ana.base import opticks_environment
from opticks.ana.evt import Evt

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    opticks_environment()

    evt = Evt(tag="-1", src="G4Gun", det="G4Gun")
    print evt




   

