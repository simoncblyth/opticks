#!/usr/bin/env python
"""
gdml2gltf.py
================

grep selected\":\ 1 $TMP/tgltf/tgltf-gdml--.pretty.gltf | wc -l
    9068

Invoke this with proper environment setup via::

   op --gdml2gltf 


"""
import os, logging, sys, numpy as np

log = logging.getLogger(__name__)

## want to exercise python tree balancing without disturbing other things
os.environ["OPTICKS_GLTFPATH"] = os.path.expandvars("$TMP/gdml2gltf/g4_00.gltf")

from opticks.ana.base import opticks_main
from opticks.analytic.sc import gdml2gltf_main


if __name__ == '__main__':


    args = opticks_main()

    sc = gdml2gltf_main( args )


    



