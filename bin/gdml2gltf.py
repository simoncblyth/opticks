#!/usr/bin/env python
"""
gdml2gltf.py
================

TODO:

* gltf should get standard place inside the geocache

grep selected\":\ 1 $TMP/tgltf/tgltf-gdml--.pretty.gltf | wc -l
    9068


"""
import os, logging, sys, numpy as np

log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.analytic.sc import gdml2gltf_main


if __name__ == '__main__':


    args = opticks_main()

    sc = gdml2gltf_main( args )


    



