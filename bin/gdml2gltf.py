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
#os.environ["OPTICKS_GLTFPATH"] = os.path.expandvars("$TMP/gdml2gltf/g4_00.gltf")

from opticks.ana.base import opticks_main
from opticks.analytic.sc import gdml2gltf_main


if __name__ == '__main__':


    args = opticks_main()

    sc = gdml2gltf_main( args )


    



