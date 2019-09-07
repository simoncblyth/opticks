#!/usr/bin/env python
#-*- coding: utf-8 -*-
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

import os, logging

log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.analytic.treebase import Tree
from opticks.analytic.gdml import GDML

def gdml2idmap(tree, pv_sd_name, pv_sd_pmtid):
    name_cache = set()
    for k,v in tree.byindex.iteritems():
        #print k, v.pv.name
        name = v.pv.name.split('0x')[0]

        is_sd = name in pv_sd_name 
        pmtid = 0

        if is_sd:
            pmtid = pv_sd_pmtid[name]
            pv_sd_pmtid[name] += 1

        print "%d %d %d"%(v.index, pmtid, is_sd)

        name_cache.add(name)
    log.info(name_cache)
    log.info(pv_sd_pmtid)
    # set(['pAcylic', 'PMT_3inch_log_phys', 'PMT_3inch_cntr_phys', 'PMT_3inch_body_phys', 'lUpperChimney_phys', 'pCentralDetector', 'pExpHall', 'pPoolLining', 'pLowerChimneySteel', 'pUpperChimneyTyvek', 'pMask', 'lSurftube_phys', 'lFasteners_phys', 'lMaskVirtual_phys', 'top', 'PMT_3inch_inner2_phys', 'pvacSurftube', 'PMT_20inch_inner2_phys', 'pOuterWaterPool', 'pInnerWater', 'PMT_3inch_inner1_phys', 'pBtmRock', 'PMT_20inch_body_phys', 'lLowerChimney_phys', 'pTarget', 'lSteel_phys', 'PMT_20inch_log_phys', 'pUpperChimneySteel', 'pLowerChimneyAcrylic', 'pUpperChimneyLS', 'pLowerChimneyTyvek', 'PMT_20inch_inner1_phys', 'pLowerChimneyBlocker', 'pLowerChimneyLS', 'pTopRock'])


if __name__ == "__main__":
    args = opticks_main()
    log.info(args)
    gdmlpath = os.environ['OPTICKS_GDMLPATH']
    log.info(gdmlpath)
    log.info("start GDML parse")
    gdml = GDML.parse(gdmlpath)

    log.info("start treeify")
    tree = Tree(gdml.world)  
    # print tree
    # print dir(tree)
    # print tree.root


    # FOR JUNO ONLY
    pv_sd_name = None
    pv_sd_pmtid = None

    if args.j1707:
        pv_sd_name = ['PMT_20inch_inner1_phys', 'PMT_3inch_inner1_phys']
        pv_sd_pmtid = {'PMT_20inch_inner1_phys': 0, 'PMT_3inch_inner1_phys': 300000}

    gdml2idmap(tree, pv_sd_name, pv_sd_pmtid)
