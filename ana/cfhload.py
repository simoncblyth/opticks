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
"""

import os, logging, numpy as np
from opticks.ana.base import opticks_main
from opticks.ana.cfh import CFH
log = logging.getLogger(__name__)

if __name__ == '__main__':
    ok = opticks_main(tag="1", src="torch", det="concentric")

    ctx = {'det':ok.det, 'tag':ok.tag }

    tagd = CFH.tagdir_(ctx)

    seq0s = os.listdir(tagd)

    seq0 = seq0s[0] 
  
    irec = len(seq0.split("_")) - 1

    ctx.update({'seq0':seq0, 'irec':str(irec) })

    log.info(" ctx %r " % ctx )

    for q in ok.qwn.replace(",",""):

        ctx.update({'qwn':q })

        h = CFH.load_(ctx)

        print h



