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
ox.py: Load final photons
===========================

::

    ox.py --det PmtInBox --tag 10 --src torch 
    ox.py --det dayabay  --tag 1  --src torch 
    ox.py --det tboolean-torus  --tag 1  --src torch 

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

     print ox

