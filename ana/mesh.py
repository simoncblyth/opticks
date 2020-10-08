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


import sys, os, json, numpy as np, logging
log = logging.getLogger(__name__)

tx_load = lambda _:list(map(str.strip, open(_).readlines()))
js_load = lambda _:json.load(open(_))

class Mesh(object):
    def __init__(self, kd):
        log.info("Mesh for kd : %s " % kd )
        lvnames = tx_load(os.path.join(kd, "GItemList/GMeshLib.txt"))
        name2idx = dict(zip( lvnames, range(len(lvnames)) ))
        idx2name = dict(zip( range(len(lvnames)), lvnames ))   
        self.keydir = kd
        self.name2idx = name2idx
        self.idx2name = idx2name


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    from opticks.ana.key import keydir
    kd = keydir(os.environ["OPTICKS_KEY"]) 

    mh = Mesh(kd)
    args = sys.argv[1:]

    iargs = map(int, args) if len(args) > 0 else mh.idx2name.keys()
    for idx in iargs:
        print("%3d : %s " % ( idx, mh.idx2name[idx] ))
    pass

     

 
