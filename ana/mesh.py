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

tx_load = lambda _:map(str.strip, open(_).readlines())
js_load = lambda _:json.load(file(_))

class Mesh(object):
    def path(self, rel):
        return os.path.join(self.idpath, rel)

    @classmethod
    def make(cls, path=None):
        if path is None:
            path = os.path.abspath(os.curdir)
        pass
        return cls(cls.find_idpath(path))

    @classmethod
    def find_idpath(cls, path):
        """
        Convert any absolute path inside the idpath into the idpath 
        """
        elem = path.split("/")
        elen = map(len,elem)

        try:
            digp = elen.index(32)   # digest has length of 32 
            idpath = "/".join(elem[:digp+2])  # one past the digest 
        except ValueError:
            idpath = os.environ["IDPATH"]
            log.warning("using IDPATH from environment")
        pass
        return idpath

    def __init__(self, idpath):
        log.info("Mesh for idpath : %s " % idpath )
        self.idpath = idpath
        self.map = self.path("MeshIndex/GItemIndexSource.json")
        mesh = js_load(self.map)
        log.info("loading map %s kv pairs %d " % (self.map, len(mesh.values())))
        self.name2idx = mesh
        self.idx2name = dict(zip(map(int,mesh.values()), mesh.keys()))




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    mh = Mesh.make()
    args = sys.argv[1:]

    iargs = map(int, args) if len(args) > 0 else mh.idx2name.keys()
    for idx in iargs:
        print "%3d : %s " % ( idx, mh.idx2name[idx] )
    pass

     

 
