#!/usr/bin/env python

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
        digp = elen.index(32)   # digest has length of 32 
        idpath = "/".join(elem[:digp+2])  # one past the digest 
        return idpath

    def __init__(self, idpath):
        log.info("Mesh for idpath : %s " % idpath )
        self.idpath = idpath
        mesh = js_load(self.path("MeshIndex/GItemIndexSource.json"))
        self.name2idx = mesh
        self.idx2name = dict(zip(map(int,mesh.values()), mesh.keys()))




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    mh = Mesh.make()
    args = sys.argv[1:]
    for idx in map(int,args):
        print mh.idx2name[idx]
    pass

     

 
