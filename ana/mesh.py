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

     

 
