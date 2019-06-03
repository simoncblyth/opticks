#!/usr/bin/env python
"""
key.py
===============

See also ana/geocache.bash 

Analogous to bpath.py in old route 


Notes
-------

* some old IDPATH comparison code squatting in this namespace moved to cfgeocache

"""
import os, logging, numpy as np, argparse
log = logging.getLogger(__name__)




class Key(object):
    @classmethod
    def Keydir(cls, key):
        """
        Should match bash geocache-keydir
        """
        elem = key.split(".")
        assert len(elem) == 4, elem 
        exe,cls,top,dig = elem 
        assert len(dig) == 32, "OPTICKS_KEY digest %s is expected to be length 32, not %d " % (dig, len(dig))

        tmpl = "$LOCAL_BASE/opticks/geocache/{exe}_{top}_g4live/g4ok_gltf/{dig}/1".format(**locals())
        keydir = os.path.expandvars(tmpl)
        return keydir

    def __init__(self, key=os.environ["OPTICKS_KEY"]):
        keydir = Key.Keydir(key) 
        exists = os.path.isdir(keydir)

        self.key = key
        self.keydir = keydir
        self.exists = exists
        self.digest = key.split(".")[-1]

        assert exists, "keydir does not exist %s " % str(self)
    

    def __str__(self):
        return "\n".join([self.key, self.keydir])


if __name__ == '__main__':
    pass
    logging.basicConfig(level=logging.INFO)
   
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument( "-v", "--verbose", action="store_true", help="Report key and keydir" )
    args = parser.parse_args()

    key = Key()
    if args.verbose:
        print(key)
    else:
        print(key.keydir)
    pass
