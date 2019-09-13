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
key.py
===============

See also::

   ana/geocache.bash 
   boostrap/BOpticksResource.cc

NB path layout changes need to be done in triplicate in bash/py/C++ 

Analogous to bpath.py in old route 


Notes
-------

* some old IDPATH comparison code squatting in this namespace moved to cfgeocache

"""
import os, logging, numpy as np, argparse
log = logging.getLogger(__name__)




class Key(object):
    @classmethod
    def CachePrefix(cls):
        cache_prefix_default = os.path.expanduser("~/.opticks") 
        cache_prefix = os.environ.get("OPTICKS_CACHE_PREFIX", cache_prefix_default ) 
        return cache_prefix 

    @classmethod
    def Keydir(cls, key):
        """
        Should match bash geocache-keydir
        """
        if key is None: return None
        elem = key.split(".")
        assert len(elem) == 4, elem 
        exe,kls,top,dig = elem 
        assert len(dig) == 32, "OPTICKS_KEY digest %s is expected to be length 32, not %d " % (dig, len(dig))
        cache_prefix = cls.CachePrefix()
        tmpl = "{cache_prefix}/geocache/{exe}_{top}_g4live/g4ok_gltf/{dig}/1".format(**locals())
        keydir = os.path.expandvars(tmpl)
        return keydir

    def __init__(self, key=os.environ.get("OPTICKS_KEY",None)):
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
