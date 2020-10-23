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
import os, json, logging, numpy as np, argparse
log = logging.getLogger(__name__)

def keydir(keyspec=None):
    if keyspec is None:
        keyspec = os.environ["OPTICKS_KEY"] 
    pass
    return Key.Keydir(keyspec)

def key_(keyspec=None):
    if keyspec is None:
        keyspec = os.environ["OPTICKS_KEY"] 
    pass
    return Key(keyspec)



class Key(object):
    @classmethod
    def GeoCachePrefix(cls):
        geocache_prefix_default = os.path.expanduser("~/.opticks") 
        geocache_prefix = os.environ.get("OPTICKS_GEOCACHE_PREFIX", geocache_prefix_default ) 
        return geocache_prefix 

    @classmethod
    def GeoCacheDir(cls):
        return "%s/geocache" % cls.GeoCachePrefix()

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
        geocache_dir = cls.GeoCacheDir()
        tmpl = "{geocache_dir}/{exe}_{top}_g4live/g4ok_gltf/{dig}/1".format(**locals())
        keydir = os.path.expandvars(tmpl)
        os.environ["KEYDIR"] = keydir 
        if not "TMP" in os.environ:
            os.environ["TMP"] = os.path.expandvars("/tmp/$USER/opticks")
        pass
        return keydir

    def __init__(self, keyspec=None):
        if keyspec is None:
            keyspec = os.environ.get("OPTICKS_KEY",None)
        pass
        keydir = Key.Keydir(keyspec) 
        exists = os.path.isdir(keydir)
        meta = json.load(open(os.path.join(keydir, "cachemeta.json")))

        self.keyspec = keyspec
        self.keydir = keydir
        self.exists = exists
        self.digest = keyspec.split(".")[-1]
        self.meta = meta 
        self.version = int(meta["GEOCACHE_CODE_VERSION"])
        self.gdmlpath = self.extract_argument_after(meta, "--gdmlpath")
  
    @classmethod
    def extract_argument_after(cls, meta, k):
        argline = meta.get("argline","-")
        args = argline.split(" ")
        try:
            ppos = args.index(k)
        except ValueError:
            ppos = -1 
        pass
        log.info("ppos %d" % ppos)
        path = None
        if ppos == -1:
            pass
        elif ppos + 1 >= len(args):
            log.fatal("truncated argline ?")
        else:
            arg = args[ppos+1] 
        pass
        return arg

 
    def __repr__(self):
        version = self.version
        keyspec = self.keyspec 
        keydir = self.keydir
        return "\n".join(["Key.v{version}:{keyspec}","{keydir}"]).format(**locals())


    def __str__(self):
        return "\n".join([self.keyspec, self.keydir, self.gdmlpath])


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
