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
catdir.py
===========

"""

from datetime import datetime
import os, re, logging
import numpy as np
from collections import OrderedDict as odict
log = logging.getLogger(__name__)

from opticks.ana.base import ini_, json_, splitlines_
from opticks.ana.datedfolder import DatedFolder, dateparser
from opticks.ana.metadata import Metadata


class Catdir(object):
    """
    Reads in metadata from dated folders corresponding to runs of::

          ggv 
          ggv --compute
          ggv --cfg4

    ::

        INFO:__main__:Catdir searching for date stamped folders beneath : /usr/local/env/opticks/rainbow 
        INFO:__main__:/usr/local/env/opticks/rainbow/mdtorch/-5/20160104_141234
        INFO:__main__:/usr/local/env/opticks/rainbow/mdtorch/-5/20160104_144620
        INFO:__main__:/usr/local/env/opticks/rainbow/mdtorch/-5/20160104_150114
        INFO:__main__:/usr/local/env/opticks/rainbow/mdtorch/-5/20160105_155311
        INFO:__main__:/usr/local/env/opticks/rainbow/mdtorch/-5/20160107_201949
        INFO:__main__:/usr/local/env/opticks/rainbow/mdtorch/-5/20160224_120049
        INFO:__main__:/usr/local/env/opticks/rainbow/mdtorch/-6/20160105_172431
        INFO:__main__:/usr/local/env/opticks/rainbow/mdtorch/-6/20160105_172538
        INFO:__main__:/usr/local/env/opticks/rainbow/mdtorch/5/20160103_165049
        INFO:__main__:/usr/local/env/opticks/rainbow/mdtorch/5/20160103_165114
        INFO:__main__:/usr/local/env/opticks/rainbow/mdtorch/5/20160104_111515

    """
    def __init__(self, path):
        """
        Path should be the folder above the tagdir in order
        to allow comparisons between equivalent (ie G4 negated) tags 
        """
        dirs, dfolds, dtimes = DatedFolder.find(path)

        log.info("Catdir searching for date stamped folders beneath : %s " % path)
        print( "\n".join(dirs))

        metamap = odict()
        for p in dirs:
            log.debug("%s", p)
            md = Metadata(p, base=path)
            tag = md.tag
            if tag not in metamap:
                metamap[tag] = []
            metamap[tag].append(md)
        pass 

        self.metamap = metamap
        self.path = path

        self.dump()


    def dump(self):
        log.info("Catdir %s tags: %s beneath %s " % ( len(self.metamap), repr(self.metamap.keys()), self.path))
        for tag, mds in self.metamap.items():
            print "%5s : %d " % (tag, len(mds)) 

    def tags(self): 
        return self.metamap.keys()

    def times(self, tag):
        """
        :param tag: event tag, eg 1,2 (be negated to get cfg4 tags)
        :return: recarray with propagate times and corresponding flags indicating cfg4/interop/compute
        """
        tags = [tag, "-%s" % tag]   # convention, negate to get equivalent cfg4 tag
        mds = []
        for tag in tags:
            mds.extend(self.metamap.get(tag, [])) 
        pass

        log.info("times metadata for tag %s " % tag + "\n".join(map(str,mds)))

        n = len(mds)
        if n == 0:
            log.warning("no metadata found")
            return None  

        numpho0 = mds[0].numPhotons

        ## collect indices with consistent photon counts
        consistent = []
        for i in range(n):
            md = mds[i]
            if md.numPhotons == numpho0:
                consistent.append(i)
            pass
        pass
            
        nc = len(consistent)
        if nc != n:
            log.warning("skipped metadata with inconsistent photon count n %s nc %s " % (n, nc)) 
        pass

        a = np.recarray((nc,), dtype=[("index", np.int32), ("time", "|O8"), ("propagate", np.float32), ("flgs", np.uint32 ), ("numPhotons",np.uint32)])

        j = 0 
        for i in range(n):
            md = mds[i]
            dat = (i, md.timestamp, md.propagate, md.flags, md.numPhotons )
            if i in consistent:
                a[j] = dat
                j += 1 
                print(dat) 
            else:
                log.warning("skipped inconsistent %s " % repr(dat))
            pass
        pass

        return a 

    def __repr__(self):
         return "%s tags %s " % (self.path, repr(self.tags()))





def test_catdir():
    #cat, tag= "rainbow", "6"
    cat, tag= "PmtInBox", "4"
    catd = Catdir(cat)
    a = catd.times(tag)

def test_catdir_find():
    cat = Catdir(os.path.expandvars("/tmp/$USER/opticks/evt/PmtInBox/torch"))
    tags = cat.tags()
    for tag in tags:
        a = cat.times(tag)
        print "a", a



if __name__ == '__main__':
    from opticks.ana.main import opticks_main
    ok = opticks_main() 
    print(ok.brief)
    print("tagdir : %s " % ok.tagdir)
    print("catdir : %s " % ok.catdir)

    cat = Catdir(ok.catdir)
    tags = cat.tags()
    for tag in tags:
        a = cat.times(tag)
        print("a ", a)
    pass




