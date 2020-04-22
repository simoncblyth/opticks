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
GPmt.py
========

* split off analytic PMT saving and loading into this GPmt class
* 
* this is loaded C++ side with ggeo/GPmt


"""

import os, logging
log = logging.getLogger(__name__)
import numpy as np
from opticks.ana.base import opticks_main, splitlines_
from opticks.ana.nbase import Buf
from gcsg import GCSG

class GPmt(object):
    atts = "boundaries materials lvnames pvnames".split() 

    def __init__(self, path, buf=None):
        """
        :param path: eg "$TMP/GPmt/0/GPmt.npy" "$IDPATH/GPmt/0/GPmt.npy"
        :param buf: part buffer
        """
        self.path = path
        self.xpath = os.path.expandvars(path)
        self.pdir = os.path.dirname(self.xpath)
        self.buf = buf 

    def txtpath(self, att):
        return self.xpath.replace(".npy","_%s.txt" % att)

    def gcsgpath(self):
        return self.xpath.replace(".npy","_csg.npy")


    def save(self):
        log.info("GPmt.save buf %s to:%s attribs and gcsg in sidecars " % (repr(self.buf.shape), self.path) )
        if not os.path.exists(self.pdir):
            os.makedirs(self.pdir)
        pass
        for att in self.atts:
            if hasattr(self.buf, att):
                path = self.txtpath(att)
                log.info("saving %s to %s " % (att, path))
                with open(path,"w") as fp:
                    fp.write("\n".join(getattr(self.buf,att))) 
                pass
            pass
        pass
        if hasattr(self.buf,"gcsg"):
            path = self.gcsgpath()
            log.info("serializing gcsg to %s " % path) 
            gcsgbuf = GCSG.serialize_list(self.buf.gcsg)
            if gcsgbuf is not None:
                log.info("saving gcsg to %s " % path)
                #log.info(gcsgbuf.view(np.int32))
                #log.info(gcsgbuf)
                np.save(path, gcsgbuf) 
            else:
                log.warning("gcsgbuf is None skip saving to %s " % path)
            pass
        pass
        np.save(self.xpath, self.buf) 

    def load(self):
        if not os.path.exists(self.pdir):
            log.fatal("no such path %s " % self.pdir)
            assert 0 
 
        a = np.load(self.xpath)
        self.buf = a.view(Buf)

        for att in self.atts:
            path = self.txtpath(att)
            if os.path.exists(path):
                setattr(self.buf, att, splitlines_(path))
            else:
                log.warning("no such path %s " % path)
              
        path = self.gcsgpath()
        if os.path.exists(path):
            self.gcsgbuf = np.load(path)
        else:
            log.warning("no such path %s " % path)
            
 
    def dump(self):
        print "buf %s " % repr(self.buf.shape)
        print "gcsgbuf %s " % repr(self.gcsgbuf.shape)
        for att in self.atts:
            tls = getattr(self.buf, att)
            print att, len(tls), " ".join(tls)



if __name__ == '__main__':
    args = opticks_main(apmtpath="$TMP/GPmt/0/GPmt.npy")


    p = GPmt(args.apmtpath)
    p.load() 
    p.dump()






