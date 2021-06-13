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


import sys, os, numpy as np, logging, argparse
log = logging.getLogger(__name__)
tx_load = lambda _:list(map(str.strip, open(_).readlines())) # py3 needs the list, otherwise stays as map 

from opticks.ana.key import keydir
KEYDIR = keydir()

class BLib(object):
    @classmethod
    def parse_args(cls, doc, **kwa):
        np.set_printoptions(suppress=True, precision=3 )
        parser = argparse.ArgumentParser(doc)
        #parser.add_argument(     "path",  nargs="?", help="Geocache directory", default=kwa.get("path",None) )
        parser.add_argument(     "--level", default="info", help="logging level" ) 
        parser.add_argument(     "-b","--brief", action="store_true", default=False ) 
        parser.add_argument(     "-n","--names", action="store_true", default=False ) 
        parser.add_argument(     "-s","--selection", default="", help="comma delimited list of selected boundary indices" ) 
        args = parser.parse_args()
        fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
        logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)
        return args  

    @classmethod
    def old_make(cls, path):
        return cls(cls.find_idpath(path))

    @classmethod
    def old_find_idpath(cls, path):
        """
        Heuristically convert any absolute path inside the idpath into the idpath 
        by looking for path element of length 32 corresponding to the digest string.
        """
        elem = path.split("/")
        elen = list(map(len,elem))
        try: 
            digp = elen.index(32)   # digest has length of 32 
            idpath = "/".join(elem[:digp+2])  # one past the digest 
        except ValueError:
            log.warning("failed to find_idpath from directory : fallback to envvar")
            idpath = os.environ.get("IDPATH", None)
        pass
        return idpath


    def path(self, rel):
        return os.path.join(self.keydir, rel)

    def __init__(self, kd=KEYDIR):
        """
        Load boundary lib index and the GItemList text files with material and surface names
        """
        self.keydir = kd
        blib = np.load(self.path("GBndLib/GBndLibIndex.npy"))
        mlib = tx_load(self.path("GItemList/GMaterialLib.txt"))
        slib = tx_load(self.path("GItemList/GSurfaceLib.txt"))
        self.blib = blib
        self.mlib = mlib
        self.slib = slib
        self._selection = range(len(self.blib)) 
    def mname(self, idx):
        return self.mlib[idx] if idx < len(self.mlib) else ""
    def sname(self, idx):
        return self.slib[idx] if idx < len(self.slib) else ""
    def bname(self, idx):
        assert idx > -1, idx 
        omat, osur, isur, imat = self.blib[idx] 
        return "/".join(
                    [ self.mname(omat), 
                      self.sname(osur), 
                      self.sname(isur), 
                      self.mname(imat) ] )


    def __repr__(self):
        return " nbnd %3d nmat %3d nsur %3d " % ( len(self.blib), len(self.mlib), len(self.slib))

    def _set_selection(self, q):
        self._selection = list(map(int, q.split(",")))
    def _get_selection(self):
        return self._selection
    selection = property(_get_selection, _set_selection)

    def __str__(self):
        return "\n".join([repr(self)] +  list(map(lambda _:"%3d : %3d : %s " % ( _, _+1, self.bname(_)) , self.selection)))
    def brief(self):
        rng = range(len(self.blib))
        rng = rng[0:5] + rng[-5:]
        return "\n".join([repr(self)] +  list(map(lambda _:"%3d : %s " % ( _, self.bname(_)) , rng )))

    def names(self):
        return "\n".join(map(lambda _:self.bname(_) , self.selection ))

    def format(self, bn):
        """
        :param bn: array of 1-based boundary indices, as obtained from eg a.bn[0] 
        """
        return "\n".join(["%3d : %s" % (b, self.bname(abs(b)-1)) for b in list(filter(lambda b:b != 0, bn))])



if __name__ == '__main__':

    args = BLib.parse_args(__doc__)

    blib = BLib()

    if args.selection:
        blib.selection = args.selection 
    pass

    if args.brief: 
        print(blib.brief())
    elif args.names:
        print(blib.names())
    else:
        print(blib)
    pass


     
