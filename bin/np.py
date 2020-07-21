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
np.py
========

This is intended as a quick and dirty dumper/comparer of npy files,
for quick initial comparisons of small numbers of arrays.

When wishing to get specific to certain collections of npy files, 
move to specific other scripts such as ab.py etc..  

Keep this general.


NB when intended argument values begin with a negative sign
you have to handhold argparser to get them in::

   np.py GMergedMesh/0/nodeinfo.npy -viF --slice="-20:-1"

Because arguments beginning with negative signs are very common with slices, 
underscores in arguments are converted to "-" signs for the slice arguments, 
allowing::

   np.py GMergedMesh/0/nodeinfo.npy -viF -s _20:_1



Jump into ipython with all geocache arrays loaded with names a,b,c,d,...::

    ip -i -- $(which np.py) GMergedMesh/1/*.npy 

    inp(){ ipython -i $(which np.py) -- $* ; }    # shortcut bash function 

    epsilon:1 blyth$ inp GMergedMesh/1/*.npy 
    a :                                  GMergedMesh/1/iidentity.npy :          (128000, 4) : 925e98ab591dcdde40a42777b8331e9d : 20200702-2350 
    b :                                 GMergedMesh/1/aiidentity.npy :        (25600, 1, 4) : 2656f9e5f92a858ac5c3d931bf4859fe : 20200702-2350 
    c :                                GMergedMesh/1/itransforms.npy :        (25600, 4, 4) : 29a7bf21dabfd4a6f9228fadb7edabca : 20200702-2350 
    d :                                    GMergedMesh/1/indices.npy :            (4752, 1) : b5d5dc7ce94690319fb384b1e503e2f9 : 20200702-2350 
    e :                                 GMergedMesh/1/boundaries.npy :            (1584, 1) : 4583b9e4b2524fc02d90306a4ae93238 : 20200702-2350 
    f :                                      GMergedMesh/1/nodes.npy :            (1584, 1) : 8cb9bf708067a07977010b6bc92bf565 : 20200702-2350 
    g :                                    GMergedMesh/1/sensors.npy :            (1584, 1) : 30e007064ccb81e841e90dde1304ccf2 : 20200702-2350 
    h :                                     GMergedMesh/1/colors.npy :             (805, 3) : 5b2f1391f85c6e29560eed612a0e890a : 20200702-2350 
    i :                                    GMergedMesh/1/normals.npy :             (805, 3) : 5482a46493c73523fdc5356fd6ed5ebc : 20200702-2350 
    j :                                   GMergedMesh/1/vertices.npy :             (805, 3) : b447acf665678da2789103b44874d6bb : 20200702-2350 
    k :                                       GMergedMesh/1/bbox.npy :               (5, 6) : a523db9c1220c034d29d8c0113b4ac10 : 20200702-2350 
    l :                              GMergedMesh/1/center_extent.npy :               (5, 4) : 3417b940f4da6db67abcf29937b52128 : 20200702-2350 
    m :                                   GMergedMesh/1/identity.npy :               (5, 4) : a921a71d379336f28e7c0b908eea9218 : 20200702-2350 
    n :                                     GMergedMesh/1/meshes.npy :               (5, 1) : 0a52a5397e61677ded7cd8a7b23bf090 : 20200702-2350 
    o :                                   GMergedMesh/1/nodeinfo.npy :               (5, 4) : c143e214851e70197a6de58b2c86b5a9 : 20200702-2350 
    p :                                 GMergedMesh/1/transforms.npy :              (5, 16) : 37ae1f7f4da2409596627cebfa5cb28b : 20200702-2350 

    In [1]: p.shape
    Out[1]: (5, 16)

    In [2]: 1584*3
    Out[2]: 4752



::

   inp tmp/blyth/OKG4Test/evt/g4live/natural/-1/so.npy source/evt/g4live/natural/-1/so.npy -v

Comparing directories containing npy::

   np.py tmp/blyth/OKG4Test/evt/g4live/natural/{-1,1} 

Even more minimal::

   python -c "import numpy as np, sys ; np.set_printoptions(suppress=True) ; print np.load(sys.argv[1]) " 



"""
import sys, fnmatch, os, logging, numpy as np, commands, argparse
from collections import OrderedDict as odict

log = logging.getLogger(__name__)

is_npy_ = lambda _:fnmatch.fnmatch(_,"*.npy")
is_npj_ = lambda _:fnmatch.fnmatch(_,"*.npj")  # npj is convention for json files containing array metadata
is_txt_ = lambda _:fnmatch.fnmatch(_,"*.txt")
is_dir_ = lambda _:os.path.isdir(_)

from opticks.bin.md5 import digest_
from opticks.ana.base import stamp_

def dump_one(a, args):
    print(a.shape)
    if args.float:
        print("f32")
        print(a[args.slice].view(np.float32))
    pass
    if args.int:
        print("i32")
        print(a[args.slice].view(np.int32))
    pass

def dump_tree(base, args):
    """
    Recursively lists shapes of all .npy files 
    and line lengths of .txt files.
    """
    print(os.path.abspath(base))
    for root, dirs, files in os.walk(base):
        sfiles = sorted(files)
        if args.txt:
            for name in filter(is_txt_,sfiles):
                path = os.path.join(root, name)
                txt_brief(path)
            pass
        pass
        for name in filter(is_npy_,sfiles):
            path = os.path.join(root, name)
            npy_brief(path, args)  
        pass   
    pass


def txt_brief( path, label="." ):
    fdig = digest_(path)
    stmp = stamp_(path)
    lines = len(file(path, "r").readlines())
    print("%s : %60s : %20s : %s : %s " % ( label, path, lines , fdig, stmp ))

def npy_brief( path, args, label="."):
    a = np.load(path)
    fdig = digest_(path)
    stmp = stamp_(path)
    print("%s : %60s : %20s : %s : %s " % ( label, path, repr(a.shape), fdig, stmp ))
    if args.verbose > 0:
        dump_one(a, args)
    pass
    return a

class A(object):
    def __init__(self, path):
        self.path = path 
        self.a = np.load(path)
        self.fdig = digest_(path)
        self.stmp = stamp_(path)

    def __repr__(self):
        return "%60s : %20s : %s : %s " % (self.path, repr(self.a.shape), self.fdig, self.stmp)

    def fdump(self, sli):
        print(self.a.shape)
        print("f32")
        print(self.a[sli].view(np.float32))

    def idump(self, sli):
        print(self.a.shape)
        print("i32")
        print(self.a[sli].view(np.int32))


def compare(a, b):
    df = a - b 
    print( " max(a-b) %10.3g  min(a-b) %10.3g " % (np.max(df), np.min(df)) ) 


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(  "paths", nargs='*', help="File or directory paths with .npy" )
    parser.add_argument("-v","--verbose", action="store_true", default=False )
    parser.add_argument("-f","--float", action="store_true", default=True )
    parser.add_argument("-F","--nofloat", dest="float", action="store_false" )
    parser.add_argument("-T","--notxt", dest="txt", action="store_false", default=True )
    parser.add_argument("-i","--int", action="store_true", default=False )
    parser.add_argument("-n","--threshold", type=int, default=1000 )
    parser.add_argument("-d","--debug", action="store_true", default=False )
    parser.add_argument("-s","--slice", default=None )
    parser.add_argument("-l","--lexical", action="store_true" )

    args = parser.parse_args()
    if args.debug:
        print(args)
    pass  
    if args.slice is not None: 
        args.slice = slice( *map(int,args.slice.replace("_","-").split(":")) )
    pass 
    if args.debug:
        print("args.slice %r " % args.slice)
    pass 

    np.set_printoptions(suppress=True, precision=4, linewidth=200, threshold=int(args.threshold))

    dirs = filter(is_dir_, args.paths)
    npys = filter(is_npy_, args.paths)
    if args.debug:
        print("dirs:%s" % repr(dirs))
        print("npys:%s" % repr(npys))
    pass

    if len(dirs) > 0:
        for p in dirs: 
            dump_tree(p, args)
        pass
    elif len(dirs) == 0 and len(npys) == 0:
        dump_tree(".", args)
    else:
        pass
    pass

    if len(npys) > 0:
        nd = odict()
        labels = "abcdefghijklmnopqrstuvwxyz"
        labels += labels.upper()

        ars = map(A, npys)

        if args.lexical:
            s_ars = sorted(ars,reverse=False, key=lambda ar:ar.path)
        else: 
            s_ars = sorted(ars,reverse=True, key=lambda ar:len(ar.a))
        pass

        for i,ar in enumerate(s_ars):
            print("%s : %r" % (labels[i],ar))
            if args.verbose > 0:
                if args.float:
                    ar.fdump(args.slice)
                pass
                if args.int:
                    ar.idump(args.slice)
                pass
            pass 
            nd[i] = ar
        pass
        ln = len(ars)

        if ln > 0: a=nd[0].a
        if ln > 1: b=nd[1].a
        if ln > 2: c=nd[2].a
        if ln > 3: d=nd[3].a
        if ln > 4: e=nd[4].a
        if ln > 5: f=nd[5].a
        if ln > 6: g=nd[6].a
        if ln > 7: h=nd[7].a
        if ln > 8: i=nd[8].a
        if ln > 9: j=nd[9].a
        if ln > 10: k=nd[10].a
        if ln > 11: l=nd[11].a
        if ln > 12: m=nd[12].a
        if ln > 13: n=nd[13].a
        if ln > 14: o=nd[14].a
        if ln > 15: p=nd[15].a
        if ln > 16: q=nd[16].a

        if ln == 2:
            compare(a,b) 
        pass
    pass
        

