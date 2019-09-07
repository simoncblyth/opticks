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


::

   ipython -i $(which np.py) -- tmp/blyth/OKG4Test/evt/g4live/natural/-1/so.npy source/evt/g4live/natural/-1/so.npy -v

   inp(){ ipython -i $(which np.py) -- $* ; }

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
        n = odict()
        labels = "abcdefghijklmnopqrstuvwxyz"
        labels += labels.upper()
        for i,path in enumerate(npys):
            n[i] = npy_brief(path, args, label=labels[i])
        pass
        ln = len(npys)

        if ln > 0: a=n[0]
        if ln > 1: b=n[1]
        if ln > 2: c=n[2]
        if ln > 3: d=n[3]

        if ln == 2:
            compare(a,b) 
        pass
    pass
        

