#!/usr/bin/env python
"""
np.py
========

This is intended as a quick and dirty dumper/comparer of npy files,
for quick initial comparisons of small numbers of arrays.

When wishing to get specific to certain collections of npy files, 
move to specific other scripts such as ab.py etc..  

Keep this general.

::

   ipython -i $(which np.py) -- tmp/blyth/OKG4Test/evt/g4live/natural/-1/so.npy source/evt/g4live/natural/-1/so.npy -v

   inp(){ ipython -i $(which np.py) -- $* ; }

   inp tmp/blyth/OKG4Test/evt/g4live/natural/-1/so.npy source/evt/g4live/natural/-1/so.npy -v


python -c "import numpy as np, sys ; np.set_printoptions(suppress=True) ; print np.load(sys.argv[1]) " 
"""
import sys, fnmatch, os, logging, numpy as np, commands, argparse
from collections import OrderedDict as odict

log = logging.getLogger(__name__)

np.set_printoptions(suppress=True, precision=4, linewidth=200)
is_npy_ = lambda _:fnmatch.fnmatch(_,"*.npy")
is_txt_ = lambda _:fnmatch.fnmatch(_,"*.txt")


from opticks.bin.md5 import digest_
from opticks.ana.base import stamp_

def dump_one(a, args):
    print(a.shape)
    if args.float:
        print("f32\n",a.view(np.float32))
    pass
    if args.int:
        print("i32\n",a.view(np.int32))
    pass

def dump_tree(base, args):
    """
    Recursively lists shapes of all .npy files 
    and line lengths of .txt files.
    """
    print(os.path.abspath(base))
    for root, dirs, files in os.walk(base):
        sfiles = sorted(files)
        for name in filter(is_txt_,sfiles):
            path = os.path.join(root, name)
            txt_brief(path)
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
    parser.add_argument("-i","--int", action="store_true", default=False )
    args = parser.parse_args()

    if len(args.paths) == 1:
        p = args.paths[0]
        if os.path.isdir(p):
            dump_tree(p, args)
        else:
            npy_brief(p, args)
        pass 
    elif len(args.paths) == 0:
        dump_tree(".", args)
    else:
        npy = filter(is_npy_, args.paths)
        n = odict()
        labels = "abcd"
        for i,path in enumerate(npy):
            n[i] = npy_brief(path, args, label=labels[i])
        pass
        ln = len(npy)

        if ln > 0: a=n[0]
        if ln > 1: b=n[1]
        if ln > 2: c=n[2]
        if ln > 3: d=n[3]

        if ln == 2:
            compare(a,b) 
        pass
    pass
        

