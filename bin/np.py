#!/usr/bin/env python
"""
python -c "import numpy as np, sys ; np.set_printoptions(suppress=True) ; print np.load(sys.argv[1]) " 
"""
import sys, fnmatch, os, logging, numpy as np
log = logging.getLogger(__name__)

np.set_printoptions(suppress=True, precision=4, linewidth=200)
is_npy_ = lambda _:fnmatch.fnmatch(_,"*.npy")
is_txt_ = lambda _:fnmatch.fnmatch(_,"*.txt")

def dump_one(a, verbose):
    print a.shape
    print "f32\n",a.view(np.float32)
    print "i32\n",a.view(np.int32)

def dump_tree(base=".", verbose=0):
    """
    Recursively lists shapes of all .npy files 
    and line lengths of .txt files.
    """
    print os.path.abspath(base)
    for root, dirs, files in os.walk(base):
        for name in filter(is_txt_,files):
            path = os.path.join(root, name)
            lines = len(file(path, "r").readlines())
            print "%20s : %s " % ( path, lines)
        pass
        for name in filter(is_npy_,files):
            path = os.path.join(root, name)
            a = np.load(path)
            print "%20s : %s " % ( path, repr(a.shape))
            if verbose > 0:
                dump_one(a, verbose)
            pass
        pass   
    pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = sys.argv[1:]
    verbose = os.environ.get("VERBOSE", 0)
 
    #log.info("args : %s " % repr(args))
    if len(args) == 1:
        p = args[0]
        if os.path.isdir(p):
            dump_tree(p, verbose)
        else:
            a = np.load(p)
            dump_one(a, verbose)
        pass 
    elif len(args) == 0:
        dump_tree(".", verbose)
    else:
        pass

        

