#!/usr/bin/env python
"""
python -c "import numpy as np, sys ; np.set_printoptions(suppress=True) ; print np.load(sys.argv[1]) " 
"""
import sys, fnmatch, os, logging, numpy as np, commands
try: 
    from hashlib import md5
except ImportError: 
    from md5 import md5

log = logging.getLogger(__name__)

np.set_printoptions(suppress=True, precision=4, linewidth=200)
is_npy_ = lambda _:fnmatch.fnmatch(_,"*.npy")
is_txt_ = lambda _:fnmatch.fnmatch(_,"*.txt")


cumulative = None

def digest_(path):
    """
    :param path:
    :return: md5 hexdigest of the content of the path or None if non-existing path

    http://stackoverflow.com/questions/1131220/get-md5-hash-of-a-files-without-open-it-in-python

    Confirmed to give same hexdigest as commandline /sbin/md5::

        md5 /Users/blyth/workflow/notes/php/property/colliers-4q2011.pdf 
        MD5 (/Users/blyth/workflow/notes/php/property/colliers-4q2011.pdf) = 3a63b5232ff7cb6fa6a7c241050ceeed

    """
    global cumulative

    if not os.path.exists(path):return None
    if os.path.isdir(path):return None
    dig = md5()

    if cumulative is None:
        cumulative = md5() 

    with open(path,'rb') as f: 
        for chunk in iter(lambda: f.read(8192),''): 
            dig.update(chunk)
            cumulative.update(chunk)
        pass
    return dig.hexdigest()






def dump_one(a, verbose):
    print(a.shape)
    print("f32\n",a.view(np.float32))
    print("i32\n",a.view(np.int32))

def dump_tree(base=".", verbose=0):
    """
    Recursively lists shapes of all .npy files 
    and line lengths of .txt files.
    """
    print(os.path.abspath(base))
    for root, dirs, files in os.walk(base):
        sfiles = sorted(files)
        for name in filter(is_txt_,sfiles):
            path = os.path.join(root, name)
            lines = len(file(path, "r").readlines())
            print("%40s : %s " % ( path, lines))
        pass
        for name in filter(is_npy_,sfiles):
            path = os.path.join(root, name)
            a = np.load(path)
            ## 
            ##fdig = commands.getoutput("md5 %s" % path).split()[-1]   
            ## md5 on macOS, md5sum on Linux : so use python for consistency
            ## 
            fdig = digest_(path)
            print("%40s : %20s : %s " % ( path, repr(a.shape), fdig ))
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

        

