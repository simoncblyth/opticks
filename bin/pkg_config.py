#!/usr/bin/env python
"""
pkg_config.py
=================

This is to check that pkg-config finds packages 
in the expected manner.  See bin/oc.bash 



"""
import os, re, logging, argparse
log = logging.getLogger(__name__)



def getlibdir(path):
    """
    Often no lib ?
    """
    fold, name = os.path.split(path) 
    elem = fold.split("/")
    jlib = -1
    for i in range(len(elem)):
        j = len(elem)-i-1
        if elem[j] in ["lib","lib64"]:
            jlib = j
            break
        pass   
    pass
    return "/".join(elem[:jlib+1]) if jlib > -1 else ""  


class Pkg(object):
    def __init__(self, path, pkg):
        self.path = path
        self.pkg = pkg

        libdir = getlibdir(path)   
        prefix = os.path.dirname(libdir)
        includedir = os.path.join(prefix, "include")
    
        self.libdir = libdir
        self.prefix = prefix
        self.includedir = includedir

    def __repr__(self):
        return "%-30s : %s " % (self.pkg, self.path)



class FindPkgs(object):

    CONFIG = re.compile("(?P<pfx>\S*?).pc$")

    def __init__(self, bases):
        ubases = []  
        for base in bases:
            if not base in ubases:
                ubases.append(base)
            pass   
        pass
        self.bases = ubases
        self.pkgs = []
        self.find_config()

    def find_config(self):
        for base in self.bases:  
            self.find_config_(base)
        pass   

    def find_config_(self, base):
        log.debug("find_config_ %s " % (base)) 
        names = os.listdir(base)
        for name in names:
            path = os.path.join(base, name)
            if not os.path.isdir(path):
                m = self.CONFIG.match(name)
                if not m: continue
                pfx = m.groupdict()['pfx']  
                pkg = Pkg(path, pfx)
                self.pkgs.append(pkg) 
            pass
        pass

    def select(self, args):
        if len(args.names) > 0 and args.casesensitive:
            pkgs = filter(lambda pkg:pkg.pkg in args.names, self.pkgs)
        elif len(args.lnames) > 0:
            pkgs = filter(lambda pkg:pkg.pkg.lower() in args.lnames, self.pkgs)
        else:
            pkgs = self.pkgs
        pass
        return pkgs[args.index:args.index+1] if args.index > -1 else pkgs




def parse_args(doc):
    parser = argparse.ArgumentParser(doc)
    parser.add_argument( "names", nargs="*", help="logging level" ) 
    parser.add_argument( "--level", default="info", help="logging level" ) 
    parser.add_argument( "-p", "--prefix",  default=False, action="store_true"  )
    parser.add_argument( "-l", "--libdir",  default=False, action="store_true" )
    parser.add_argument( "-i", "--includedir",  default=False, action="store_true" )
    parser.add_argument( "-f", "--first",   default=False, action="store_true" )
    parser.add_argument( "-x", "--index",  type=int, default=-1 )
    parser.add_argument( "-c", "--count",  action="store_true", default=False )
    parser.add_argument( "--casesensitive",  action="store_true", default=False )
    args = parser.parse_args()
    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)
    args.lnames = map(lambda _:_.lower(), args.names)
    return args 


if __name__ == '__main__':
    pass
    args = parse_args(__doc__)
    if args.first:
       args.index = 0 
    pass
    #print(args)
   
    pp = os.environ.get("PKG_CONFIG_PATH","")
    bases = filter(None, pp.split(":"))

    #log.info("\n".join(["bases:"]+bases)) 

    fpk = FindPkgs(bases)
    pkgs = fpk.select(args)
    count = len(pkgs)

    if args.count:
        print(count)
    else:    
        for pkg in pkgs:
            if args.libdir:
                print(pkg.libdir)
            elif args.includedir:
                print(pkg.includedir)
            elif args.prefix:
                print(pkg.prefix)
            else:
                print(pkg)
            pass
        pass
    pass



