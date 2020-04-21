#!/usr/bin/env python
"""
find_package.py
=================

Following the simple heuristic of looking for *Config.cmake or *-config.cmake 
in directories of the CMAKE_PREFIX_PATH envvar attempt to 
predict which package CMake will find without asking CMake.

::

   unset CMAKE_PREFIX_PATH
   export CMAKE_PREFIX_PATH=$(opticks-prefix):${CMAKE_PREFIX_PATH}:$(opticks-prefix)/externals
   find_package.py 

   CMAKE_PREFIX_PATH=$(opticks-prefix):$(opticks-prefix)/externals find_package.py 
   CMAKE_PREFIX_PATH=$(opticks-prefix) find_package.py 
   CMAKE_PREFIX_PATH=$(opticks-prefix)/externals find_package.py 
   # NB do not include the "lib" in the prefix

::

   find_package.py Boost --first --libdir


NB this is only useful if it can be done simply and is usually correct, 
otherwise might as well use CMake by generating a CMakeLists.txt 
and get the definitive answer by parsing CMake tealeaves.

Thoughts
---------

Do not want to rely on this script at first order as it is adding 
another resolver to the already two that need to be matched.  
Better to arrange that the two resolvers (CMake and pkg-config) 
yield matched results by making sure that PKG_CONFIG_PATH is set appropriately 
based on CMAKE_PREFIX_PATH.  junotop/bashrc.sh will do this so long as
the directories exist.

BUT : Boost and Geant4 lack lib/pkgconfig/name.pc files, xerces-c has one

So this script can have the role of fixing these omitted pc 
as a workaround until the Boost and Geant4 installs do 
the correct thing themselves. Do not hold breath it has been 
years since people have been asking for this. 


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

    CONFIG = re.compile("(?P<pfx>\S*?)-?[cC]onfig.cmake$")

    PRUNE = ["Modules", "Linux-g++", "Darwin-clang"]  # Linux-g++ and Darwin-clang are .. symbolic uplinks that cause infinite recursion

    BUILD = re.compile(".*build.*")

    SUBS = ["lib", "lib64"]

    def __init__(self, bases):
        ubases = []  
        for base in bases:
            if not base in ubases:
                ubases.append(base)
            pass   
        pass

        vbases = []
        for base in ubases:
            for sub in self.SUBS:
                path = os.path.join(*filter(None,[base, sub])) 
                if not os.path.isdir(path): continue
                vbases.append(path)  
            pass
        pass

        self.bases = vbases
        self.pkgs = []
        self.find_config()

    def find_config(self):
        for base in self.bases:  
            self.find_config_r(base,0)
        pass   

    def find_config_r(self, base, depth):
        log.debug("find_config_r %2d %s " % (depth, base)) 
        names = os.listdir(base)
        for name in names:
            path = os.path.join(base, name)
            if os.path.isdir(path):
                if name in self.PRUNE:
                    pass
                else:
                    m = self.BUILD.match(name) 
                    if m:
                        log.debug("build match %s " % name)
                    else:
                        self.find_config_r(path, depth+1)
                    pass
            else:
                m = self.CONFIG.match(name)
                if not m: continue
                pfx = m.groupdict()['pfx']  
                if len(pfx) == 0 or pfx == "CTest" or pfx.startswith("BCM") or pfx.find("Targets-") > -1: continue

                pkg = Pkg(path, pfx)
                self.pkgs.append(pkg) 
            pass
        pass

    def select(self, args):
        if len(args.names) > 0:
            pkgs = filter(lambda pkg:pkg.pkg in args.names, self.pkgs)
        else:
            pkgs = self.pkgs
        pass
        return pkgs[0:1] if args.first else pkgs



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
    args = parser.parse_args()
    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)
    return args 


if __name__ == '__main__':
    pass
    args = parse_args(__doc__)
    if args.first:
       args.index = 0 
    pass

    #print(args)
   
    pp = os.environ.get("CMAKE_PREFIX_PATH","")
    bases = filter(None, pp.split(":"))

    #log.info("\n".join(["bases:"]+bases)) 

    fpk = FindPkgs(bases)
    pkgs = fpk.select(args)
    count = len(pkgs)

    if args.index > -1:
        pkgs = pkgs[args.index:args.index+1]
    pass

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

