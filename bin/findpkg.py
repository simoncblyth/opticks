#!/usr/bin/env python
"""
Find.py
========

Common stuff used by find_package.py and pkg_config.py 

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
    @classmethod
    def Compare(cls, a, b ):
        log.info("Compare num_pkgs %d %d " % (len(a), len(b))) 

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


class Find(object):
    @classmethod
    def parse_args(cls, doc, default_mode="cmake"):
        parser = argparse.ArgumentParser(doc)
        parser.add_argument( "names", nargs="*", help="logging level" ) 
        parser.add_argument( "--level", default="info", help="logging level" ) 
        parser.add_argument( "-p", "--prefix",  default=False, action="store_true"  )
        parser.add_argument( "-l", "--libdir",  default=False, action="store_true" )
        parser.add_argument( "-i", "--includedir",  default=False, action="store_true" )
        parser.add_argument( "-f", "--first",   default=False, action="store_true" )
        parser.add_argument( "-x", "--index",  type=int, default=-1 )
        parser.add_argument( "-c", "--count",  action="store_true", default=False )
        parser.add_argument( "-m", "--mode",  choices=['pc', 'cmake', 'cf'], default=default_mode  )


        parser.add_argument( "--casesensitive",  action="store_true", default=False )
        args = parser.parse_args()
        fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
        logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)
        args.lnames = map(lambda _:_.lower(), args.names)
        if args.first:
           args.index = 0 
        pass
        log.debug(args)
        return args 

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


    def select(self, args):
        if len(args.names) > 0 and args.casesensitive:
            pkgs = filter(lambda pkg:pkg.pkg in args.names, self.pkgs)
        elif len(args.lnames) > 0:
            pkgs = filter(lambda pkg:pkg.pkg.lower() in args.lnames, self.pkgs)
        else:
            pkgs = self.pkgs
        pass
        return pkgs[args.index:args.index+1] if args.index > -1 else pkgs




class FindPkgConfigPkgs(Find):

    CONFIG = re.compile("(?P<pfx>\S*?).pc$")

    def __init__(self, bases):
        Find.__init__(self, bases)

    def find_config(self):
        for base in self.bases:  
            if not os.path.isdir(base):
                log.debug("base %s does not exist " % base)
            else:    
                self.find_config_(base)
            pass
        pass   

    def find_config_(self, base):
        """
        :param base: directory in which to look for pc files
        """
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


class FindCMakePkgs(Find):

    CONFIG = re.compile("(?P<pfx>\S*?)-?[cC]onfig.cmake$")

    PRUNE = ["Modules", "Linux-g++", "Darwin-clang"]  # Linux-g++ and Darwin-clang are .. symbolic uplinks that cause infinite recursion

    BUILD = re.compile(".*build.*")

    SUBS = ["lib", "lib64"]


    def __init__(self, bases):
        Find.__init__(self, bases)

    def find_config(self):
        vbases = []
        for base in self.bases:
            for sub in self.SUBS:
                path = os.path.join(*filter(None,[base, sub])) 
                if not os.path.isdir(path): continue
                vbases.append(path)  
            pass
        pass
        for base in vbases:  
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




class Main(object):
    def get_bases(self, var):
        pp = os.environ.get(var,"")
        bases = filter(None, pp.split(":"))
        log.debug("\n".join(["%s:" % var]+bases)) 
        return bases

    def __init__(self, default_mode="cmake"):

        args = Find.parse_args(__doc__, default_mode=default_mode)
        self.args = args

        cm_bases = self.get_bases("CMAKE_PREFIX_PATH")
        fcm = FindCMakePkgs(cm_bases)
        cm_pkgs = fcm.select(args)

        pc_bases = self.get_bases("PKG_CONFIG_PATH")
        fpc = FindPkgConfigPkgs(pc_bases)
        pc_pkgs = fpc.select(args)

        if args.mode in ["pc", "cmake"]:
            pkgs = []
            if args.mode == "pc":
                pkgs = pc_pkgs
            elif args.mode == "cmake":
                pkgs = cm_pkgs 
            else:
                assert 0 
            pass
            self.dump(pkgs)
        else:
            print("CMake")
            self.dump(cm_pkgs)
            print("pkg-config")
            self.dump(pc_pkgs)
            Pkg.Compare(cm_pkgs, pc_pkgs) 
        pass

    def dump(self, pkgs):
        args = self.args
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



if __name__ == '__main__':
    Main(default_mode="compare")
  

