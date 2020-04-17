#!/usr/bin/env python
"""
find_package.py
=================

Following the simple heuristic of looking for *Config.cmake or *-config.cmake 
in directories of the CMAKE_PREFIX_PATH envvar attempt to 
predict which package CMake will find without asking CMake.

::

   export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$(opticks-prefix)/externals


NB this is only useful if it can be done simply and is usually correct, 
otherwise might as well use CMake by generating a CMakeLists.txt 
and get the definitive answer by parsing CMake tealeaves.

"""
import os, re


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
        self.libdir = libdir

    def __repr__(self):
        return "%-30s : %s " % (self.pkg, self.path)


class FindPkgs(object):

    CONFIG = re.compile("(?P<pfx>\S*?)-?[cC]onfig.cmake$")

    PRUNE = ["Modules", "Linux-g++"]  # Linux-g++ is kludge to avoid Geant4 circular links

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
            self.find_config_r(base,0)
        pass   

    def find_config_r(self, base, depth):
        names = os.listdir(base)
        for name in names:
            path = os.path.join(base, name)
            if os.path.isdir(path) and not name in self.PRUNE:
                self.find_config_r(path, depth+1)
            else:
                m = self.CONFIG.match(name)
                if not m: continue
                pfx = m.groupdict()['pfx']  
                pkg = Pkg(path, pfx)
                self.pkgs.append(pkg) 
            pass
        pass


if __name__ == '__main__':
    pass

    cpp = os.environ.get("CMAKE_PREFIX_PATH","")
    bases = filter(None, cpp.split(":"))
    #print("\n".join(bases)) 

    fpk = FindPkgs(bases)
    for pkg in fpk.pkgs:
        print(pkg)
    pass

