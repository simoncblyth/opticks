#!/usr/bin/env python
"""
CMakeModules.py
==================

This copies cmake/Modules into the installation directory 
which is necessary to allow building against the release. 

Usage Example
---------------
 
Note that the destination directory is deleted and populated on every run 

::

    [blyth@localhost ~]$  CMakeModules.py $(opticks-home) --dest $(opticks-dir)




"""
import sys, re, os, logging, argparse, shutil
log = logging.getLogger(__name__)

class SourceTree(object):
    MODULES_DIR = "cmake/Modules"
    SKIPDIRS = [
                'old', 
                'inactive',
                'include', 
               ] 

    def __init__(self, home, dest):
        self.src = os.path.join(home, self.MODULES_DIR)
        self.dst = os.path.join(dest, self.MODULES_DIR)
        if os.path.isdir(self.dst):
            log.debug("remove dst tree %s " % self.dst )
            shutil.rmtree( self.dst )
        pass
        self.filtercopy()

    def filtercopy(self):
        log.info("Copying from src %s to dst %s " % (self.src, self.dst ))
        shutil.copytree( self.src, self.dst,  symlinks=False, ignore=self ) 

    def skipdir(self, name):
        return name in self.SKIPDIRS

    def skipfile(self, name):
        return False 

    def __call__(self, src, names):
        """
        # contining will not ignore, hence selects 
        """
        ignore = []
        for name in names:
            path = os.path.join(src, name)
            if os.path.isdir(path) and not self.skipdir(name): continue
            if os.path.isfile(path) and not self.skipfile(name): continue
            ignore.append(name) 
        pass 
        return ignore
          

if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(     "--home",  default=os.path.expanduser("~/opticks"), help="Base opticks-home directory in which to look for cmake/Modules " )
    parser.add_argument(     "--dest", default="/tmp/test-CMakeModules-py", help="destination directory inside which a cmake/Modules directory will be removed if present, recreated and populated" ) 
    parser.add_argument(     "--level", default="info", help="logging level" ) 
    args = parser.parse_args()

    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)

    src = SourceTree(args.home, args.dest)



