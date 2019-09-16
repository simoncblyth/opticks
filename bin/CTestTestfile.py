#!/usr/bin/env python
"""
CTestTestfile.py
==================

This enables ctest running of installed tests, 
without the full build tree.

CTestTestfile.cmake files which list unit tests are copied 
from the build tree into a newly created tree with only these files.
A top level CTestTestfile.cmake composed of top level subdirs is added, 
which allows all tests to be run with a single ctest command.

Usage Example
---------------
 
Note that the destination directory is deleted and populated on every run 

::

    [blyth@localhost ~]$  CTestTestfile.py $(opticks-bdir) --dest /tmp/tests
    remove dest tree /tmp/tests 
    Copying CTestTestfile.cmake files from buildtree /home/blyth/local/opticks/build into a new destination tree /tmp/tests 
    write testfile to /tmp/tests/CTestTestfile.cmake 


Workflow
-----------

::

    CTestTestfile.py $(opticks-bdir) --dest /tmp/tests

    cd /tmp/tests

    ctest.sh 
        simple ctest wrapper to tee ctest.log and set non-interactive options 

    ctest -N
         list names of tests without running them 

    ctest -N -R SysRapTest.SEnvTest
    ctest -N -R IntegrationTests
         list tests matching a patterm

    ctest -R IntegrationTests --output-on-failure
         run tests matching a pattern    
 

"""
import sys, re, os, logging, argparse, shutil
from opticks.bin.CMakeLists import OpticksCMakeProj 
log = logging.getLogger(__name__)

class BuildTree(object):
    NAME = "CTestTestfile.cmake"
    SKIPDIRS = ["CMakeFiles", "Testing", ]

    def __init__(self, root, projs):
        self.root = root
        self.projs = projs

    def filtercopy(self, dstbase):
        for proj in self.projs:
            src = os.path.join(self.root, proj)
            dst = os.path.join(dstbase, proj)  
            if os.path.isdir(dst):
                log.debug("remove dst tree %s " % dst )
                shutil.rmtree( dst )
            pass
            shutil.copytree( src, dst,  symlinks=False, ignore=self ) 
        pass
        top = os.path.join( dstbase, self.NAME )
        return top 

    def skipdir(self, name):
        return name in self.SKIPDIRS

    def skipfile(self, name):
        return not name == self.NAME 

    def __call__(self, src, names):
        ignore = []
        for name in names:
            path = os.path.join(src, name)
            if os.path.isdir(path) and not self.skipdir(name): continue
            if os.path.isfile(path) and not self.skipfile(name): continue
            # contining will not ignore, so will select 
            ignore.append(name) 
        pass 
        return ignore
          

if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(     "root",  nargs=1, help="Base directory in which to look for CTestTestfile.cmake " )
    parser.add_argument(     "--level", default="info", help="logging level" ) 
    parser.add_argument(     "--dest", default="/tmp/tests", help="destination directory tree to be removed, recreated and populated" ) 
    args = parser.parse_args()
    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)

    ok = OpticksCMakeProj()
    
    bdir = args.root[0]
    dest = args.dest

    bt = BuildTree(bdir, projs=ok.subdirs)
    top = bt.filtercopy( dest )

    log.info("Copying %s files from buildtree %s into a new destination tree %s " % (BuildTree.NAME,  bdir, dest ))
    log.info("write testfile to %s " % top ) 
    ok.write_testfile( top )


