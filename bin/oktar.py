#!/usr/bin/env python
"""
oktar.py
=============

The common prefix includes the trailing slash::

   ...
   Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/cmake/Modules/FindPLog.cmake
   Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/cmake/Modules/FindGLM.cmake

   Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/ 3279 

"""
import os, logging, sys, tarfile, argparse, shutil
log = logging.getLogger(__name__)

class OKTar(object):
    """
    """
    def __init__(self, path, base):

        path = os.path.expanduser(path)
        name = os.path.basename(path)
        stem = os.path.splitext(name)[0]

        t = tarfile.open(path, "r")
        n = t.getnames()

        pfx = os.path.commonprefix(n)
        assert pfx[-1] == os.sep, ("expect a trailing slash", pfx)
        pfx = pfx[:-1] 
        elem = pfx.split(os.sep)
        assert len(elem) == 2, elem  
        assert stem == elem[0], ( stem, elem[0], "stem of tarball name must match the top level prefix element inside" )

        self.t = t 
        self.n = n 
        self.path = path 
        self.base = base
        self.pfx = pfx 

    def dump(self):
        for mi in self.t.getmembers():
            sz = mi.size/1e6
            if sz < 1.: continue
            print(" %10.3f : %s " % ( sz, mi.name))
        pass   

    def explode(self):
        self.prepare_base()
        pass
        xdir = os.path.join(self.base, self.pfx)
        if os.path.isdir(xdir):
            log.info("common prefix extraction dir exists already %s " % xdir)
            log.info("removing xdir %s " % xdir )
            shutil.rmtree(xdir) 
        pass    
        log.info("from base %s extracting tarball with common prefix %s " % (self.base, self.pfx)) 
        self.t.extractall(self.base) 

    def prepare_base(self):
        if not os.path.isdir(self.base):
            log.info("creating base %s " % self.base)
            os.makedirs(self.base)
        pass  

    def __str__(self):
        return "\n".join(self.n)
    def __repr__(self):
        return "%s %d " % ( self.pfx, len(self.n)) 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(__doc__)

    base = "/tmp/cvmfs/opticks.ihep.ac.cn/ok/releases"
    parser.add_argument( "path",  nargs=1, help="Path of distribution tarball, eg ~/Opticks-0.0.0_alpha.tar " )
    parser.add_argument( "--base",  default=base, help="Path from which to explode tarballs %(default)s ")   
    parser.add_argument( "--explode", action="store_true", default=False, help="Explode the tarball ")   
    parser.add_argument( "--dump",    action="store_true", default=False, help="Dump names and sizes of tarball members ")   
    parser.add_argument( "--level", default="info", help="logging level" ) 
    args = parser.parse_args()

    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)

    t = OKTar(args.path[0], args.base)
    print(repr(t))
    #print(t)

    if args.dump: 
        t.dump()
    pass
 
    if args.explode: 
        t.explode()
    pass



