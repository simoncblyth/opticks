#!/usr/bin/env python
"""
okdist.py
=============

Used from Opticks okdist- bash functions.

"""
import os, logging, argparse
log = logging.getLogger(__name__)

from opticks.bin.dist import Dist

class OKDist(Dist):
    """
    Creates a mostly binary tarball of the Opticks build products 
    along with some txt files that are needed at runtime.  These
    include the OpenGL shader sources. 

    With Geant4 libs and data excluded

    305M    /home/blyth/local/opticks/Opticks-0.0.0_alpha.tar


    """
    exclude_dir_name = [ '.git', 
                         '.hg', 
                         'Geant4-10.2.1', 
                         'Geant4-10.4.2']  

    exclude_dir_name_prev = [ 
                         'cmake',     # cmake exported targets and config
                         'pkgconfig',  
                           ]   

    ## dist.py:Dist iterates over each of these dirs 
    bases = [
             'cmake',    
             'tests',                # tree of CTestTestfile.cmake  
             'metadata',
             'lib',               
             'lib64',              
             'ptx',
             'bin',
             'externals',
             'include', 
             'py',                   # installed python module tree
             ]


    extras = []

    def __init__(self, distprefix, distname ):
        """
        :param distprefix:
        :param distname:
        """
        extra_bases = []
        Dist.__init__(self, distprefix, distname, extra_bases)

    def exclude_file(self, name):
        exclude = False 
        if name.endswith(".log"):
            exclude = True
        pass
        return exclude



if __name__ == '__main__':

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(     "--distname",  help="Distribution name including the extension, expect .tar or .tar.gz" )
    parser.add_argument(     "--distprefix",  help="Distribution prefix, ie the top level directory structure within distribution file." )
    parser.add_argument(     "--level", default="info", help="logging level" ) 
    args = parser.parse_args()

    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)

    log.info("distprefix %s distname %s " % (args.distprefix, args.distname))

    dist = OKDist(args.distprefix, args.distname)

    print(dist.large())



