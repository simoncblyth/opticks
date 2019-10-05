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

    bases = ['include', 
             'lib',                  # order 400 executables
             'lib64',              
             'externals/share/bcm',  #  cmake infrastructure enhancement 
             'externals/lib',
             'externals/lib64',
             'externals/OptiX/lib64', 
             'externals/glm/glm',
             'externals/plog/include',
             'externals/include/OpenMesh',
             'externals/include/assimp',
             'externals/include/YoctoGL',
             'externals/include/DualContouringSample',
             'installcache/PTX', 
             'gl',                   # shaders 
             'tests',                # tree of CTestTestfile.cmake  
             'integration', 
             'py',                   # installed python module tree
             'bin',
             'opticksaux',           # a few example GDML files
             'metadata',
             'cmake/Modules',        # infraastructure for Finding, configuring etc..
             ]

    bases_g4 = [
             'externals/config',
             'externals/share/Geant4-10.4.2/data',       # adds about 1.6G to .tar when included
               ] 

    bases_xercesc = [
             'externals/include/xercesc',
                   ]

    extras = []

    def __init__(self, distprefix, distname, include_geant4, include_xercesc ):


        extra_bases = []
        if include_geant4:
            extra_bases += self.bases_g4
        pass
        if include_xercesc:
            extra_bases += self.bases_xercesc
        pass 

        self.include_geant4 = include_geant4 
        self.include_xercesc = include_xercesc 
        Dist.__init__(self, distprefix, distname, extra_bases)

    def exclude_file(self, name):
        exclude = False 
        if name.endswith(".log"):
            exclude = True
        pass
        if name.startswith("libG4OK"):  ## Opticks Geant4 interface lib named like g4 libs
            exclude = False
        elif name.startswith("libG4") and self.include_geant4 == False:
            exclude = True
        elif name.startswith("libxerces") and self.include_xercesc == False:
            exclude = True
        pass
        return exclude



if __name__ == '__main__':

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(     "--distname",  help="Distribution name including the extension, expect .tar or .tar.gz" )
    parser.add_argument(     "--distprefix",  help="Distribution prefix, ie the top level directory structure within distribution file." )
    parser.add_argument(     "--include_geant4",   default=False, action="store_true", help="Include Geant4 libraries and datafiles from the distribution" )
    parser.add_argument(     "--include_xercesc",  default=False, action="store_true", help="Include xercesc libraries and includes in the distribution" )
    parser.add_argument(     "--level", default="info", help="logging level" ) 
    args = parser.parse_args()

    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)

    log.info("distprefix %s distname %s " % (args.distprefix, args.distname))

    dist = OKDist(args.distprefix, args.distname, include_geant4=args.include_geant4, include_xercesc=args.include_xercesc)





