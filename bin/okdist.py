#!/usr/bin/env python
"""
okdist.py
=============

Used from Opticks okdist- bash functions.

"""
import os, tarfile, logging, argparse
log = logging.getLogger(__name__)

class Dist(object):
    """
    Creates a mostly binary tarball of the Opticks build products 
    along with some txt files that are needed at runtime.  These
    include the OpenGL shader sources. 

    With Geant4 libs and data excluded

    305M    /home/blyth/local/opticks/Opticks-0.0.0_alpha.tar

    """
    exclude_dir_name = [ '.git', 
                         '.hg', 
                         'cmake',     # cmake exported targets and config
                         'pkgconfig',  
                         'Geant4-10.2.1', 
                         'Geant4-10.4.2']  

    bases = ['include', 
             'lib',                  # order 400 executables
             'lib64',              
             'externals/lib',
             'externals/lib64',
             'externals/OptiX/lib64', 
             'installcache/PTX', 
             'gl',                   # shaders 
             'tests',                # tree of CTestTestfile.cmake  
             'integration', 
             'py',                   # installed python module tree
             'bin',
             'opticksaux',           # a few example GDML files
             'metadata',
             ]

    bases_g4 = [
             'externals/config',
             'externals/share/Geant4-10.4.2/data',       # adds about 1.6G to .tar when included
               ] 

    extras = []



    @classmethod
    def Create(cls, distprefix, distname, exclude_geant4=False ):
        if not exclude_geant4:
             cls.bases.extend(cls.bases_g4)
        pass
        return cls(distprefix, distname, exclude_g4lib=exclude_geant4 )
 

    def __init__(self, distprefix, distname, exclude_g4lib=False):
        """
        :param distprefix: top level dir structure within distribution 
        :param distname:

        Example of three element prefix::

             Opticks/0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg

        Create a tarfile named using the first portion of the prefix 
        with upper containing directory structure using the prefix 
        included in the archive.

        To extract into current directory without these container dirs 
        use the strip option::

            tar zxvf Opticks-0.1.0.tar.gz --strip 2

        """
        self.exclude_g4lib = exclude_g4lib

        assert len(distprefix.split("/")) in [2,3] , "expecting 2 or 3 element distprefix %s " % distprefix 

        if distname.endswith(".tar.gz"): 
            mode = "w:gz"
        elif distname.endswith(".tar"):
            mode = "w"
        elif distname.endswith(".zip"):
            mode = None
        else:
            mode = None
        pass 

        assert not mode is None, "distribution format for distname %s is not supported " % (distname)
        log.info("writing distname %s mode %s " % (distname, mode) )

        self.prefix = distprefix
        self.dist = tarfile.open(distname, mode)    

        for base in self.bases:
            self.recurse_(base, 0) 
        pass
        for extra in self.extras:
            assert os.path.exists(extra), extra
            self.add(extra)
        pass 
        self.dist.close()

    def exclude_dir(self, name):
        exclude = name in self.exclude_dir_name
        if exclude:
            log.info("exclude_dir %s " % name) 
        pass
        return exclude  

    def exclude_file(self, name):
        exclude = False 
        if name.endswith(".log"):
            exclude = True
        pass
        if name.startswith("libG4OK"):  ## Opticks Geant4 interface lib named like g4 libs
            exclude = False
        elif name.startswith("libG4") and self.exclude_g4lib:
            exclude = True
        pass
        #if exclude:
        #    log.info("exclude_file %s " % name) 
        pass
        return exclude

    def add(self, path):
        log.debug(path)
        arcname = os.path.join(self.prefix,path) 
        self.dist.add(path, arcname=arcname, recursive=False)

    def recurse_(self, base, depth ):
        """
        NB all paths are relative to base 
        """
        assert os.path.isdir(base), "expected directory %s does not exist " % base

        log.info("base %s depth %s " % (base, depth))

        names = os.listdir(base)
        for name in names:
            path = os.path.join(base, name)
            if os.path.isdir(path):
                 exclude = self.exclude_dir(name)
                 if not exclude:
                     self.recurse_(path, depth+1)
                 pass
            else:
                 exclude = self.exclude_file(name)
                 if not exclude:
                     self.add(path)
                 pass
            pass
        pass
    pass
pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(     "--distname",  help="Distribution name including the extension, expect .tar or .tar.gz" )
    parser.add_argument(     "--distprefix",  help="Distribution prefix, ie the top level directory structure within distribution file." )
    parser.add_argument(     "--exclude_geant4",   action="store_true", help="Exclude Geant4 libraries and datafiles from the distribution" )
    parser.add_argument(     "--level", default="info", help="logging level" ) 
    args = parser.parse_args()

    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)

    log.info("distprefix %s distname %s " % (args.distprefix, args.distname))

    dist = Dist.Create(args.distprefix, args.distname, exclude_geant4=args.exclude_geant4  )



