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
    include the OpenGL shader sources and OpticksPhoton.h which is parsed 
    at runtime to instanciate OpticksFlags.


    lib64/cmake 
    lib64/pkgconfig
         These are the cmake exported targets, 
          so for CMake level integration will need to include them  


    Bases
         
    lib 
         contains order 400 executables
   


    Not installed things which perhaps should be 

    bin/ok.sh 
 
 
 
    """
    exclude_dir_name = ['cmake','pkgconfig',  'Geant4-10.2.1', 'Geant4-10.4.2']    # G4 exclusion temporarily 
    #exclude_dir_name = ['Geant4-10.2.1', 'Geant4-10.4.2']    # G4 exclusion temporarily 

    #bases = ['lib','lib64','externals/lib','externals/lib64','externals/OptiX/lib64', 'installcache', 'gl', 'geocache' ]
    bases = ['lib','lib64','externals/lib','externals/lib64','externals/OptiX/lib64', 'installcache', 'gl' ]

    extras = ['include/OpticksCore/OpticksPhoton.h', ]


    def __init__(self, prefix, ext=".tar.gz"):
        """
        :param prefix: example Opticks-0.1.0/x86_64-centos7-gcc48-dbg 

        Create a tarfile named using the first portion of the prefix 
        with upper containing directory structure using the prefix 
        included in the archive.

        To extract into current directory without these container dirs 
        use the strip option::

            tar zxvf Opticks-0.1.0.tar.gz --strip 2

        """
        assert len(prefix.split("/")) == 2 , "expecting two element prefix %s " % prefix 

        distname = os.path.dirname(prefix)  
        tarname = "%s%s" % (distname, ext)

        if ext == ".tar.gz":
            mode = "w:gz"
        else:
            mode = "w"
        pass  

        log.info("writing %s " % tarname )

        self.prefix = prefix
        self.tar = tarfile.open(tarname, mode)    
        for base in self.bases:
            self.recurse_(base, 0) 
        pass
        for extra in self.extras:
            assert os.path.exists(extra), extra
            self.add(extra)
        pass 
        self.tar.close()

    def exclude_dir(self, name):
        return name in self.exclude_dir_name

    def exclude_file(self, name):
        return name.startswith("libG4") or name.endswith(".log")

    def add(self, path):
        print(path)
        arcname = os.path.join(self.prefix,path) 
        self.tar.add(path, arcname=arcname, recursive=False)

    def recurse_(self, base, depth ):
        """
        NB all paths are relative to base 
        """
        assert os.path.isdir(base), "expected directory %s does not exist " % base

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
    parser.add_argument(     "--prefix",  help="Distribution prefix" )
    parser.add_argument(     "--ext",  default=".tar", help="Distribution extension, expect .tar or .tar.gz" )
    parser.add_argument(     "--level", default="info", help="logging level" ) 
    args = parser.parse_args()

    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)
    log.info("prefix %s ext %s " % (args.prefix, args.ext))

    dist = Dist(args.prefix, ext=args.ext )



