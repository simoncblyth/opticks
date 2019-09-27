#!/usr/bin/env python
"""
dist.py
=============

Used from::

    okdist- okdist.py 
    scdist- scdist.py 


"""
import os, tarfile, logging, argparse
log = logging.getLogger(__name__)


class Dist(object):
    def __init__(self, distprefix, distname, extra_bases):
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

        assert len(distprefix.split("/")) in [1,2,3] , "expecting 1, 2 or 3 element distprefix %s " % distprefix 

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
        for base in extra_bases:
            self.recurse_(base, 0) 
        pass
        for extra in self.extras:
            assert os.path.exists(extra), extra
            self.add(extra)
        pass 
        self.dist.close()

    def add(self, path):
        log.debug(path)
        arcname = os.path.join(self.prefix,path) 
        self.dist.add(path, arcname=arcname, recursive=False)

    def exclude_dir(self, name):
        exclude = name in self.exclude_dir_name
        if exclude:
            log.info("exclude_dir %s " % name) 
        pass
        return exclude  

    def exclude_file(self, name):
        exclude = False 
        return exclude

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


