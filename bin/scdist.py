#!/usr/bin/env python
"""
scdist.py
=============

Used from Opticks scdist- bash functions.

"""
import os, logging, argparse
log = logging.getLogger(__name__)

from opticks.bin.dist import Dist

class SCDist(Dist):
    """
    """
    exclude_dir_name = [
                       ]  

    bases = [
             'bin', 
             'geocache', 
             'rngcache',   
             'metadata',
             ]

    extras = []

    def __init__(self, distprefix, distname):
        Dist.__init__(self, distprefix, distname, extra_bases=[])

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

    dist = SCDist(args.distprefix, args.distname )





