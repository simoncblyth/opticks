#!/usr/bin/env python
"""
datedfolder.py
================

Find dated folders beneath a base directory argument or beneath the current directory.

::

    [blyth@localhost issues]$ ~/opticks/ana/datedfolder.py /home/blyth/local/opticks/results/geocache-bench
    INFO:__main__:DatedFolder.find searching for date stamped folders beneath : /home/blyth/local/opticks/results/geocache-bench 
    /home/blyth/local/opticks/results/geocache-bench/OFF_TITAN_RTX/20190422_175618
    /home/blyth/local/opticks/results/geocache-bench/ON_TITAN_RTX/20190422_175618
    /home/blyth/local/opticks/results/geocache-bench/OFF_TITAN_V/20190422_175618
    /home/blyth/local/opticks/results/geocache-bench/ON_TITAN_V/20190422_175618
    /home/blyth/local/opticks/results/geocache-bench/OFF_TITAN_V_AND_TITAN_RTX/20190422_175618
    [blyth@localhost issues]$ 

"""
from datetime import datetime
import os, re, sys, logging
log = logging.getLogger(__name__)

class DateParser(object):
    ptn = re.compile("(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})")
    def __call__(self, name):
        """
        :param name: basename of a directory 
        :return dt: datetime if matches the date format 20160819_153245
        """
        m = self.ptn.match(name)
        if m is None:return None
        if len(m.groups()) != 6:return None
        dt = datetime(*map(int, m.groups()))
        return dt 

dateparser = DateParser() 

def finddir(base, dirfilter=lambda _:True, relative=True):
    paths = []
    for root, dirs, files in os.walk(base):
        for name in dirs:
            path = os.path.join(root,name)
            d = dirfilter(path)
            if d is not None:
                paths.append(path[len(base)+1:] if relative else path)
            pass
        pass
    pass 
    return paths

class DatedFolder(object):
    @classmethod
    def find(cls, base):
        """
        Groups of executable runs are grouped together by them using 
        the same datestamp datedfolder name.
        So there can be more dirs that corresponding dfolds and dtimes.
        """
        while base.endswith("/"):
            base = base[:-1]
        pass

        df = cls()
        log.info("DatedFolder.find searching for date stamped folders beneath : %s " % base)
        metamap = {}
        dirs = finddir(base, df)                       

        dfolds = list(set(map(os.path.basename, dirs))) # list of unique basenames of the dated folders
        dtimes = map(dateparser, dfolds )               # list of datetimes
        return dirs, dfolds, dtimes 
 
    def __call__(self, path):
        name = os.path.basename(path) 
        #log.info(name)
        return dateparser(name)


if __name__ == '__main__':
     logging.basicConfig(level=logging.INFO)
     base = sys.argv[1] if len(sys.argv) > 1 else "." 
     dirs, dfolds, dtimes = DatedFolder.find(base)

     print("\n".join(dfolds))
     for df in sorted(dfolds):
         print("\n".join(filter(lambda _:_.endswith(df),dirs)))
     pass 

    
    
