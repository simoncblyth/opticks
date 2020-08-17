#!/usr/bin/env python
#
# Copyright (c) 2019 Opticks Team. All Rights Reserved.
#
# This file is part of Opticks
# (see https://bitbucket.org/simoncblyth/opticks).
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and 
# limitations under the License.
#

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
        :return dt: datetime if matches the date format 20160819_153245 otherwise return None
        """
        m = self.ptn.match(name)
        if m is None:return None
        if len(m.groups()) != 6:return None
        dt = datetime(*map(int, m.groups()))
        return dt 

dateparser = DateParser() 

def finddir(base, dirfilter=lambda _:True, relative=True):
    """
    :param base: directory to walk looking for directories to be dirfiltered
    :param dirfilter: function that takes directory path argument and returns true when selected
    :param relative: when True returns the base relative path, otherwise absolute paths are returned
    :return paths: all directory paths found that satisfy the filter
    """
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
        :param base: directory
        :return dirs, dfolds, dtimes: 
 
        Groups of executable runs are grouped together by them using 
        the same datestamp datedfolder name.
        So there can be more dirs that corresponding dfolds and dtimes.
        """
        while base.endswith("/"):
            base = base[:-1]
        pass

        df = cls()
        log.info("DatedFolder.find searching for date stamped folders beneath : %s " % base)

        dirs = finddir(base, df)                        # list of dated folders beneath base
        dfolds = list(set(map(os.path.basename, dirs))) # list of unique basenames of the dated folders
        dtimes = list(map(dateparser, dfolds ))         # list of datetimes

        return dirs, dfolds, dtimes 
 
    def __call__(self, path):
        """
        :param path: directory 
        :return datetime or None:   
        """
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

    
    
