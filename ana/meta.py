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
meta.py
========

Attempt at more general metadata handling than metadata.py.
for example read the names of the files in the dir do not 
assume certain ones are there, just read them according to extensions.

"""
from datetime import datetime
import os, re, logging, sys
import numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import ini_, json_, splitlines_
from opticks.ana.datedfolder import DatedFolder, dateparser

class Meta(object):
    def __init__(self, path, base):
        """
        :param path: path of dated folder relative to base directory 
        :param base: directory 

        Reads json and ini files from the directory identified
        ito the d dict keyed on file stem names.
        """
        self.path = path
        self.base = base
         

        absdir = os.path.join(base, path) 
        self.absdir = absdir 

        self.datefold = os.path.basename(path)
        self.timestamp = dateparser(self.datefold)
        self.parentfold = os.path.basename(os.path.dirname(path))        
        assert self.timestamp is not None, "expected datedfolder argument %s " % path 

        self.d = {}
        for n in os.listdir(absdir):
            p = os.path.join(absdir,n)
            if not os.path.isfile(p): continue
            stem,ext = os.path.splitext(n) 
            if ext == ".json":
                self.d[stem] = json_(p)
            elif ext == ".ini":
                self.d[stem] = ini_(p)
            else:
                pass
            pass
        pass


    def __getitem__(self, kspec):
        """
        :param kspec: eg parameters.NumPhotons or OpticksEvent_launch.launch001
        :return value: 

        Provides values from metadata dicts read from files such as parameters.json 
        or OpticksEvent_launch.ini from the dated folder.
        """
        elem = kspec.split(".")
        assert len(elem) == 2
        top, key = elem 
        assert top in self.d, (top, d.keys())
        p = self.d[top]
        return p.get(key, -1)

    def __repr__(self):
        return "%50s : %30s : %20s : %s " % ( self.path, self.parentfold, self.timestamp, repr(self.d.keys()) )


if __name__ == '__main__':
    base = sys.argv[1] if len(sys.argv) > 1 else "." 
    dirs, dfolds, dtimes = DatedFolder.find(base)
    assert len(dfolds) == len(dtimes)

    for p in dirs:
        m = Meta(p, base)
        print(m)
    pass

    
    
