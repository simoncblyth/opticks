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
level.py
========================================
"""

import os, sys, logging
from opticks.ana.log import fatal_, error_, warning_, info_, debug_
from opticks.ana.log import underline_, blink_ 
log = logging.getLogger(__name__)

class Level(object):
    FATAL = 20
    ERROR = 10
    WARNING = 0 
    INFO = -10
    DEBUG = -20

    MISSING = -30   ## expedient whilst debugging missing levels

    level2name = { FATAL:"FATAL", ERROR:"ERROR", WARNING:"WARNING", INFO:"INFO", DEBUG:"DEBUG", MISSING:"MISSING" }
    name2level = { "FATAL":FATAL, "ERROR":ERROR, "WARNING":WARNING, "INFO":INFO, "DEBUG":DEBUG, "MISSING":MISSING  }
    level2func = { FATAL:fatal_, ERROR:error_, WARNING:warning_, INFO:info_, DEBUG:debug_, MISSING:fatal_ }


    @classmethod
    def FromName(cls, name):
        if name is None:
            name = "MISSING"
        pass     
        level = cls.name2level[name] 
        return cls(name, level) 
    @classmethod
    def FromLevel(cls, level):
        if level is None:
            level = MISSING
        pass
        name = cls.level2name[level] 
        return cls(name, level) 

    def __init__(self, name, level):
        self.name = name
        self.nam = "_%s_" % name if name == "FATAL" else name    # hack for alignment in %16s columns
        self.level = level
        self.fn_ = self.level2func[level]


if __name__ == '__main__':
    pass
    for nam in "FATAL ERROR WARNING INFO DEBUG".split():
        lev = Level.FromName(nam)
        if nam == "FATAL": nam = "_" + nam + "_"   # the undescores are a hack that succeeds to get columns to align 
        fmt = " %4d : %8s : %16s : %4d "  
        print(fmt % ( lev.level, lev.name, lev.fn_(nam), lev.level ))  

    # for levels other than fatal , just padding by 9 extra chars gets things to line up 

    for nam in "FATAL ERROR WARNING INFO DEBUG".split():
        lev = Level.FromName(nam)

        if nam is "FATAL": nam = nam + " "

        for w in range(8,30):
            fmt = " %4d : %" + str(w) + "s : "
            print( fmt % ( w, lev.fn_(nam))  ) 
 



