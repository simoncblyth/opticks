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


import os, logging
import numpy as np

from opticks.ana.nload import A
from opticks.ana.base import idp_

class CGDMLDetector(object):
    def __init__(self):
        self.gtransforms = np.load(idp_("CGDMLDetector/0/gtransforms.npy"))
        self.ltransforms = np.load(idp_("CGDMLDetector/0/ltransforms.npy"))

    def __repr__(self):
        return "\n".join([
              "gtransforms %s " % repr(self.gtransforms.shape),
              "ltransforms %s " % repr(self.ltransforms.shape)
              ])

    def getGlobalTransform(self, frame):
        return self.gtransforms[frame]

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    frame = 3153

    det = CGDMLDetector()
    print det
    mat = det.getGlobalTransform(frame)
    print "mat %s " % repr(mat)


