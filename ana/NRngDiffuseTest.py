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
NRngDiffuseTest.py 
=======================

See npy-/tests/NRngDiffuseTest.cc

"""
import os, sys, logging
import numpy as np
log = logging.getLogger(__name__)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(suppress=True, precision=3)


if __name__ == '__main__':


    plt.ion()
    fig = plt.figure()


    
    d = np.load(os.path.expandvars("$TMP/NRngDiffuseTest_diffuse.npy"))
    s = np.load(os.path.expandvars("$TMP/NRngDiffuseTest_sphere.npy"))


    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(d[:,0], d[:,1], d[:,2])

     

