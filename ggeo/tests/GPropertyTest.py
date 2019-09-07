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
Usage::

    ./GPropertyTest.sh


http://stackoverflow.com/questions/9497524/displaying-3-histograms-on-1-axis-in-a-legible-way-matplotlib

"""
import numpy as np
import matplotlib.pyplot as plt


psample = 1/np.load("/tmp/psample.npy") 

isample = np.load("/tmp/isample.npy") 

#isample = np.load("/tmp/insitu.npy") 
photons = np.load("/tmp/photons.npy") 




params = dict(bins=100, 
              range=(0, 900), 
              normed=True, 
              log=True, histtype='step')

plt.hist(psample, label='p', **params)

#plt.hist(isample, label='i', **params)
plt.hist(photons[:,0,3], label='i', **params)



plt.show()


