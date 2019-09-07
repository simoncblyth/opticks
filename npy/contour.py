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

* https://matplotlib.org/2.0.0/examples/pylab_examples/contour_demo.html


"""
import numpy as np
import matplotlib.pyplot as plt

plt.ion()
plt.close()

delta = 0.1

x = np.arange(-4.0, 4.0, delta)
y = np.arange(-4.0, 4.0, delta)

X, Y = np.meshgrid(x, y)


Z = 2 - np.sqrt(X*X + Y*Y) 


plt.figure()



qcs = plt.contour(X, Y, Z)
plt.clabel(qcs, inline=1, fontsize=10)
plt.show()


