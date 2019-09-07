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



::

    
    In [19]: M = np.random.random((10, 4, 4))

    In [20]: M
    Out[20]: 
    array([[[0.26  , 0.8375, 0.65  , 0.9379],
            [0.425 , 0.5007, 0.0893, 0.9828],
            [0.7195, 0.5231, 0.0094, 0.8324],
            [0.7935, 0.9463, 0.4482, 0.071 ]],

           ...

           [[0.4159, 0.5709, 0.0778, 0.8898],
            [0.7658, 0.8104, 0.5436, 0.6296],
            [0.7726, 0.5003, 0.7588, 0.5328],
            [0.3231, 0.4282, 0.5839, 0.8149]]])

    In [21]: M.max(axis=(1,2))
    Out[21]: array([0.9828, 0.9816, 0.9906, 0.9307, 0.7959, 0.9424, 0.9273, 0.9979, 0.9815, 0.8898])

    In [22]: M.max(axis=(1,2)).shape
    Out[22]: (10,)



    In [24]: np.amax?

    Signature: np.amax(a, axis=None, out=None, keepdims=<class 'numpy._globals._NoValue'>)
    Docstring:
    Return the maximum of an array or maximum along an axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate.  By default, flattened input is
        used.

        .. versionadded:: 1.7.0

        If this is a tuple of ints, the maximum is selected over multiple axes,
        instead of a single axis or all the axes as before.




* https://jakevdp.github.io/PythonDataScienceHandbook/02.04-computation-on-arrays-aggregates.html

The way the axis is specified here can be confusing to users coming from other
languages. The axis keyword specifies the dimension of the array that will be
collapsed, rather than the dimension that will be returned. So specifying
axis=0 means that the first axis will be collapsed: for two-dimensional arrays,
this means that values within each column will be aggregated.



"""
import numpy as np


M = np.random.random((3, 4))





