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


import numpy as np
import matplotlib.pyplot as plt


X,Y,Z,W = 0,1,2,3

def one_line(ax, a, b, c ):
    x1 = a[X]
    y1 = a[Y]
    x2 = b[X]
    y2 = b[Y]
    ax.plot( [x1,x2], [y1,y2], c ) 




if __name__ == '__main__':

    plt.ion()
    plt.close()


    fig = plt.figure()
    ax = fig.add_subplot(111)


    a = np.linspace(0, 2*np.pi, 7 )
    s = np.sin(a)
    c = np.cos(a)

    n = 2 


    ax.plot( c, s)
    ax.plot( n*c, n*s)

    for i in range(7):
        one_line(ax, [0,0], [n*c[i], n*s[i]], "-" )

    i = 1
    one_line(ax, [1,0], [1+(n-1)*c[i], (n-1)*s[i]], "-" )


    plt.show()


