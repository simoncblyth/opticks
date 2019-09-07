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
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines



def make_line( p0, p1, **kwa):
    return mlines.Line2D([p0[0],p1[0]], [p0[1],p1[1]], **kwa)


class Ax(object):
    def __init__(self, ax, p):
        self.ax = ax
        self.p = p 

    def rect( self, LL, WH, kwa):
        rect = Rectangle( p[LL], p[WH,X], p[WH,Y], **kwa ) 
        self.ax.add_patch(rect)

    def _line_point( self, A, B, kwl, kwp ): 
        l = make_line( p[A], p[B], **kwl )
        self.ax.add_line(l)
        self.ax.plot( p[B,X], p[B,Y], **kwp ) 

    def line_point( self, AB, kwl, kwp ): 
        if len(AB) == 2:
            A = AB[0] 
            B = AB[1]
            self._line_point(A,B, kwl, kwp) 
        elif len(AB) == 3:
            A = AB[0] 
            B = AB[1]
            C = AB[2]
            self._line_point(A,B, kwl, kwp) 
            self._line_point(B,C, kwl, kwp) 
        else:
            assert 0
 


if __name__ == '__main__':

    plt.ion()
    fig = plt.figure()
    plt.title("to_boundary")

    _ax = fig.add_subplot(111)
    _ax.set_ylim([0,10])
    _ax.set_xlim([0,10])

    p = np.zeros((20,2), dtype=np.float32 )

    ax = Ax(_ax, p )


    X,Y = 0,1

    LL,A,B,DX,A1,M1,B1,A2,B2,WH,A3,B3,C3,M3 = 0,1,2,3,4,5,6,7,9,10,11,12,13,14


    p[LL] = (1,7)
    p[A] = (p[LL,X], p[LL,Y] - 5 )
    p[B] = (p[LL,X]+2.5, p[LL,Y] )
    p[DX] = (1,0)

    p[A1] = p[A] + p[DX]
    p[B1] = p[B] + p[DX]

    p[A2] = p[A] + 1.5*p[DX]
    p[B2] = p[B] + 1.5*p[DX]

    p[A3] = p[A] + 2*p[DX]

    p[M1] = 0.7*p[B]+p[DX]
    p[M3] = 0.4*p[B]+2*p[DX]

    p[C3] = 0.8*p[B]+5*p[DX]

    p[WH] =(8,2) 

    ax.rect( LL, WH, dict(alpha=1, fill=False) ) 

    ax.line_point( [A, B] , dict(linestyle="dashed"), dict(marker="*", color="b"))

    ax.line_point( [A1, M1], {}, dict(marker="*", color="b") )

    ax.line_point( [A2, B2], {}, dict(marker="*", color="b") )

    ax.line_point( [A3, M3, C3], {}, dict(marker="*", color="b") )


    plt.text( p[M1,X], p[M1,Y], 'Absorb' , {'ha':'left', 'va':'bottom' }, rotation=55 )
    plt.text( p[B2,X], p[B2,Y], '"Sail"' , {'ha':'left', 'va':'bottom' }, rotation=55 )
    plt.text( p[M3,X], p[M3,Y]+0.2, 'Scatter' , {'ha':'left', 'va':'bottom' }, rotation=25 )



    fig.show()



