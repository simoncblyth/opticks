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
Converting a recursive into an iterative algorithm, 
using Hilbert curve algorithm as the example.

Example from Ch.9 Rod Stephens, Essential Algorithms, Wiley 

"""
import logging
log = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 18,10.2 

X,Y,Z = 0,1,2


class RelativeDraw(object):
    def __init__(self, ax):
        self.ax = ax
        self.cx = 20
        self.cy = 20
        self.stack = []

    def __call__(self, dx, dy, c='b' ):
        x1 = self.cx
        y1 = self.cy
        x2 = x1 + dx
        y2 = y1 + dy
        ax.plot( [x1,x2], [y1,y2], c ) 
        self.cx = x2
        self.cy = y2
        #print self

    def push(self, *vals ):
        self.stack.append(vals)
    def pop(self):
        return self.stack.pop()

    def __repr__(self):
        return "%6.2f %6.2f " % (self.cx, self.cy)

    def square(self):
        self(100,0)
        self(0,100)
        self(-100,0)
        self(0,-100)

    def hilbert_r(self, depth=5, dx=0, dy=10):
        # section 1
        if depth > 0:
            self.hilbert_r(depth-1, dy, dx)
        pass
        # section 2
        self(dx, dy, 'r')
        if depth > 0: 
            self.hilbert_r(depth-1, dx, dy)
        pass
        # section 3
        self(dy, dx, 'g')
        if depth > 0: 
            self.hilbert_r(depth-1, dx, dy)
        pass
        # section 4
        self(-dx, -dy, 'b')
        if depth > 0: 
            self.hilbert_r(depth-1, -dy, -dx)
        pass
        # section 5

    def hilbert_i(self, depth=5, dx=0, dy=10):
        section = 1
        while section > 0:
            if section == 1:
                section += 1
                if depth > 0: 
                    self.push(section, depth, dx, dy) # fake recursive call : hilbert_r(depth-1, dy, dx)
                    depth = depth - 1
                    tmp = dx
                    dx = dy
                    dy = tmp
                    section = 1
                pass
            elif section == 2:
                section += 1
                self(dx, dy, 'r')
                if depth > 0: 
                    self.push(section, depth, dx, dy) # fake recursive call : hilbert_r(depth-1, dx, dy)
                    depth = depth - 1
                    section = 1
                pass
            elif section == 3:
                section += 1
                self(dy, dx, 'g')
                if depth > 0: 
                    self.push(section, depth, dx, dy) # fake recursive call : hilbert_r(depth-1, dx, dy)
                    depth = depth - 1
                    section = 1
                pass
            elif section == 4:
                section += 1
                self(-dx, -dy, 'b')
                if depth > 0: 
                    self.push(section, depth, dx, dy) # fake recursive call : hilbert_r(depth-1, -dy, -dx)
                    depth = depth - 1
                    tmp = dx
                    dx = -dy
                    dy = -tmp
                    section = 1
                pass
            elif section == 5:
                # recursive returns, unwinding the stack
                if len(self.stack) > 0:
                    section, depth, dx, dy = self.pop()
                else:
                    section = -1 # no more stacks to pop
                pass
            pass
        pass          




if __name__ == '__main__':

    plt.ion()
    plt.close("all")

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, aspect='equal')

    rd = RelativeDraw(ax)

     
    ax.set_xlim(-100,200)
    ax.set_ylim(-100,200)

    #rd.hilbert_r()
    rd.hilbert_i()
   
    ax.axis('auto') 
 
    fig.show()



