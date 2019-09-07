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
import numpy as np, math 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse, PathPatch


def make_rect(xy , wh, **kwa ):
    """
    :param xy: center of rectangle
    :param wh: halfwidth, halfheight
    """
    ll = ( xy[0] - wh[0], xy[1] - wh[1] )
    return Rectangle( ll,  2.*wh[0], 2.*wh[1], **kwa  )

def rep0(ax, sz, N, x):
    for i in range(N):
        r = make_rect( [x,(i-N/2)], sz )
        ax.add_patch(r) 
    pass

def rep1(ax, sz, N, x):
    r = make_rect( [x,0], [sz[0],N/2] )
    ax.add_patch(r) 
    pass



def try0(ax):


    #gpu = False
    gpu = True

    sy = 4000 if gpu else 10  
    sx = sy

    ax.set_ylim([-sy,sy])
    ax.set_xlim([-sx,sx])

    xx = [-sx*0.9,-sx*0.5,0]
    cc = [1,8,16]

    if gpu:
        xx.append(sx*0.5)
        cc.append(5120)
    pass

    sz = [sx*0.1,sx*0.1]

    for i in range(len(xx)):
        rep0(ax, sz, cc[i], xx[i])
    pass


def try1(ax):

    r = make_rect( [-9,0], [1,1] )
    ax.add_patch(r)   

    r = make_rect( [-5,0], [1,4] )
    ax.add_patch(r)   

    r = make_rect( [ 0,0], [1,8] )
    ax.add_patch(r)   

    r = make_rect( [ 5,0], [1,5120] )
    ax.add_patch(r)   


def try2(ax):


    # sy = 3000
    #sx = 10 

    ax.set_ylim([0,6000])
    ax.set_xlim([0,16])


    #ax.set_yscale('log')

    x = np.arange(0,13)
    y = np.power(2,x)

    plt.plot( x, y , drawstyle="steps-mid")
    
    plt.plot( [0,14], [5120,5120] )
    plt.annotate( "5120 CUDA cores", xy=[0,5120+100] )

    plt.plot( [x[4]], [y[4]], marker="*") 

    plt.plot( [0,14], [16,16] )
    plt.annotate( "16 CPU cores", xy=[0,16+100] )




if __name__ == "__main__":

    plt.ion()
    fig = plt.figure(figsize=(6,5.5))
    plt.title("\"Shape\" of CPU vs GPU ")

    ax = fig.add_subplot(111)

    s = 1.4

    v0 = 5120.
    qv0 = np.sqrt(v0)
    ax.set_ylim([-s*qv0, s*qv0])
    ax.set_xlim([-s*qv0, s*qv0])


    c0 = Circle( [0,0], radius=qv0 )
    ax.add_patch(c0)
    plt.annotate( "Area ~ 5120", xy=[0,0], horizontalalignment='center', fontsize=16 )

    v1 = 16 
    qv1 = np.sqrt(v1)
    c1 = Circle( [0, qv0+10], radius=qv1 )
    ax.add_patch(c1)
    plt.annotate( "Area ~ 16", xy=[0, qv0+15], horizontalalignment='center', fontsize=16 )


    v2 = 1 
    qv2 = np.sqrt(v2)
    c2 = Circle( [0, -(qv0+10)], radius=qv2 )
    ax.add_patch(c2)
    plt.annotate( "Area ~ 1", xy=[0, -(qv0+20)], horizontalalignment='center', fontsize=16 )





    fig.show()

