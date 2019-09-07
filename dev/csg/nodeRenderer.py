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
plt.rcParams["figure.max_open_warning"] = 200
import matplotlib.patches as mpatches


from opticks.dev.csg.csg import CSG



class Renderer(object):
    def __init__(self, ax, axes=[0,1]):
        self.ax = ax
        self.axes = axes

    def limits(self, sx=200, sy=150):
        self.ax.set_xlim(-sx,sx)
        self.ax.set_ylim(-sy,sy)

    colors = ['r','g','b','c','m','y','k']

    def color(self, i, other=False):
        n = len(self.colors)
        ic = (n-i-1)%n if other else i%n 
        return self.colors[ic]

    def render(self, root):
        if root.is_primitive:
            self.render_primitive(root)
        elif root.left is not None and root.right is not None:
            self.render(root.left)
            self.render(root.right)
        else:
            assert 0  
        pass


    def old_render(self, root):
        #log.info("render %r " % root )
        p = Node.leftmost(root)
        while p is not None:
            if p.is_bileaf:
                self.render_primitive(p.l)
                self.render_primitive(p.r)
            else:
                pass
                #print "render not-bileaf p : %s " % p
            pass
            p = p.next_
        pass

    def render_primitive(self, node):
        if node.typ == node.SPHERE:
            self.render_sphere(node)
        elif node.typ == node.BOX:
            self.render_box(node)
        elif node.typ == node.EMPTY:
            pass
        else:
            assert 0, "no render_primitive imp for %r " % node 

    def autocolor(self, patch, idx):
        ec = self.color(idx)
        #fc = self.color(idx, other=True)

        ec = 'b'
        fc = 'none'
        #log.info("autocolor idx %d ec %s " % (idx,ec) )
        patch.set_ec(ec)
        patch.set_fc(fc)

    def render_sphere(self,node):
        center = node.param[:3]
        radius = node.param[3] 
        #log.info("%s : render_sphere center %s radius %s " % (node.tag, repr(center), radius) )

        art = mpatches.Circle(center[self.axes],radius) 
        self.autocolor(art, 0)
        self.add_patch(art)

    def render_box(self,node):
        cen = node.param[:3]
        sid = node.param[3]
        bmin = cen - sid
        bmax = cen + sid
        dim = bmax - bmin
        width = dim[self.axes[0]]
        height = dim[self.axes[1]]
        botleft = bmin[self.axes]

        #log.info("%s : render_box cen %s sid %s " % (node.tag, repr(cen), sid) )
        art = mpatches.Rectangle( botleft, width, height)
        #self.autocolor(art, node.idx)
        self.add_patch(art)

    def add_patch(self, art):
        self.ax.add_patch(art)
    pass
pass


if __name__ == '__main__':

    root = CSG("sphere", param=[0,0,0,10])

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1, aspect='equal')
    ax2 = fig.add_subplot(1,2,2, aspect='equal')
    axs = [ax1,ax2]

    for ax in axs:
        rdr = Renderer(ax)
        #rdr.limits(400,400)
        rdr.render(root)
        ax.axis('auto') 
    pass

    fig.suptitle("suptitle", horizontalalignment='left', family='monospace', fontsize=10, x=0.1, y=0.99) 
    fig.show()




