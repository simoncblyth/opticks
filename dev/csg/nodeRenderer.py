#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.max_open_warning"] = 200
import matplotlib.patches as mpatches

from node import Node, BOX, EMPTY, SPHERE


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
        if node.shape == SPHERE:
            self.render_sphere(node)
        elif node.shape == BOX:
            self.render_box(node)
        elif node.shape == EMPTY:
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
        self.autocolor(art, node.idx)
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
        self.autocolor(art, node.idx)
        self.add_patch(art)

    def add_patch(self, art):
        self.ax.add_patch(art)
    pass
pass


if __name__ == '__main__':
    from node import lrsph_u
    root = lrsph_u

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




