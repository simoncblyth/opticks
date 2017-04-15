#!/usr/bin/env python
"""
Plotting from the serialized PMT analytic geometry data

The very thin CATHODE and BOTTOM are composed of sphere parts (sparts) 
with close inner and outer radii.  
Inner sparts have parent attribute that points to outer sparts

To plot that, need to clip away the inside.

http://matplotlib.1069221.n5.nabble.com/removing-paths-inside-polygon-td40632.html

Could also just set a width, but thats cheating and the point of the
plotting is to check the parts...


"""
import numpy as np
import logging, os
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt 
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from opticks.ana.base import opticks_main
from opticks.sysrap.OpticksCSG import CSG_

#TYPECODE = {'Sphere':1, 'Tubs':2, 'Box':3 }  ## equivalent to pre-unified hardcoded and duplicitous approach 
TYPECODE = {'Sphere':CSG_.SPHERE, 'Tubs':CSG_.TUBS, 'Box':CSG_.BOX }


path_ = lambda _:os.path.expandvars("$IDPATH/GMergedMesh/1/%s.npy" % _)


X = 0
Y = 1
Z = 2

ZX = [Z,X]
ZY = [Z,Y]
XY = [X,Y]


class Mesh(object):
    def __init__(self):
        self.v = np.load(path_("vertices"))
        self.f = np.load(path_("indices"))
        self.i = np.load(path_("nodeinfo"))
        self.vc = np.zeros( self.i.shape[0]+1 )
        np.cumsum(self.i[:,1], out=self.vc[1:])
    def verts(self, solid):
        return self.v[self.vc[solid]:self.vc[solid+1]]

class Sphere(object):
    def __init__(self, center, radius):
        self.center = center 
        self.radius = radius 
    def __repr__(self):
        return "Sphere %s %s  " % (repr(self.center), self.radius)

    def as_patch(self, axes):
        circle = mpatches.Circle(self.center[axes],self.radius)
        return circle 

class ZTubs(object):
    def __init__(self, position, radius, sizeZ):
        self.position = position
        self.radius = radius 
        self.sizeZ = sizeZ 

    def __repr__(self):
        return "ZTubs pos %s rad %s sizeZ %s " % (repr(self.position), self.radius, self.sizeZ)

    def as_patch(self, axes):
        if Z == axes[0]:
            width = self.sizeZ
            height = 2.*self.radius
            botleft = self.position[axes] - np.array([self.sizeZ/2., self.radius])
            patch = mpatches.Rectangle(botleft, width, height)
        elif Z == axes[1]:
            assert 0
        else:
            patch = mpatches.Circle(self.position[axes],self.radius)
        return patch


class Bbox(object):
    def __init__(self, min_, max_ ):
        self.min_ = np.array(min_)
        self.max_ = np.array(max_)
        self.dim  = max_ - min_

    def as_patch(self, axes):
         width = self.dim[axes[0]]
         height = self.dim[axes[1]]
         botleft = self.min_[axes]
         rect = mpatches.Rectangle( botleft, width, height)
         return rect

    def __repr__(self):
        return "Bbox %s %s %s " % (self.min_, self.max_, self.dim )


class Pmt(object):
    def __init__(self, path):
        path = os.path.expandvars(path)
        log.info("loading Pmt from %s " % path)
        self.data = np.load(path).reshape(-1,4,4)
        self.num_parts = len(self.data)
        self.all_parts = range(self.num_parts)
        self.partcode = self.data[:,2,3].view(np.int32)
        self.partnode = self.data[:,3,3].view(np.int32)
        self.index    = self.data[:,1,1].view(np.int32)
        self.parent   = self.data[:,1,2].view(np.int32)
        self.flags    = self.data[:,1,3].view(np.int32)

    def parts(self, solid):
        """
        :param solid: index of node/solid 
        :return parts array:
        """
        pts = np.arange(len(self.partnode))
        if solid is not None:
            pts = pts[self.partnode == solid]
        pass
        return pts

    def bbox(self, p):
        part = self.data[p]
        return Bbox(part[2,:3], part[3,:3])

    def shape(self, p):
        """
        :param p: part index
        :return shape instance: Sphere or ZTubs 
        """
        code = self.partcode[p]
        if code == TYPECODE['Sphere']:
            return self.sphere(p)
        elif code == TYPECODE['Tubs']:
            return self.ztubs(p)
        else:
            log.warning("Pmt.shape typecode %d not recognized, perhaps an old pre-enum-unification .npy ?" % code )
            return None 

    def sphere(self, p):
        """
        Creates *Shape* instance from Part data identified by index

        :param p: part index
        :return Sphere:
        """
        part = self.data[p]
        sp = Sphere( part[0][:3], part[0][3])

        log.debug("p %2d sp %s " % (p, repr(sp)))
      
        #if p == 10:
        #    sp.center = np.array([ 0.,  0.,  -69.], dtype=np.float32)

        return sp

    def ztubs(self, p):
        """
        Creates *ZTubs* instance from Part data index p 
        """
        q0,q1,q2,q3 = self.data[p]
        return ZTubs( q0[:3], q0[3], q1[0])


class PmtPlot(object):
    def __init__(self, ax, pmt, axes):
        self.ax = ax
        self.axes = axes
        self.pmt = pmt
        self.patches = []
        self.ec = 'none'
        self.edgecolor = ['r','g','b','c','m','y','k']
        self.highlight = {}

    def color(self, i, other=False):
        n = len(self.edgecolor)
        idx = (n-i-1)%n if other else i%n
        return self.edgecolor[idx]
        
    def plot_bbox(self, parts=[]):
        for i,p in enumerate(parts):
            bb = self.pmt.bbox(p)
            _bb = bb.as_patch(self.axes)
            self.add_patch(_bb, self.color(i))

    def plot_shape_simple(self, parts=[]):
        for i,p in enumerate(parts):
            sh = self.pmt.shape(p)
            _sh = sh.as_patch(self.axes)
            self.add_patch(_sh, self.color(i))

    def plot_shape(self, parts=[], clip=True):
        log.info("plot_shape parts %r " % parts)
        for i,p in enumerate(parts):
            is_inner = self.pmt.parent[p] > 0
            bb = self.pmt.bbox(p)
            _bb = bb.as_patch(self.axes)

            ec = self.color(i)
            #ec = 'none'
            self.add_patch(_bb, ec)

            sh = self.pmt.shape(p)
            _sh = sh.as_patch(self.axes)
            ec = self.color(i,other=True)
            fc = self.highlight.get(p, 'none')

            if is_inner:
                fc = 'w'

            self.add_patch(_sh, ec, fc)
            if clip:
                _sh.set_clip_path(_bb)

    def add_patch(self, patch, ec, fc='none'):
        patch.set_fc(fc)
        patch.set_ec(ec)
        self.patches.append(patch)
        self.ax.add_artist(patch)

    def limits(self, sx=200, sy=150):
        self.ax.set_xlim(-sx,sx)
        self.ax.set_ylim(-sy,sy)





def mug_plot(fig, pmt, pts):
    for i, axes in enumerate([ZX,XY]):
        ax = fig.add_subplot(1,2,i+1, aspect='equal')
        pp = PmtPlot(ax, pmt, axes=axes) 
        pp.plot_shape(pts, clip=True)
        pp.limits()

def clipped_unclipped_plot(fig, pmt, pts):
    for i, clip in enumerate([False, True]):
        ax = fig.add_subplot(1,2,i+1, aspect='equal')
        pp = PmtPlot(ax, pmt, axes=ZX) 
        pp.plot_shape(pts, clip=clip)
        pp.limits()

def solids_plot(fig, pmt, solids=range(5)):

    if len(solids)>4:
        ny,nx = 3,2
    else:
        ny,nx = 2,2

    for i,solid in enumerate(solids):
        pts = pmt.parts(solid)
        ax = fig.add_subplot(nx,ny,i+1, aspect='equal')
        pp = PmtPlot(ax, pmt, axes=ZX) 
        pp.plot_shape(pts, clip=True)
        pp.limits()
    pass


def one_plot(fig, pmt, pts, clip=True, axes=ZX, highlight={}):
    ax = fig.add_subplot(1,1,1, aspect='equal')
    pp = PmtPlot(ax, pmt, axes=axes) 
    pp.highlight = highlight
    pp.plot_shape(pts, clip=clip)
    pp.limits()



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = opticks_main(apmtidx=2)
    apmtpath = args.apmtpath


    # 0:4  PYREX
    # 4:8  VACUUM
    # 8:12 CATHODE
    # 12:14 BOTTOM
    # 14:15 DYNODE

    highlight = {}
    highlight[8] = 'r'
    highlight[9] = 'r'
    highlight[10] = 'r'
    highlight[11] = 'b'

    ALL, PYREX, VACUUM, CATHODE, BOTTOM, DYNODE = None,0,1,2,3,4 

    mesh = Mesh()

    pmt = Pmt(apmtpath)

    fig = plt.figure()

    axes = ZX

    #solid = CATHODE 
    #solid = BOTTOM 
    #solid = DYNODE 
    solid = ALL

    pts = pmt.parts(solid)


    #pts = np.arange(8)

    #mug_plot(fig, pmt, pts)
    #clipped_unclipped_plot(fig, pmt, pts)

    #one_plot(fig, pmt, pts, highlight=highlight)
    one_plot(fig, pmt, pts, axes=axes, clip=True)

    # hmm not possible to split at part level, as those are sub solid
    #if mesh:
    #    vv = mesh.verts(solid)
    #    plt.scatter(vv[:,axes[0]],vv[:,axes[1]],c=vv[:,Y])


    #solids_plot(fig, pmt, solids=range(5))


    fig.show()
    fig.savefig(os.path.expandvars("$TMP/pmtplot.png"))

