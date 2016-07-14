#!/usr/bin/env python
"""

NB this is messy initial dev, superceeded by:

dd.py 
tree.py 
plot.py 


"""
from dd import *
import math
import matplotlib.pyplot as plt 
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

log = logging.getLogger(__name__)


def make_rect(bl, tr, **kwa):
    return mpatches.Rectangle( bl, tr[0]-bl[0], tr[1]-bl[1], **kwa)


class Circ(object):
    circle = None
    @classmethod
    def intersect(cls, a, b ):
        """
        http://mathworld.wolfram.com/Circle-CircleIntersection.html
        """
        R = a.radius
        r = b.radius

        xy_a = a.pos
        xy_b = b.pos

        log.debug(" A %s xy %s " % ( R, repr(xy_a)) )  
        log.debug(" B %s xy %s " % ( r, repr(xy_b)) )  

        assert(xy_b[1] == xy_a[1]) 
        d = xy_b[0] - xy_a[0]
        if d == 0:
            return None

        dd_m_rr_p_RR = d*d - r*r + R*R 

        x = dd_m_rr_p_RR/(2.*d)
        yy = (4.*d*d*R*R - dd_m_rr_p_RR*dd_m_rr_p_RR)/(4.*d*d)
        y = math.sqrt(yy)

        npos = [x + xy_a[0], -y]
        ppos = [x + xy_a[0],  y]

        return Chord(npos, ppos, a, b ) 

    def __init__(self, pos, radius, startTheta=None, deltaTheta=None, width=None):
        self.pos = pos
        self.radius = radius
        self.startTheta = startTheta
        self.deltaTheta = deltaTheta
        self.width = width 

    def theta(self, xy):
        dx = xy[0]-self.pos[0]
        dy = xy[1]-self.pos[1]
        return math.atan(dy/dx)*180./math.pi

    def as_circle(self, **kwa):
        w = self.width  
        self.circle = mpatches.Circle(self.pos,self.radius, linewidth=w, **kwa)
        return self.circle 

    def as_patch(self, **kwa):
        st = self.startTheta
        dt = self.deltaTheta
        w = self.width  

        if st is None and dt is None:
            return [mpatches.Circle(self.pos,self.radius, linewidth=w, **kwa)]
        elif st is None and dt is not None:
            log.info("wedge %s : %s " % (-dt, dt))
            return [mpatches.Wedge(self.pos,self.radius,-dt, dt, width=w, **kwa)] 
        else:
            log.info("wedges %s : %s " % (st, st+dt))
            log.info("wedges %s : %s " % (-st-dt, -st))
            return [mpatches.Wedge(self.pos,self.radius, st, st+dt, width=w, **kwa),
                    mpatches.Wedge(self.pos,self.radius, -st-dt, -st, width=w, **kwa)] 
 

class Bbox(object):
    def __init__(self, bl, tr, art):
         self.bl = bl
         self.tr = tr
         self.art = art

    width  = property(lambda self:self.tr[0] - self.bl[0])
    height = property(lambda self:self.tr[1] - self.bl[1])

    def as_rect(self):
         kwa = {}
         kwa['facecolor'] = "none"
         kwa['edgecolor'] = "b"
         rect = mpatches.Rectangle( self.bl, self.tr[0]-self.bl[0], self.tr[1]-self.bl[1], **kwa)
         self.art.mybbox = rect
         return rect

    def __repr__(self):
         return "Bbox   %s %s width:%s height:%s" % (repr(self.bl), repr(self.tr), self.width, self.height)




class Chord(object):
    """
    Common Chord of two intersecting circles
    """
    def __init__(self, npos, ppos, a, b ):
        self.npos = npos
        self.ppos = ppos
        self.a = a 
        self.b = b  
        nx, ny = self.npos
        px, py = self.ppos
        assert nx == px  # vertical chord    
        self.x = nx 
        

    def as_lin(self): 
        return Lin(self.npos, self.ppos)

    def get_circ(self, c):
        nTheta = c.theta(self.npos) 
        pTheta = c.theta(self.ppos) 
        if nTheta > pTheta:
           sTheta, dTheta = pTheta, nTheta - pTheta
        else: 
           sTheta, dTheta = nTheta, pTheta - nTheta
        return Circ(c.pos, c.radius, sTheta, dTheta)

    def get_right_bbox(self, c):
        cr = c.radius
        cx, cy = c.pos
        nx, ny = self.npos
        px, py = self.ppos
        assert nx == px  # vertical chord    

        if nx > cx:  
            """
            chord x to right of the center  
            right_bbox is limited by the chord on left, circle on right
            """
            tr = (cx+cr, py)        
            bl = (nx, ny)
        else:      
            """    
            chord to the left of center line
            so right_bbox limited by circle on right
            """
            tr = (cx+cr,cy+cr)
            bl = (nx,cy-cr)
        pass
        return Bbox(bl, tr, c.circle)


    def get_left_bbox(self, c):
        cr = c.radius
        cx, cy = c.pos
        nx, ny = self.npos
        px, py = self.ppos
        assert nx == px  # vertical chord    

        if nx > cx:  
            """
            chord x to right of the center  
            """
            tr = (nx, cy+cr)        
            bl = (cx-cr, cy-cr)
        else:      
            """    
            chord to the left of center line
            """
            tr = (px,py)
            bl = (cx-cr,ny)
        pass
        return Bbox(bl, tr, c.circle)


    def get_circ_a(self):
        return self.get_circ(self.a) 
    def get_circ_b(self):
        return self.get_circ(self.b) 
    def get_right_bbox_a(self):
        return self.get_right_bbox(self.a)
    def get_right_bbox_b(self):
        return self.get_right_bbox(self.b)
    def get_left_bbox_a(self):
        return self.get_left_bbox(self.a)
    def get_left_bbox_b(self):
        return self.get_left_bbox(self.b)



class Lin(object):
    def __init__(self, a, b ):
        self.a = a
        self.b = b 

    def as_line(self, lw=0.1, alpha=0.5):
        lx = [self.a[0], self.b[0]]
        ly = [self.a[1], self.b[1]]
        line = mlines.Line2D( lx, ly, lw=lw, alpha=alpha)
        return line


class Tub(object):
    def __init__(self, pos, radius, sizeZ):
        self.pos = pos
        self.radius = radius
        self.sizeZ = sizeZ

    def as_patch(self, **kwa):
        botleft = [ self.pos[0], self.pos[1] - self.radius ]
        width = self.sizeZ
        height = 2.*self.radius
        log.info("rect %s %s %s " % (botleft, width, height ))
        return [mpatches.Rectangle(botleft, width, height, **kwa)]


class RevPlot(object):
    edgecolor = ['r','g','b']

    def find_bounds(self, revs):
        zmin = 1e6
        zmax = -1e6
        ymin = 1e6
        ymax = -1e6
        for i,rev in enumerate(self.revs):
            xyz,r,sz = rev.xyz, rev.radius, rev.sizeZ
            z = xyz[2]
            pos = (z,0.)

            if rev.typ == "Sphere":
                if z-r < zmin:
                    zmin = z-r*1.1
                if z+r > zmax:
                    zmax = z+r*1.1
                if -r < ymin:
                    ymin = -r
                if r > ymax:
                    ymax = r

            elif rev.typ == "Tubs": 
                pass
            pass

        self.xlim = [zmin, zmax]
        self.ylim = [ymin, ymax]

    def set_bounds(self):
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
   
    def convert(self, revs):
        kwa = {}
        for i,rev in enumerate(revs):
 
            kwa['edgecolor'] = self.edgecolor[i%len(self.edgecolor)]
            kwa['facecolor'] = 'none'
            kwa['alpha'] = 0.5 

            xyz,r,sz = rev.xyz, rev.radius, rev.sizeZ
            z = xyz[2]
            pos = (z,0.)

            if rev.typ == "Sphere":
                circ = Circ( pos,r,  rev.startTheta, rev.deltaTheta, rev.width) 
                self.circs.append(circ)
                self.patches.extend(circ.as_patch(**kwa))
            elif rev.typ == "Tubs":
                tub = Tub( pos, r, rev.sizeZ ) 
                self.patches.extend(tub.as_patch(**kwa))   
            pass
        pass

    def triplet(self):
        c1 = self.circs[0]
        c2 = self.circs[1]
        c3 = self.circs[2]

        self.patches.append(c1.as_circle(fc='none',ec='r'))
        self.patches.append(c2.as_circle(fc='none',ec='g'))
        self.patches.append(c3.as_circle(fc='none',ec='b'))

        h12 = Circ.intersect(c1,c2)
        self.lins.append(h12.as_lin())

        h23 = Circ.intersect(c2,c3)
        self.lins.append(h23.as_lin())

        rbb_c2 = h23.get_right_bbox_a()
        rbb_c3 = h23.get_right_bbox_b()
        if rbb_c2.width < rbb_c3.width:
            self.patches.append(rbb_c2.as_rect())
        else:
            self.patches.append(rbb_c3.as_rect())

        lbb_c2 = h23.get_left_bbox_a()
        lbb_c3 = h23.get_left_bbox_b()
        if lbb_c2.width < lbb_c3.width:
            self.patches.append(lbb_c2.as_rect())
        else:
            self.patches.append(lbb_c3.as_rect())



    def __init__(self, ax, revs, clip=True):     
        self.ax = ax
        self.revs = revs
        self.clip = clip
        self.find_bounds(revs)
        self.set_bounds()

        self.circs = []
        self.rects = []
        self.lins = []
        self.patches = []

        self.convert(revs)

        #if len(self.circs) == 3:
        #   self.triplet()


        # for each slice of z between the chords
        # need to identify the relevant single primitive
        # and just collect those
        for p in self.patches:
            ax.add_artist(p)

        for i in range(len(self.lins)):
            a = 0.1 if i == 1 else 0.9
            ax.add_line(self.lins[i].as_line(alpha=a))

        if clip:
            for a in ax.findobj(lambda a:hasattr(a, 'mybbox')):
                log.info("clipping %s " % repr(a))
                a.set_clip_path(a.mybbox)


def split_plot(g, lvns):
    lvns = lvns.split()
    xlim = None 
    ylim = None 
    nplt = len(lvns)
    if nplt == 1:
        nx, ny = 1, 1
    else:
        nx, ny = 2, 2

    fig = plt.figure()
    for i in range(nplt):
        lv = g.logvol_(lvns[i])
        revs = lv.allrev()
        print "\n".join(map(str,revs))

        ax = fig.add_subplot(nx,ny,i+1, aspect='equal')
        ax.set_title(lv.name)

        rp = RevPlot(ax,revs)

        if xlim is None:
            xlim = rp.xlim 
        else:
            ax.set_xlim(*xlim) 

        if ylim is None:
           ylim = rp.ylim 
        else:
            ax.set_ylim(*ylim) 

    pass 
    fig.show()
    fig.savefig("/tmp/pmt_split_plot.png")



def single_plot(g, lvns_, maxdepth=10, clip=True):
    lvns = lvns_.split()
    revs = []
    for i in range(len(lvns)):
        lv = g.logvol_(lvns[i])
        revs += lv.allrev(maxdepth=maxdepth)
        print "\n".join(map(str,revs))

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_title(lvns_)
    rp = RevPlot(ax,revs, clip=clip)



    fig.show()
    fig.savefig("/tmp/pmt_single_plot.png")

    return ax



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    g = Dddb.parse("$PMT_DIR/hemi-pmt.xml")
    g.dump_context('PmtHemi')

    #lvns = "lvPmtHemi lvPmtHemiVacuum lvPmtHemiCathode"
    lvns = "lvPmtHemi"
    #lvns = "lvPmtHemiVacuum"
    #lvns = "lvPmtHemiCathode"
    #lvns = "lvPmtHemiBottom"
    #lvns = "lvPmtHemiDynode"

    #split_plot(g, lvns)
    ax = single_plot(g, lvns, maxdepth=2, clip=True)




