#!/usr/bin/env python

import numpy as np
from opticks.ana.nbase import count_unique
from node import Node
from intersect import Ray, IIS

_ctrl_color = {
       0:'c', 
       1:'k', 
       2:'m', 
       3:'b', 
       4:'r', 
       5:'y',
       6:'r'
}


X,Y,Z,W = 0,1,2,3

def one_line(ax, a, b, c ):
    x1 = a[X]
    y1 = a[Y]
    x2 = b[X]
    y2 = b[Y]
    ax.plot( [x1,x2], [y1,y2], c ) 


class T(object):
    def __init__(self, root, debug=[], skip=[], notes="", source=None, scale=None, sign=None, num=200, level=1, **kwa):
        """
        :param root:
        :param name:
        :param debug: list of ray index for dumping
        """

        if source is None:
            source = "leaflight"
        if scale is None:
            scale = 3.
        if sign is None:
            sign = -1.

        root.annotate()

        self.root = root
        self.name = root.name
        self.debug = debug
        self.skip = skip
        self.notes = notes

        self.source = source
        self.scale = scale
        self.sign = sign
        self.num = num


        self.level = level
        self.kwa = kwa

        self.prob = []
        self.ierr = 0 
        self.rerr = 0 

        self._rays = None
        self._i = None

    
    icu = property(lambda self:count_unique(self.i.seq[0]))
    rcu = property(lambda self:count_unique(self.i.seq[1]))
   
    def _get_suptitle(self):
        #icu_ = desc_ctrl_cu(self.icu, label="ITERATIVE")
        #rcu_ = desc_ctrl_cu(self.rcu, label="RECURSIVE")
        icu_ = repr(self.icu)
        rcu_ = repr(self.rcu)
        smry = "%10s IR-mismatch-rays %d/%d  ierr:%d rerr:%d " % (self.name, len(self.prob),self.nray, self.ierr, self.rerr)
        return "\n".join([repr(self.root),smry,icu_, rcu_])
    suptitle = property(_get_suptitle)

    def _make_rays(self):
        """
        Should collect rays in ndarray not a list 
        """
        rays = []
        if "xray" in self.source:
            rays += [Ray(origin=[0,0,0], direction=[1,0,0])]
        pass

        if "leaflight" in self.source:
            for leaf in Node.inorder_r(self.root, nodes=[], leaf=True, internal=False):
                rays += Ray.leaflight(leaf, num=self.num, sign=self.sign, scale=self.scale)
            pass

        if "aringlight" in self.source:
            ary = Ray.aringlight(num=self.num, radius=1000)
            rays += Ray.make_rays(ary)
        pass

        if "origlight" in self.source:
            rays += Ray.origlight(num=self.num)
        pass

        if "lsquad" in self.source:
            rays += [Ray(origin=[-300,y,0], direction=[1,0,0]) for y in range(-1000,1000+1,10)]
        pass

        if "rsquad" in self.source:
            rays += [Ray(origin=[300,y,0], direction=[-1,0,0]) for y in range(-1000,1000+1,10)]
        pass

        if "qray" in self.source:
            s = 300
            r = range(-s,s+1,2)
            rays += [Ray(origin=[s,y,0], direction=[-1,0,0])  for y in r]
            rays += [Ray(origin=[-s,y,0], direction=[1,0,0])  for y in r]
            rays += [Ray(origin=[x,-s,0], direction=[0,1,0])  for x in r]
            rays += [Ray(origin=[x, s,0], direction=[0,-1,0]) for x in r]
        pass
 
        if "seray" in self.source:
            s = 300
            r = range(-s,s+1,2)
            rays += [Ray(origin=[v,v-s,0], direction=[-1,1,0]) for v in r]
        pass
        return rays

    def _get_rays(self):
        if self._rays is None:
            self._rays = self._make_rays()
        pass
        return self._rays
    rays = property(_get_rays)

    nray = property(lambda self:len(self.rays))

    def _make_i(self):
        a = np.zeros((2,self.nray,4,4), dtype=np.float32 )
        i = IIS(a)
        i._ctrl_color = _ctrl_color
        return i 

    def _get_i(self):
        if self._i is None:
            self._i = self._make_i()
        pass
        return self._i 
    i = property(_get_i)


    def run(self, isectors_=[None,None], rr=[1,0]):
        """
        :param isectors_: array of callables that take ray arguments and returns an intersect
        """
        for iray, ray in enumerate(self.rays):
            for r in rr:
                intersect_ = isectors_[r]      
                if intersect_ is not None:
                    self.i[r, iray] = intersect_(ray) 
                pass
            pass
        pass


    def compare_intersects(self, csg, rr=[1,0]):
        csg.tst = self
        tst = self

        for iray, ray in enumerate(tst.rays):
            for r in rr:
                csg.reset(ray=ray, iray=iray, debug=tst.debug) 
                if r:
                    csg.alg = 'R'
                    isect = csg.recursive_intersect_2(tst.root) 
                else: 
                    csg.alg = 'I'
                    isect = csg.iterative_intersect(tst.root)
                pass
                tst.i[r,iray] = isect 
            pass

            t = tst.i.t[:,iray]  # intersect t 
            ok_t = np.allclose( t[0], t[1] )

            n = tst.i.n[:,iray]  # intersect normal
            ok_n = np.allclose( n[0], n[1] )

            o = tst.i.o[:,iray] # ray.origin
            ok_o = np.allclose( o[0], o[1] )

            d = tst.i.d[:,iray] # ray.direction
            ok_d = np.allclose( d[0], d[1] )

            p = tst.i.ipos[:,iray]
            ok_p = np.allclose( p[0], p[1] )

            ## hmm could compre all at once both within intersect 

            if not (ok_p and ok_n and ok_t and ok_d and ok_o):
                tst.prob.append(iray)
            pass
        pass

    def plot_intersects(self, axs, normal=None, origin=None, rayline=None, raytext=False, rr=None):
        """
        None args yielding defaults handy for debugging as 
        can comment changes to None param values in caller without 
        needing to know what defaults are 
        """
        if normal is None: normal = self.kwa.get("normal",False)
        if origin is None: origin = self.kwa.get("origin",False)
        if rayline is None: rayline = self.kwa.get("rayline",False)
        if raytext is None: raytext = self.kwa.get("raytext",False)
        if rr is None: rr = self.kwa.get("rr",[1,0])

        sc = 30 

        i = self.i
        t = i.t
        q = i.seq
        n = i.n
        o = i.o
        d = i.d
        p = i.ipos
        c = i.cseq

        m = o + 100*d  # miss endpoint 

        pr = self.prob

        for r in rr:
            ax = axs[r]

            # markers for ray origins
            if origin:
                ax.scatter( o[r][:,X] , o[r][:,Y], c=c[r], marker='x')

            mis = t[r] == 0
            sel = t[r] > 0

            if rayline:
                # short dashed lines representing miss rays
                for _ in np.where(mis)[0]:
                    one_line( ax, o[r,_], m[r,_], c[r][_]+'--' )
                pass

                # lines from origin to intersect for hitters 
                for _ in np.where(sel)[0]:
                    one_line( ax, o[r,_], p[r,_], c[r][_]+'-' )
                    if raytext:
                        if _ % 2 == 0:
                            ax.text( o[r,_,X]*1.1, o[r,_,Y]*1.1, _, horizontalalignment='center', verticalalignment='center' )
                            ax.text( p[r,_,X]*1.1, p[r,_,Y]*1.1, _, horizontalalignment='center', verticalalignment='center' )
                        pass
                    pass
                pass
            pass

            # dots for intersects
            ax.scatter( p[r][sel,X] , p[r][sel,Y], c=c[r], marker='D' )

            # lines from intersect in normal direction scaled with sc 
            if normal:
                for _ in np.where(sel)[0]:
                    one_line( ax, p[r,_], p[r,_] + n[r,_]*sc, c[r][_]+'-' )
                pass
            pass

            if len(pr) > 0:
                sel = pr
                ax.scatter( p[r][sel,X] , p[r][sel,Y], c=c[r][sel] )
            pass
            #sel = slice(0,None)

        pass


if __name__ == '__main__':
    from node import lrsph_u
    root = lrsph_u

    t = T(lrsph_u)

 
