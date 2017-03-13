#!/usr/bin/env python

import numpy as np
import logging, os
log = logging.getLogger(__name__)
from opticks.ana.nbase import count_unique
from node import Node
from intersect import IIS
from ray import Ray, RRS

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
    def __init__(self, root, seed=0, irays=[], skip=[], notes="", source=None, scale=None, sign=None, num=200, level=1, **kwa):
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

        if type(root) is Node:
            root.annotate()
            self.name = root.name
        elif type(root) is str:
            self.name = root
        pass
        self.root = root
        self.seed = seed
        self.irays = irays
        self.iray = None
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

        self.discrepant = []


        self._rays = None
        self._i = None

   

    debug = property(lambda self:self.iray in self.irays)
 
    icu = property(lambda self:count_unique(self.i.seq[0]))
    rcu = property(lambda self:count_unique(self.i.seq[1]))
   
    def _get_suptitle(self):
        #icu_ = desc_ctrl_cu(self.icu, label="ITERATIVE")
        #rcu_ = desc_ctrl_cu(self.rcu, label="RECURSIVE")
        icu_ = repr(self.icu)
        rcu_ = repr(self.rcu)
        smry = "%10s imps %s  mismatch-rays %d/%d  ierr:%d rerr:%d " % (self.name, ",".join(self.keys), len(self.prob),self.nray, self.ierr, self.rerr)
        return "\n".join([repr(self.root),smry,icu_, rcu_])
    suptitle = property(_get_suptitle)

    def make_rays(self):
        """
        Should collect rays in ndarray not a list 
        """
        rays = []
        if "leaflight" in self.source:
            for leaf in Node.inorder_r(self.root, nodes=[], leaf=True, internal=False):
                ll = Ray.leaflight(leaf, num=self.num, sign=self.sign, scale=self.scale)
                if len(ll) == 0:
                    log.warning("leaflight returned empty")
                else:
                    rays.append(ll)
                pass
            pass
            #log.info("make_rays leaflight %d " % len(rays))
        pass
        if "ringlight" in self.source:
            rl = Ray.ringlight(num=self.num, radius=1000)
            assert len(rl)
            rays.append(rl)
            #log.info("make_rays ringlight %d " % len(rays))
        pass
        if "origlight" in self.source:
            ol = Ray.origlight(num=self.num)
            assert len(ol)
            rays.append(ol)
            #log.info("make_rays origlight %d " % len(rays))
        pass
        if "xray" in self.source:
            xr = Ray.plane()
            assert len(xr)
            rays.append(xl)
        pass
        if "lsquad" in self.source:
            ls = Ray.plane(offset=[-300,0,0], yside=1000, direction=[1,0,0])
            assert len(ls)
            rays.append(ls)
        pass
        if "rsquad" in self.source:
            rs = Ray.plane(offset=[300,0,0], yside=1000, direction=[-1,0,0])
            assert len(rs)
            rays.append(rs)
        pass
        if "qray" in self.source:
            qr = Ray.qray() 
            assert len(qr)
            rays.append(qr)
        pass
        if "seray" in self.source:
            se = Ray.seray() 
            assert len(se)
            rays.append(se)
        pass
        if "randbox" in self.source:
            rb = Ray.randbox(num=self.num) 
            assert len(rb)
            rays.append(rb)
        pass

        #print rays  
        return RRS(np.vstack(rays))

    #def _get_rays(self):
    #    if self._rays is None:
    #        self._rays = self._make_rays()
    #    pass
    #    return self._rays
    #rays = property(_get_rays)
    #nray = property(lambda self:len(self.rays))

    def _make_i(self):
        a = np.zeros((3,self.nray,4,4), dtype=np.float32 )
        i = IIS(a)
        i._ctrl_color = _ctrl_color
        return i 

    def _get_i(self):
        if self._i is None:
            self._i = self._make_i()
        pass
        return self._i 
    i = property(_get_i)


    def run(self, imps, keys=["evaluative","recursive"]):
        """
        :param imps_: dict of callables that take ray arguments and returns an intersect
        :param keys: list of keys into the imps dict specifying the order to run the implementations
        """
        self.rays = {}
        self.nray = None
        self.imps = imps
        self.keys = keys

        if self.seed is not None:
            log.debug("np.random.seed %u " % self.seed)
            np.random.seed(self.seed)
        pass

        rays = self.make_rays()

        for k in range(len(self.keys)): # duplicate rays for each imp
            self.rays[k] = rays.copy()
        pass
        self.nray = len(rays)

        debugging = len(self.irays) > 0 
        irays = self.irays if debugging else range(self.nray)
        for iray in irays:
            if debugging:
                print "%5d run " % iray  
            for k,key in enumerate(self.keys):
                intersect_ = imps[key]      
                if intersect_ is not None:
                    self.iray = iray
                    self.i[k, iray] = intersect_(self.rays[k].rr(iray), self) 
                    pass
                pass
            pass
        pass

    

    def compare(self):
        r_att = 'seq o d tmin'
        r_discrep = {}
        for att in r_att.split():
            q0 = getattr(self.rays[0], att)
            q1 = getattr(self.rays[1], att)

            if att in ['seq']:
                q_ok = np.all( q0 == q1 )
            else:
                q_ok = np.allclose( q0, q1)
            pass 
            if not q_ok:
                r_discrep[att] = np.where(q0 != q1)[0]
            pass
        pass
        self.r_discrep = r_discrep


        i_att = 't n o d ipos'
        i_discrep = {}
        for att in i_att.split():
            q = getattr(self.i, att)
            q_ok = np.allclose( q[0], q[1] )
            if not q_ok:
                i_discrep[att] = np.where(q[0] != q[1])[0]   # how to do notclose ?
            pass
        pass
        self.i_discrep = i_discrep

        discrep = r_discrep.values() + i_discrep.values()
        self.discrepant = np.unique(np.hstack(discrep)) if len(discrep) > 0 else []
        if len(self.discrepant) > 0:
            log.warning("compare finds discrepancies %s " %  self.suptitle)
        pass


    is_discrepant = property(lambda self:len(self.discrepant) > 0 ) 

    def _get_suptitle(self):
        smry = "%10s IR-mismatch-rays %d/%d  discrepant:%r " % (self.name, len(self.discrepant),self.nray, self.discrepant)
        return "\n".join(filter(None,[repr(self.root),smry]))
    suptitle = property(_get_suptitle)



    def compare_intersects(self, csg, rr=[1,0]):
        """
        TODO: move this to follow the more flixible run approach
        """
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

    def plot_intersects(self, axs, normal=None, origin=None, rayline=None, raytext=False, keys=None):
        """
        None args yielding defaults handy for debugging as 
        can comment changes to None param values in caller without 
        needing to know what defaults are 
        """
        if normal is None: normal = self.kwa.get("normal",False)
        if origin is None: origin = self.kwa.get("origin",False)
        if rayline is None: rayline = self.kwa.get("rayline",False)
        if raytext is None: raytext = self.kwa.get("raytext",False)
        if keys is None: keys = self.kwa.get("keys","evaluative recursive".split())

        sc = 30 

        i = self.i


        if len(self.irays) == 0:
            t = i.t
            q = i.seq
            n = i.n
            o = i.o
            d = i.d
            p = i.ipos
            c = i.cseq
        else:
            sub = self.irays
            t = i.t[:, sub]
            q = i.seq[:, sub]
            n = i.n[:, sub]
            o = i.o[:, sub]
            d = i.d[:, sub]
            p = i.ipos[:, sub]
            c = i.cseq[:, sub]
        pass


        m = o + 100*d  # miss endpoint 

        pr = self.prob

        for r,key in enumerate(keys):
            ax = axs[key]

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

    t = T(lrsph_u, source="ringlight")
    r = t.rays[0]
    i = t.rays[1]
    
 
