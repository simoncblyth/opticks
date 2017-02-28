#!/usr/bin/env python
"""
For dev notes see

* iterativeTreeEvaluationFake.py 
* iterativeTreeEvaluation.py 

Supect the threaded postorder traverse 
skips primitves that are not down at bottom 
level of the tree... see eg in Renderer which
is skipping the b3 box in smb_lbox::

    U1.Union(D2.Difference(s4.s,b5.b),b3.b)    

This will also impact the intersections as same threaded 
postorder is used.


"""
import logging
log = logging.getLogger(__name__)

import numpy as np
from opticks.ana.nbase import count_unique
import matplotlib.pyplot as plt
plt.rcParams["figure.max_open_warning"] = 200

import matplotlib.patches as mpatches


#from node import UNION, INTERSECTION, DIFFERENCE, desc
from node import Node, BOX, SPHERE, EMPTY
from node import trees
from node import root0, root1, root2, root3, root4
from node import ubo, lrsph_d1, lrsph_d2, lrsph_u, lrsph_i
from node import lrbox_u, lrbox_i, lrbox_d1, lrbox_d2, lbox_ue
from node import smb, smb_lbox, smb_lbox_ue

from intersect import intersect_primitive, intersect_miss, Ray, IIS

from opticks.optixrap.cu.boolean_h import desc_state, Enter, Exit, Miss
from opticks.optixrap.cu.boolean_h import desc_op, UNION, INTERSECTION, DIFFERENCE
from opticks.optixrap.cu.boolean_h import boolean_table, boolean_decision
from opticks.optixrap.cu.boolean_h import desc_acts, act_index
from opticks.optixrap.cu.boolean_h import ReturnMiss, ReturnA, ReturnAIfCloser, ReturnAIfFarther
from opticks.optixrap.cu.boolean_h import ReturnB, ReturnBIfCloser, ReturnBIfFarther, ReturnFlipBIfCloser
from opticks.optixrap.cu.boolean_h import AdvanceAAndLoop, AdvanceAAndLoopIfCloser
from opticks.optixrap.cu.boolean_h import AdvanceBAndLoop, AdvanceBAndLoopIfCloser


from ctrl import CtrlReturnMiss, CtrlReturnLeft, CtrlReturnRight, CtrlReturnFlipRight
from ctrl import CtrlLoopLeft, CtrlLoopRight
from ctrl import desc_ctrl, ctrl_index, desc_ctrl_cu, _ctrl_color

X,Y,Z = 0,1,2

def one_line(ax, a, b, c ):
    x1 = a[X]
    y1 = a[Y]
    x2 = b[X]
    y2 = b[Y]
    ax.plot( [x1,x2], [y1,y2], c ) 



class CSG(object):
    def __init__(self, level=1, epsilon=None):
         if epsilon is None:
             epsilon = 1e-4 
         pass
         self.level = level
         self.epsilon = epsilon
         self.reset()

    def _get_debug(self):
         return self.level if self.iray in self._debug else 0
    debug = property(_get_debug)

    def reset(self, ray=None, iray=0, debug=[]):
        self.alg = '?'
        self.ray = ray 
        self.iray = iray
        self.count = 0 
        self.lname = None
        self.rname = None
        self._debug = debug

    def classify(self, tt, nn, tmin):
        if tt > tmin:
            state = Enter if np.dot(nn, self.ray.direction) < 0. else Exit 
        else:
            state = Miss 
        pass
        return state

    def binary_ctrl(self, operation, left, right, tminL=0, tminR=0):
        """
        :param operation: enum
        :param left: isect
        :param right: isect
        :param tminL: used for isect classification
        :param tminR: 
        :return ctrl: bitfield with a single bit set, 0x1 << n  

        * increasing tmin will cause some Enter/Exit intersect states to become Miss
        * when doin LoopL LoopR need to leave the other side state unchanged, hence
          using split tminL/tminR

        """
        act = 0
        stateL = self.classify(left.t, left.n, tminL)  
        stateR = self.classify(right.t, right.n, tminR)

        acts = boolean_table(operation, stateL, stateR )
        act = boolean_decision(acts, left.t, right.t )

        if act & (ReturnMiss):
            ctrl = CtrlReturnMiss
        elif act & (ReturnAIfCloser | ReturnAIfFarther | ReturnA):
            ctrl = CtrlReturnLeft 
        elif act & (ReturnBIfCloser | ReturnBIfFarther | ReturnB): 
            ctrl = CtrlReturnRight 
        elif act & (ReturnFlipBIfCloser): 
            ctrl = CtrlReturnFlipRight 
        elif act & (AdvanceAAndLoop | AdvanceAAndLoopIfCloser):
            ctrl = CtrlLoopLeft
        elif act & (AdvanceBAndLoop | AdvanceBAndLoopIfCloser):
            ctrl = CtrlLoopRight
        else: 
            assert 0, desc_acts(act)
        pass
        return ctrl

    def binary_result(self, ctrl, miss, left, right):
        """
        :param ctrl: bitfield with single bit set 
        :param miss: isect
        :param left: isect
        :param right: isec
        :return result: as picked by ctrl 
        """
        if ctrl == CtrlReturnMiss:
            result = miss[:]
        elif ctrl == CtrlReturnLeft:
            result = left[:]
        elif ctrl == CtrlReturnRight: 
            result = right[:]
        elif ctrl == CtrlReturnFlipRight: 
            result = right[:]
            result.n = -result.n
        else:
            assert 0, desc_ctrl(ctrl)
        pass
        result.seq = 0 # debugging scrub any former
        result.addseq(ctrl_index(ctrl))  ## record into ray?, as keep getting fresh II instances
        assert len(result.shape) == 2 and result.shape[0] == 4 and result.shape[1] == 4
        return result        


    def check_tree(self, root):
        lop = Node.leftmost(root)
        assert lop.is_bileaf 
        assert not lop.is_primitive
        if lop is root:
            pass
            #log.info("left-op is root, ie height 1 tree with a single operation and two leaf primitives")
        else:
            assert lop.next_ is not None, "threaded postorder requires Node.postorder_threading_r "
        pass

    def postorder_threaded_traverse(self, root):
        self.check_tree(root)
        p = Node.leftmost(root)
        while p is not end:
            print p
            p = p.next_ 
        pass

    pfx = property(lambda self:"%s%0.3d" % (self.alg, self.iray))

    def iterative_intersect(self, root, depth=0, tmin=0, debug=1):
        """
        :param root:
        :return isect: I instance
        """
        self.check_tree(root)

        lhs = []
        rhs = []

        tranche = []
        tranche.append([Node.leftmost(root),None])

        while len(tranche) > 0:
            begin, end = tranche.pop()
            log.debug("start tranche begin %s end %s " % (begin, end))
            
            p = begin
            while p is not end:

                act = 0 
                left = None
                right = None
                miss =  intersect_miss(p, self.ray, tmin)
                result = None
    
                tminL = tmin
                tminR = tmin

                ctrl = (CtrlLoopLeft | CtrlLoopRight)
                log.debug("p loop %s : %s " % (desc_ctrl(ctrl), p))

                reiterate_ = False

                loopcount = 0 
                while ctrl & (CtrlLoopLeft | CtrlLoopRight ):

                    if ctrl & CtrlLoopLeft: 
                        if p.l.is_leaf:
                            left = intersect_primitive(p.l, self.ray, tminL)
                        else:
                            try:
                                left = lhs.pop()
                            except IndexError:
                                left = miss
                                self.tst.ierr += 1
                                log.error("%s : lhs pop from empty" % (self.pfx))
                        pass
                    pass
                    if ctrl & CtrlLoopRight: 
                        if p.r.is_leaf:
                            right = intersect_primitive(p.r, self.ray, tminR)
                        else:
                            try:
                                right = rhs.pop()
                            except IndexError:
                                right = miss
                                self.tst.ierr += 1
                                log.error("%s : rhs pop from empty" % (self.pfx))
                        pass
                    pass

                    ctrl = self.binary_ctrl(p.operation, left, right, tminL, tminR)    

                    if ctrl in [CtrlReturnMiss, CtrlReturnLeft, CtrlReturnRight, CtrlReturnFlipRight]:
                        pass # will fall out of the ctrl while as no longer loopers
                    elif ctrl == CtrlLoopLeft: 
                        tminL = left.t + self.epsilon
                        if not p.l.is_leaf:
                            rhs.append(right)
                            tranche.append([p,None])  
                            tranche.append([Node.leftmost(p.l),p.l.next_])
                            reiterate_ = True 
                        pass
                    elif ctrl == CtrlLoopRight: 
                        tminR = right.t + self.epsilon
                        if not p.r.is_leaf:
                            lhs.append(left)
                            tranche.append([p,None])  
                            tranche.append([Node.leftmost(p.r),p.r.next_])
                            reiterate_ = True 
                        pass
                    else:
                         assert 0, desc_ctrl(ctrl) 
                    pass
                    if reiterate_:  # non-bileaf loopers have to break to traverse subtree
                        break 
                    pass
                    loopcount += 1 
                    assert loopcount < 10  
                    pass 
                pass  # end while ctrl loop   
          
                if reiterate_:  
                    log.info("break out of postorder traversal, for re-iteration of subtree ntranche %d " % len(tranche))
                    break 

                assert ctrl in [CtrlReturnMiss, CtrlReturnLeft,CtrlReturnRight,CtrlReturnFlipRight]
                result = self.binary_result(ctrl, miss, left, right)

                if p.is_left:
                    lhs.append(result)
                else:
                    rhs.append(result)
                pass

                p = p.next_ 
            pass               # postorder tranche traversal while loop
        pass


        #assert len(lhs) == 0, lhs
        #assert len(rhs) == 1, rhs   # end with p.idx = 1 for the root

        if len(lhs) != 0:
            self.tst.ierr += 1     
            log.error("%s : lhs ends with %d, expect 0 " % (self.pfx, len(lhs)))

        if len(rhs) != 1:
            self.tst.ierr += 1     
            log.error("%s : rhs ends with %d, expect 1 " % (self.pfx,len(rhs)))

        return rhs[0]
     

    def recursive_intersect(self, p, tmin=0):
        assert p.is_operation
        miss =  intersect_miss(p, self.ray, tmin)
        tminL = tmin
        tminR = tmin
        loopcount = 0 
        ctrl = (CtrlLoopLeft | CtrlLoopRight)

        while ctrl & (CtrlLoopLeft | CtrlLoopRight ):
            if ctrl & CtrlLoopLeft: 
                if p.l.is_leaf:
                    left = intersect_primitive(p.l, self.ray, tminL)
                else:
                    left = self.recursive_intersect(p.l, tmin=tminL)
                pass
            pass
            if ctrl & CtrlLoopRight: 
                if p.r.is_leaf:
                    right = intersect_primitive(p.r, self.ray, tminR)
                else:
                    right = self.recursive_intersect(p.r, tmin=tminR)
                pass
            pass
            ctrl = self.binary_ctrl(p.operation, left, right, tminL, tminR)    

            if ctrl == CtrlLoopLeft:
                tminL = left.t + self.epsilon
            elif ctrl == CtrlLoopRight:
                tminR = right.t + self.epsilon
            else:
                pass # will fall out the ctrl loop
            pass 
            loopcount += 1 
            assert loopcount < 10  
        pass  #  end while ctrl loop

        assert ctrl in [CtrlReturnMiss, CtrlReturnLeft, CtrlReturnRight, CtrlReturnFlipRight ]
        result = self.binary_result(ctrl, miss, left, right)
        return result
 

    @classmethod
    def trep_fmt(cls, tmin, tl, tr ):
        return "tmin/tl/tr %5.2f %5.2f %5.2f " % (tmin if tmin else -1, tl if tl else -1, tr if tr else -1 )

    def compare_intersects(self, tst, rr=[1,0]):

        self.tst = tst
        for iray, ray in enumerate(tst.rays):
            for r in rr:
                self.reset(ray=ray, iray=iray, debug=tst.debug) 
                if r:
                    self.alg = 'R'
                    isect = self.recursive_intersect(tst.root) 
                else: 
                    self.alg = 'I'
                    isect = self.iterative_intersect(tst.root)
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

    def plot_intersects(self, tst, axs, normal=None, origin=None, rayline=None, rr=[1,0]):
        """
        None args yielding defaults handy for debugging as 
        can comment changes to None param values in caller without 
        needing to know what defaults are 
        """
        if normal is None: normal = False
        if origin is None: origin = False
        if rayline is None: rayline = False

        sc = 30 

        i = tst.i
        t = i.t
        q = i.seq
        n = i.n
        o = i.o
        d = i.d
        p = i.ipos
        c = i.cseq

        m = o + 100*d  # miss endpoint 

        pr = tst.prob

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
                    if _ % 2 == 0:
                        ax.text( o[r,_,X]*1.1, o[r,_,Y]*1.1, _, horizontalalignment='center', verticalalignment='center' )
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






class T(object):
    def __init__(self, root, debug=[], skip=[], notes="", source=None, num=200, level=1):
        """
        :param root:
        :param name:
        :param debug: list of ray index for dumping
        """

        if source is None:
            source = "aringlight,origlight"
        pass

        self.root = root
        self.name = root.name
        self.debug = debug
        self.skip = skip
        self.notes = notes
        self.source = source
        self.num = num
        self.level = level

        self.prob = []
        self.ierr = 0 
        self.rerr = 0 

        self._rays = None
        self._i = None

    
    icu = property(lambda self:count_unique(self.i.seq[0]))
    rcu = property(lambda self:count_unique(self.i.seq[1]))
   
    def _get_suptitle(self):
        icu_ = desc_ctrl_cu(self.icu, label="ITERATIVE")
        rcu_ = desc_ctrl_cu(self.rcu, label="RECURSIVE")
        smry = "%10s IR-mismatch-rays %d/%d  ierr:%d rerr:%d " % (self.name, len(self.prob),self.nray, self.ierr, self.rerr)
        return "\n".join([repr(self.root),smry,icu_, rcu_])
    suptitle = property(_get_suptitle)

    def _make_rays(self):
        rays = []
        if "xray" in self.source:
            rays += [Ray(origin=[0,0,0], direction=[1,0,0])]
        pass

        if "aringlight" in self.source:
            ary = Ray.aringlight(num=self.num, radius=1000)
            rays += Ray.make_rays(ary)
        pass

        if "origlight" in self.source:
            rays += Ray.origlight(num=self.num)
        pass

        if "lsquad" in self.source:
            rays += [Ray(origin=[-300,y,0], direction=[1,0,0]) for y in range(-50,50+1,10)]
        pass

        if "rsquad" in self.source:
            rays += [Ray(origin=[300,y,0], direction=[-1,0,0]) for y in range(-50,50+1,10)]
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

    logformat = "%(asctime)s %(name)s %(levelname)-8s %(message)s"
    logging.basicConfig(level=logging.INFO,format=logformat)

    plt.rcParams['figure.figsize'] = 18,10.2 

    plt.ion()
    plt.close("all")

    source = None
    epsilon = None
    origin = None
    normal = None

    #root = trees[1]
    #root = trees[3]
    #root = trees[4]
    #root = trees[5]

    roots = trees
    #roots = [lrbox_d1, ubo]
    #roots = [lbox_ue, smb_lbox, smb_lbox_ue]
    #roots = [smb_lbox_ue]

    normal = True

    for iroot, root in enumerate(roots):
    #if 1:
        #iroot, root, source = 0, smb_lbox_ue, "origlight"  # iterative doesnt see inner box, that recursive does
        #iroot, root, source = 0, smb_lbox_ue, "aringlight"
        #iroot, root, source = 0, smb_lbox, "aringlight"


        print "%2d : %15s : %s " % (iroot, root.name, root )
        #if root.ok: continue 

        root.annotate()

        fig = plt.figure()

        ax1 = fig.add_subplot(1,2,1, aspect='equal')
        ax2 = fig.add_subplot(1,2,2, aspect='equal')
        axs = [ax1,ax2]

        tst = T(root,level=3,debug=[0], num=100, source=source)
        csg = CSG(level=tst.level, epsilon=epsilon)

        rr = [0,1]   # recursive only [1], iterative only [0], or both [0,1] [1,0]

        csg.compare_intersects( tst, rr=rr )
        csg.plot_intersects( tst, axs=axs, rr=rr, normal=normal, origin=origin)

        # seems patches cannot be shared between axes, so use separate Renderer
        # for each  
        for ax in axs:
            rdr = Renderer(ax)
            #rdr.limits(400,400)
            rdr.render(root)

        fig.suptitle(tst.suptitle, horizontalalignment='left', family='monospace', fontsize=10, x=0.1, y=0.99) 

        fig.show()


        i = tst.i


