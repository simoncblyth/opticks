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

    def binary_result(self, ctrl, miss, left, right, tmin):
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
        result.rtmin = tmin
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
        tranche.append([tmin,Node.leftmost(root),None])

        tcount = 0 

        while len(tranche) > 0:
            tmin, begin, end = tranche.pop()
            #log.info("%s : start tranche %d begin %s end %s " % (self.pfx, tcount, begin, end))
            tcount += 1    
            assert tcount < 10

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
                                log.error("%s : lhs pop from empty" % self.pfx)
                                self.tst.ierr += 1
                                left = miss
                            pass
                        pass
                    pass
                    if ctrl & CtrlLoopRight: 
                        if p.r.is_leaf:
                            right = intersect_primitive(p.r, self.ray, tminR)
                        else:
                            try:
                                right = rhs.pop()
                            except IndexError:
                                log.error("%s : rhs pop from empty" % self.pfx)
                                self.tst.ierr += 1
                                right = miss
                            pass
                        pass
                    pass

                    ctrl = self.binary_ctrl(p.operation, left, right, tminL, tminR)    

                    if ctrl in [CtrlReturnMiss, CtrlReturnLeft, CtrlReturnRight, CtrlReturnFlipRight]:
                        pass # will fall out of the ctrl while as no longer loopers
                    elif ctrl == CtrlLoopLeft: 
                        tminL = left.t + self.epsilon
                        if not p.l.is_leaf:
                            rhs.append(right)
                            tranche.append([tmin,p,None])  # should this be tminL ? its for continuation
                            tranche.append([tminL,Node.leftmost(p.l),p.l.next_])
                            reiterate_ = True 
                        pass
                    elif ctrl == CtrlLoopRight: 
                        tminR = right.t + self.epsilon
                        if not p.r.is_leaf:
                            lhs.append(left)
                            tranche.append([tmin,p,None])  # should this be tminR ?
                            tranche.append([tminR,Node.leftmost(p.r),p.r.next_])
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
                    #log.info("break out of postorder traversal, for re-iteration of subtree ntranche %d " % len(tranche))
                    break 

                assert ctrl in [CtrlReturnMiss, CtrlReturnLeft,CtrlReturnRight,CtrlReturnFlipRight]
                result = self.binary_result(ctrl, miss, left, right, tmin)

                if p.is_left:
                    if self.debug:
                        log.info("%s : lhs append %d : %s " % (self.pfx, len(lhs), lhs ))
                    lhs.append(result)
                else:
                    if self.debug:
                        log.info("%s : rhs append %d : %s " % (self.pfx, len(rhs), rhs ))
                    rhs.append(result)
                pass
                if self.debug:
                    p.iterative = result

                p = p.next_ 
            pass               # postorder tranche traversal while loop
        pass

        #assert len(lhs) == 0, lhs
        #assert len(rhs) == 1, rhs   # end with p.idx = 1 for the root

        if not len(lhs) == 0:
            log.error("%s : lhs has %d, expect 0 " % (self.pfx, len(lhs)) )
            self.tst.ierr += 1
        pass

        if not len(rhs) == 1:
            log.error("%s : rhs has %d, expect 1 " % (self.pfx, len(rhs)) )
            self.tst.ierr += 1
        pass

        return rhs[0]
     

    def recursive_intersect(self, p, tmin=0):
        """
        binary recursion stack starts unwinding when
        reach bileaf node p

        * p.l.is_leaf and p.r.is_leaf
        
        recursive argument changes:
        
        * p <- p.l  tmin <- tminL
        * p <- p.r  tmin <- tminR


        * http://stackoverflow.com/questions/12468251/convert-recursion-to-iteration
        * http://stackoverflow.com/questions/7548026/convert-recursive-binary-tree-traversal-to-iterative

        """ 
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
        result = self.binary_result(ctrl, miss, left, right, tmin)
        if self.debug:
            p.recursive = result

        return result
 
    def recursive_intersect_2(self, p, tmin=0):

        if p.is_leaf:
            return intersect_primitive(p, self.ray, tmin)
        pass
        miss =  intersect_miss(p, self.ray, tmin)
        tminL = tmin
        tminR = tmin

        loopcount = 0 
        ctrl = (CtrlLoopLeft | CtrlLoopRight)
        while ctrl & (CtrlLoopLeft | CtrlLoopRight ):
            if ctrl & CtrlLoopLeft: 
                left = self.recursive_intersect_2(p.l, tmin=tminL)
            pass
            if ctrl & CtrlLoopRight: 
                right = self.recursive_intersect_2(p.r, tmin=tminR)
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
        result = self.binary_result(ctrl, miss, left, right, tmin)
        if self.debug:
            p.recursive = result
        return result
 




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
        icu_ = desc_ctrl_cu(self.icu, label="ITERATIVE")
        rcu_ = desc_ctrl_cu(self.rcu, label="RECURSIVE")
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

    roots = trees
    #roots = [root3, root4]
    #roots = [lrbox_u]
    #roots = [root4]
    #roots = [lrbox_d1, ubo]
    #roots = [lbox_ue, smb_lbox, smb_lbox_ue]
    #roots = [smb_lbox_ue]

    rr = [0,1]   # recursive only [1], iterative only [0], or both [0,1] [1,0]

    tsts = []
    if 1:
        for root in roots:
            #if root.ok: continue 
            #if root.name in ["root3","root4"]: continue
            tsts.append(T(root,level=3,debug=[], num=25, source="leaflight",origin=True,rayline=True, scale=0.1, sign=1))
        pass

    t_root3a = T(root3, level=3, debug=[], num=100, source="aringlight" )
    t_root3b = T(root3, level=3, debug=[], num=100, source="origlight" )

    #t_root4a = T(root4, level=3, debug=[0], num=200, source="aringlight", notes="ok" )
    t_root4a = T(root4, level=3, debug=[0], num=20, source="leaflight", notes="ok", rayline=True, raytext=False, origin=True)
    t_root4b = T(root4, level=3, debug=[1], num=100, source="origlight", notes="pop from empty lhs" )

    #tsts.append(t_root4a)
    
    # None: means use value from T ctor, or the default if not defined there
    source = None
    epsilon = None
    origin = None
    normal = None
    rayline = None


    for itst,tst in enumerate(tsts):
        print "%2d : %15s : %s " % (itst, tst.root.name, tst.root )

        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1, aspect='equal')
        ax2 = fig.add_subplot(1,2,2, aspect='equal')
        axs = [ax1,ax2]

        csg = CSG(level=tst.level, epsilon=epsilon)

        tst.compare_intersects( csg, rr=rr )
        tst.plot_intersects( axs=axs, rr=rr, normal=normal, origin=origin, rayline=rayline)

        # seems patches cannot be shared between axes, so use separate Renderer
        # for each  
        for ax in axs:
            rdr = Renderer(ax)
            #rdr.limits(400,400)
            rdr.render(tst.root)

        fig.suptitle(tst.suptitle, horizontalalignment='left', family='monospace', fontsize=10, x=0.1, y=0.99) 
        fig.show()

        i = tst.i


