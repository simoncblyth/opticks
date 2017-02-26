#!/usr/bin/env python
"""
For dev notes see

* iterativeTreeEvaluationFake.py 
* iterativeTreeEvaluation.py 

"""

import logging
log = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


from node import Node, UNION, INTERSECTION, DIFFERENCE, BOX, SPHERE, EMPTY, desc
from node import trees, root0, root1, root2, root3, root4
from node import ubo, lrsph_d1, lrsph_d2, lrsph_u, lrsph_i

from intersect import intersect_primitive, intersect_miss, Ray

from boolean import boolean_table 
from boolean import desc_state, Enter, Exit, Miss
from boolean import desc_acts, BooleanStart, BooleanError, RetMiss, RetL, RetR, RetLIfCloser, RetRIfCloser, LoopL, LoopLIfCloser, LoopR, LoopRIfCloser
from boolean import RetFlippedRIfCloser
from boolean import IterativeError
from boolean import act_index, _act_index

from ctrl import CtrlLeft, CtrlRight, CtrlBreak, CtrlReturn, CtrlResumeFromLeft, CtrlResumeFromRight, desc_ctrl


class CSG(object):
    def __init__(self, level=1):
         self.level = level
         self.reset()

    def _get_debug(self):
         return self.level if self.iray in self._debug else 0
    debug = property(_get_debug)

    def reset(self, ray=None, iray=0, debug=[]):
        self.ray = ray 
        self.iray = iray
        self.count = 0 
        self.lname = None
        self.rname = None
        self._debug = debug

    def traverse_r(self, root, depth=0):
        if root.is_primitive:
            print "traverse_r(primitive) %d %r " % (depth, root)
        elif root.is_operation:
            print "traverse_r(operation) %d %r " % (depth, root)
            self.traverse_r(root.l, depth=depth+1)
            self.traverse_r(root.r, depth=depth+1)
        else:
            print "traverse_r(other) %d %r " % (depth, root)
        pass

    def classify(self, tt, nn, tmin):
        if tt > tmin:
            state = Enter if np.dot(nn, self.ray.direction) < 0. else Exit 
        else:
            state = Miss 
        pass
        return state

    def binary_action(self, operation, left, right, tminL=0, tminR=0):
        """
        * increasing tmin will cause some Enter/Exit intersect states to become Miss
        * when doin LoopL LoopR need to leave the other side state unchanged, hence
          using split tminL/tminR

        """
        act = BooleanError

        stateL = self.classify(left.t, left.n, tminL)  
        stateR = self.classify(right.t, right.n, tminR)

        acts = boolean_table(operation, stateL, stateR )

        #opr = "%s(%s:%s,%s:%s)" % ( desc[operation],left.name,desc_state[stateL], right.name,desc_state[stateR] )
        #trep = self.trep_fmt(tmin, left.t, right.t )

        act_RetMiss = (RetMiss & acts)
        act_RetL = (RetL & acts)
        act_RetR = (RetR & acts)
        act_LoopL = (LoopL & acts)
        act_LoopR = (LoopR & acts)
        act_RetLIfCloser = ((RetLIfCloser & acts) and left.t <= right.t)
        act_LoopLIfCloser = ((LoopLIfCloser & acts) and left.t <= right.t)
        act_RetRIfCloser = ((RetRIfCloser & acts) and right.t < left.t)
        act_RetFlippedRIfCloser = ((RetFlippedRIfCloser & acts) and right.t < left.t)
        act_LoopRIfCloser = ((LoopRIfCloser & acts) and right.t < left.t)

        if act_RetMiss:
            act = RetMiss
        elif act_RetL or act_RetLIfCloser: 
            act = RetLIfCloser if act_RetLIfCloser else RetL
        elif act_RetR or act_RetRIfCloser or act_RetFlippedRIfCloser: 
            if act_RetFlippedRIfCloser:
                act = RetFlippedRIfCloser 
            elif act_RetRIfCloser:
                act = RetRIfCloser
            else:
                act = RetR
            pass
        elif act_LoopL or act_LoopLIfCloser:
            act = LoopLIfCloser if act_LoopLIfCloser else LoopL
        elif act_LoopR or act_LoopRIfCloser:
            act = LoopRIfCloser if act_LoopRIfCloser else LoopR
        else:
            log.fatal("UNHANDLED acts %d " % (acts))
            assert 0
        pass
        return act 


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

    def iterative_intersect(self, root, depth=0, tmin=0, debug=1):
        """
        :param root:
        :return isect: I instance
        """
        self.check_tree(root)

        lhs = []
        rhs = []

        tranche = []
        tranche.append([Node.leftmost(root),None,CtrlLeft|CtrlRight])

        while len(tranche) > 0:
            begin, end, ctrl = tranche.pop()
            log.debug("start tranche %s begin %s end %s " % (desc_ctrl(ctrl), begin, end))
            
            p = begin
            while p is not end:

                act = 0 
                left = None
                right = None
                miss =  intersect_miss(p, self.ray, tmin)
                result = None
    
                tminL = tmin
                tminR = tmin

                ctrl = (CtrlLeft | CtrlRight)
                log.debug("p loop %s : %s " % (desc_ctrl(ctrl), p))

                loopcount = 0 
                while ctrl & (CtrlLeft | CtrlRight ):

                    loopcount += 1 
                    assert loopcount < 10  

                    if ctrl & CtrlLeft: 
                        if p.l.is_leaf:
                            left = intersect_primitive(p.l, self.ray, tminL)
                        else:
                            left = lhs.pop()
                        pass
                    pass

                    if ctrl & CtrlRight: 
                        if p.r.is_leaf:
                            right = intersect_primitive(p.r, self.ray, tminR)
                        else:
                            right = rhs.pop()
                        pass
                    pass

                    act = self.binary_action(p.operation, left, right, tminL, tminR)    

                    if act & (LoopL | LoopLIfCloser): 
                        ctrl = CtrlLeft
                        tminL = left.t
                    elif act & (LoopR | LoopRIfCloser): 
                        ctrl = CtrlRight
                        tminR = right.t 
                    elif act & (RetR | RetRIfCloser | RetL | RetLIfCloser | RetMiss | RetFlippedRIfCloser): 
                        ctrl = CtrlReturn 
                    else: 
                        assert 0, desc_acts(act)

                    #log.debug("p %5s(%s,%s) %s -> %s " % (p.tag,left,right, desc_acts(act), desc_ctrl(ctrl)))

                    ## hmm maybe need tmin stack ???
                    pass
                    if ctrl & CtrlLeft: 
                        if not p.l.is_leaf:
                            rhs.append(right)
                            tranche.append([p,None,CtrlResumeFromLeft])  
                            tranche.append([Node.leftmost(p.l),p.l.next_,CtrlLeft])
                            ctrl |= CtrlBreak
                        pass
                    pass
                    if ctrl & CtrlRight: 
                        if not p.r.is_leaf:
                            lhs.append(left)
                            tranche.append([p,None,CtrlResumeFromRight])  
                            tranche.append([Node.leftmost(p.r),p.r.next_,CtrlRight])
                            ctrl |= CtrlBreak
                        pass
                    pass
                    if ctrl & ( CtrlBreak | CtrlReturn):
                        break # out of ctrl-while loop (ie when no need for immediate looping for bileaf nodes)
                    pass # ctrl-while end
                pass   
          
                if ctrl & CtrlBreak:  
                    ctrl = ctrl & ~CtrlBreak 
                    log.info("break out of postorder traversal, for re-iteration of subtree ntranche %d " % len(tranche))
                    break 

                assert ctrl & CtrlReturn, desc_ctrl(ctrl)

                if act & (RetL | RetLIfCloser): 
                    result = left[:]
                elif act & (RetFlippedRIfCloser): 
                    result = right[:]
                    result.n = -result.n
                elif act & (RetR | RetRIfCloser): 
                    result = right[:]
                elif act & RetMiss: 
                    result = miss[:]
                else:
                    assert 0, desc_acts(act)
                pass
                result.addseq(act_index(act))
                
                assert len(result.shape) == 2 and result.shape[0] == 4 and result.shape[1] == 4
                log.debug("return %s " % result )

                if p.is_left:
                    lhs.append(result)
                else:
                    rhs.append(result)
                pass

                p = p.next_ 
            pass               # postorder tranche traversal while loop
        pass
        assert len(lhs) == 0, lhs
        assert len(rhs) == 1, rhs   # end with p.idx = 1 for the root
        return rhs[0]
     

    def recursive_intersect(self, p, ctrl=CtrlLeft|CtrlRight, depth=0, tmin=0):
        assert p.is_operation
        miss =  intersect_miss(p, self.ray, tmin)
        tminL = tmin
        tminR = tmin
        loopcount = 0 
        ctrl = (CtrlLeft | CtrlRight)

        while ctrl & (CtrlLeft | CtrlRight ):
            loopcount += 1 
            assert loopcount < 10  

            if ctrl & CtrlLeft: 
                if p.l.is_leaf:
                    left = intersect_primitive(p.l, self.ray, tminL)
                else:
                    left = self.recursive_intersect(p.l, depth=depth+1, tmin=tminL)
                pass
            pass
            if ctrl & CtrlRight: 
                if p.r.is_leaf:
                    right = intersect_primitive(p.r, self.ray, tminR)
                else:
                    right = self.recursive_intersect(p.r, depth=depth+1, tmin=tminR)
                pass
            pass

            act = self.binary_action(p.operation, left, right, tminL, tminR)    

            if act & (LoopL | LoopLIfCloser): 
                ctrl = CtrlLeft
                tminL = left.t
            elif act & (LoopR | LoopRIfCloser): 
                ctrl = CtrlRight
                tminR = right.t 
            elif act & (RetR | RetRIfCloser | RetL | RetLIfCloser | RetMiss | RetFlippedRIfCloser): 
                ctrl = CtrlReturn 
            else: 
                assert 0, desc_acts(act)
            pass
            if ctrl & CtrlReturn:
                break 
            pass 
        pass
        if act & (RetL | RetLIfCloser): 
            result = left[:]
        elif act & (RetFlippedRIfCloser): 
            result = right[:]
            result.n = -result.n
        elif act & (RetR | RetRIfCloser): 
            result = right[:]
        elif act & RetMiss: 
            result = miss[:]
        else:
            assert 0, desc_acts(act)
        pass
        result.addseq(act_index(act))
        return result[:]
 

    @classmethod
    def trep_fmt(cls, tmin, tl, tr ):
        return "tmin/tl/tr %5.2f %5.2f %5.2f " % (tmin if tmin else -1, tl if tl else -1, tr if tr else -1 )

    def compare_intersects(self, tst):
        nray = len(tst.rays)

        self.i = np.zeros((2,nray,4,4), dtype=np.float32 )

        self.prob = []
        nerr = 0 

        for iray, ray in enumerate(tst.rays):
            for recursive in [1,0]:
                self.typ = "RECURSIVE" if recursive else "ITERATIVE"
                self.reset(ray=ray, iray=iray, debug=tst.debug) 
                if recursive:
                    isect = self.recursive_intersect(tst.root, depth=0) 
                else: 
                    isect = self.iterative_intersect(tst.root)
                pass
                self.i[recursive,iray] = isect 

                if self.debug and isect is not None:
                    log.info("[%d] %s intersect tt %s nn %r " % (-1, self.typ, isect.t, isect.n ))
                pass
            pass

            t0 = self.i[0,iray,0,3]  # intersect t 
            t1 = self.i[1,iray,0,3]
            ok_t = np.allclose( t0, t1 )

            n0 = self.i[0,iray,0,:3]  # intersect normal
            n1 = self.i[1,iray,0,:3]
            ok_n = np.allclose( n0, n1 )

            o0 = self.i[0,iray,2,:3] # ray.origin
            o1 = self.i[1,iray,2,:3]
            ok_o = np.allclose( o0, o1 )

            d0 = self.i[0,iray,3,:3] # ray.direction
            d1 = self.i[1,iray,3,:3]
            ok_d = np.allclose( d0, d1 )

            p0 = np.repeat(t0, 3).reshape(-1,3)*d0 + o0
            p1 = np.repeat(t1, 3).reshape(-1,3)*d1 + o1
            ok_p = np.allclose( p0, p1 )

            if not (ok_p and ok_n and ok_t and ok_d and ok_o):
                self.prob.append(iray)
            pass

        pass
        log.info("%10s %d/%d rays with intersect mismatches : %s  iterative nerr %d " % (tst.name, len(self.prob),nray,repr(self.prob), nerr))


    

    def plot_intersects(self, ax, normal=False, xof=[0,300],yof=[0,0]):
        """

            csg.i[1,:,1,3].view(np.uint32)

        """
        sc = 30 

        tt = {}
        tt[0] = self.i[0,:,0,3]  # parametric t at intersection
        tt[1] = self.i[1,:,0,3]

        nn = {}
        nn[0] = self.i[0,:,0,:3]  # normal at intersection
        nn[1] = self.i[1,:,0,:3]

        oo = {}
        oo[0] = self.i[0,:,2,:3] # ray.origin
        oo[1] = self.i[1,:,2,:3]

        dd = {}
        dd[0] = self.i[0,:,3,:3] # ray.direction
        dd[1] = self.i[1,:,3,:3]

        pp = {}
        pp[0] = np.repeat(tt[0], 3).reshape(-1,3)*dd[0] + oo[0]   # intersect position
        pp[1] = np.repeat(tt[1], 3).reshape(-1,3)*dd[1] + oo[1]


        qq = {}
        qq[0] = self.i[0,:,1,3].view(np.uint32)    # act sequence
        qq[1] = self.i[1,:,1,3].view(np.uint32)

        _color = {  0:'r', 3:'g', 4:'b' }

        label_ = lambda _:desc_acts(0x1 << _)
        color_ = lambda _:_color.get(_,'c')
            
        cqq = {}
        cqq[0] = map(color_, qq[0])
        cqq[1] = map(color_, qq[1])


        X,Y,Z = 0,1,2

        sel = tt[0] > 0

        prob = self.prob



        for r in [0,1]:
            ax.scatter( xof[r] + pp[r][sel,X] , yof[r] + pp[r][sel,Y], c=cqq[r])

            if normal:
                ax.scatter( xof[r] + pp[r][sel,X] + nn[r][sel,X]*sc, yof[r] + pp[r][sel,Y]+nn[r][sel,Y]*sc)

            #if len(prob) > 0:
            #    ax.scatter( xoff + self.ipos[recursive, prob,0], self.ipos[recursive, prob,1], c="r" )
            #    ax.scatter( xoff + self.ipos[recursive, prob,0], self.ipos[recursive, prob,1], c="g" )




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

    def _get_rays(self):
        rays = []
        if "xray" in self.source:
            rays += [Ray(origin=[0,0,0], direction=[1,0,0])]

        if "aringlight" in self.source:
            ary = Ray.aringlight(num=self.num, radius=1000)
            rays += Ray.make_rays(ary)

        if "origlight" in self.source:
            rays += Ray.origlight(num=self.num)

        if "lsquad" in self.source:
            rays += [Ray(origin=[-300,y,0], direction=[1,0,0]) for y in range(-50,50+1,10)]

        if "rsquad" in self.source:
            rays += [Ray(origin=[300,y,0], direction=[-1,0,0]) for y in range(-50,50+1,10)]
        pass

        return rays

    rays = property(_get_rays)



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
        log.info("render %r " % root )
        p = Node.leftmost(root)
        while p is not None:
            #print p
            if p.is_bileaf:
                self.render_primitive(p.l)
                self.render_primitive(p.r)
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
        fc = 'none'
        patch.set_ec(ec)
        patch.set_fc(fc)

    def render_sphere(self,node):
        center = node.param[:3]
        radius = node.param[3] 

        patch = mpatches.Circle(center[self.axes],radius) 
        self.autocolor(patch, node.idx)
        self.add_patch(patch)

    def render_box(self,node):
        cen = node.param[:3]
        sid = node.param[3]
        bmin = cen - sid
        bmax = cen + sid
        dim = bmax - bmin
        width = dim[self.axes[0]]
        height = dim[self.axes[1]]
        botleft = bmin[self.axes]
        patch = mpatches.Rectangle( botleft, width, height)
        self.autocolor(patch, node.idx)
        self.add_patch(patch)

    def add_patch(self, patch):
        self.ax.add_artist(patch)
    pass
pass




if __name__ == '__main__':

    logformat = "%(asctime)s %(name)s %(levelname)-8s %(message)s"
    logging.basicConfig(level=logging.INFO,format=logformat)

    plt.rcParams['figure.figsize'] = 18,10.2 

    plt.ion()
    plt.close("all")

    #root = trees[1]
    #root = trees[3]
    #root = trees[4]
    #root = trees[5]

    #for root in trees:
    if 1:
        #root = ubo
        #root = lrsph_d1

        source = None
  
 
        ## this should be cresent moon shape 

        #source, root = "lsquad", lrsph_d1    # left sphere intersects only, with propa outwards normals
        #source, root = "rsquad", lrsph_d2     # right sphere intersects only, with propa outwards normals

        #source, root = "lsquad", lrsph_d2     # incorrect inner intersects, with 
        #source, root = "rsquad", lrsph_d1  

        #root = lrsph_u
        #source, root = "origlight", lrsph_i


        source, root = "origlight", lrsph_d1  
        #source, root = "aringlight", lrsph_d1  


        root.annotate()

        fig = plt.figure()
        #fig.set_size_inches(20,20)

        ax = fig.add_subplot(1,1,1, aspect='equal')

        tst = T(root,level=3,debug=[], num=100, source=source)
        csg = CSG(level=tst.level)

        csg.compare_intersects( tst )
        csg.plot_intersects( ax, xof=[0,0], yof=[150,-150], normal=False)

        rdr = Renderer(ax)
        rdr.limits(300,300)
        rdr.render(root)

        fig.show()


        self = csg


