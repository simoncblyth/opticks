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


from node import Node, root0, root1, root2, root3, root4, UNION, INTERSECTION, DIFFERENCE, BOX, SPHERE, EMPTY, desc
from intersect import intersect_primitive, I, Ray

from boolean import boolean_table 
from boolean import desc_state, Enter, Exit, Miss
from boolean import desc_acts, BooleanStart, BooleanError, RetMiss, RetL, RetR, RetLIfCloser, RetRIfCloser, LoopL, LoopLIfCloser, LoopR, LoopRIfCloser, FlipR
from boolean import ResumeFromLoopL, ResumeFromLoopR, NewTranche

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
        act_LoopRIfCloser = ((LoopRIfCloser & acts) and right.t < left.t)

        if act_RetMiss:
            act = RetMiss
        elif act_RetL or act_RetLIfCloser: 
            act = RetLIfCloser if act_RetLIfCloser else RetL
        elif act_RetR or act_RetRIfCloser: 
            act = RetRIfCloser if act_RetRIfCloser else RetR
            if (FlipR & acts): right.n = -right.n  ## hmm flip/flop danger ..move to point of use ? via RetFlippedR RetFlippedRIfCloser ?
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
            log.info("left-op is root, ie height 1 tree with a single operation and two leaf primitives")
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
                miss =  I(None,None,None,RetMiss)
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
                    elif act & (RetR | RetRIfCloser | RetL | RetLIfCloser | RetMiss): 
                        ctrl = CtrlReturn 
                    else: 
                        assert 0, desc_acts(act)

                    log.debug("p %5s(%s,%s) %s -> %s " % (p.tag,left,right, desc_acts(act), desc_ctrl(ctrl)))

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
                    result = left
                elif act & (RetR | RetRIfCloser): 
                    result = right
                elif act & RetMiss: 
                    result = miss
                else:
                    assert 0, desc_acts(act)
                pass
                log.debug("return %s " % result )

                if p.is_left:
                    lhs.append(result)
                else:
                    rhs.append(result)
                pass

                p = p.next_ 
            pass      # postorder tranche traversal while loop
        pass
        assert len(lhs) == 0, lhs
        assert len(rhs) == 1, rhs   # end with p.idx = 1 for the root
        return rhs[0]
     


    def recursive_intersect(self, root, depth=0, tmin=0):
        """
        * minimizing use of member vars makes recursive algorithm easier to understand
        * instead use local vars, which will have different existance at the 
          different levels of the recursion
        * can think of member vars as effective globals wrt the recursion

        * loopers result in recursive call repeating the same node, with tmin advanced 
        * recursively never traverse the root again ? are always going down with root.left root.right
          never going up ?

        * although the traverse never goes up, completion of the recursive instance calls 
          does go back up once hitting primitives in the leaves 

        """
        assert root
        assert self.typ == "RECURSIVE"
        if depth == 0:
            self.top = root
        pass
        self.node = root  

        if root.is_primitive:
            return intersect_primitive(root, self.ray, tmin)

        elif root.is_operation:

            act = BooleanStart
            left = None
            right = None
            miss =  I(None,None,None,RetMiss)
            loopcount = 0 
            looplimit = 10 

            while loopcount < looplimit and act in [BooleanStart, LoopLIfCloser, LoopL, LoopRIfCloser, LoopR]:
                loopcount += 1 

                tminL = left.t if act in [LoopL,LoopLIfCloser] else tmin 
                tminR = right.t if act in [LoopR,LoopRIfCloser] else tmin 

                if act in [BooleanStart, LoopL, LoopLIfCloser]:
                    left  = self.recursive_intersect(root.l, depth=depth+1, tmin=tminL) 
                pass
                if act in [BooleanStart, LoopR, LoopRIfCloser]:
                    right  = self.recursive_intersect(root.r, depth=depth+1, tmin=tminR)
                pass
                act = self.binary_action(root.operation, left, right, tminL, tminR)    
            pass     
   
            self.act = act 

            if self.debug > 1:
                log.info("(%d)[%d] RECURSIVE %s : %s -> %s : %s " % (self.iray,loopcount,root.name,"opr",desc_acts(self.act),"trep" ))

            result = None
            if act in [RetL, RetLIfCloser]:
                result = left
            elif act in [RetR, RetRIfCloser]:
                result = right
            elif act in [RetMiss]:
                result = miss
            else:
                log.fatal("unexpected act %s " % act )
                assert 0
            return result
            pass
        pass
        log.fatal("unexpected fallthru ")
        assert 0
        return None

    @classmethod
    def trep_fmt(cls, tmin, tl, tr ):
        return "tmin/tl/tr %5.2f %5.2f %5.2f " % (tmin if tmin else -1, tl if tl else -1, tr if tr else -1 )

    def compare_intersects(self, tst):
        nray = len(tst.rays)
        self.ipos = np.zeros((2,nray, 3), dtype=np.float32 ) 
        self.ndir = np.zeros((2,nray, 3), dtype=np.float32 ) 
        self.tval = np.zeros((2,nray), dtype=np.float32 )
        self.aval = np.zeros((2,nray), dtype=np.int32 )

        self.prob = []
        for iray, ray in enumerate(tst.rays):
            for recursive in [1,0]:
                self.typ = "RECURSIVE" if recursive else "ITERATIVE"
                self.reset(ray=ray, iray=iray, debug=tst.debug) 

                if iray in tst.skip:
                    log.warning("skipping iray %d " % iray)
                    continue 

                if self.debug > 0:
                    log.info(" ray(%d) %r " % (iray,ray) ) 
                if self.debug > 1:
                    log.info(" %r " % (tst.root)) 


                if recursive:
                    isect = self.recursive_intersect(tst.root, depth=0) 
                else: 
                    isect = self.iterative_intersect(tst.root)
                pass
                if self.debug:
                    log.info("[%d] %s intersect tt %s nn %r " % (-1, self.typ, isect.t, isect.n ))

                if not isect.t is None:
                    ix = 0 if self.typ == "ITERATIVE" else 1
                    self.ipos[ix,iray] = ray.position(isect.t)
                    self.ndir[ix,iray] = isect.n
                    self.tval[ix,iray] = isect.t
                    self.aval[ix,iray] = isect.code
                pass
            pass

            ok_pos = np.allclose( self.ipos[0,iray], self.ipos[1,iray] )
            ok_dir = np.allclose( self.ndir[0,iray], self.ndir[1,iray] )
            ok_tva = np.allclose( self.tval[0,iray], self.tval[1,iray] )

            if not (ok_pos and ok_dir and ok_tva):
                self.prob.append(iray)

        pass
        log.info("%10s %d/%d rays with intersect mismatches : %s " % (tst.name, len(self.prob),nray,repr(self.prob)))

    def plot_intersects(self, ax, normal=False, offset=600):
        sc = 10 

        prob = self.prob

        for recursive in [0,1]:
            xoff = offset if recursive else 0
            ax.scatter( xoff + self.ipos[recursive,:,0]                        , self.ipos[recursive,:,1] )

            if normal:
                ax.scatter( xoff + self.ipos[recursive,:,0]+self.ndir[recursive,:,0]*sc , self.ipos[recursive,:,1]+self.ndir[recursive,:,1]*sc )
    
            if len(prob) > 0:
                ax.scatter( xoff + self.ipos[recursive, prob,0], self.ipos[recursive, prob,1], c="r" )
                ax.scatter( xoff + self.ipos[recursive, prob,0], self.ipos[recursive, prob,1], c="g" )




class T(object):
    def __init__(self, root, debug=[], skip=[], notes="", source="aringlight,origlight", num=200, level=1):
        """
        :param root:
        :param name:
        :param debug: list of ray index for dumping
        """
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
        pass

    def add_patch(self, patch):
        self.ax.add_artist(patch)
    pass
pass



if __name__ == '__main__':

    logformat = "%(asctime)s %(name)s %(levelname)-8s %(message)s"
    logging.basicConfig(level=logging.INFO,format=logformat)

    plt.ion()
    plt.close()

    root = root4
    root.tree_labelling()
    Node.dress(root)


    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, aspect='equal')

    tst = T(root,level=3,debug=[], num=200)
    csg = CSG(level=tst.level)

    csg.compare_intersects( tst )
    csg.plot_intersects( ax, offset=300)

    rdr = Renderer(ax)
    rdr.limits(500,500)
    rdr.render(root)

    fig.show()



