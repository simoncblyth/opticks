#!/usr/bin/env python
"""
For dev notes see

* iterativeTreeEvaluationFake.py 
* iterativeTreeEvaluation.py 

"""

import logging
log = logging.getLogger(__name__)

import numpy as np
from opticks.ana.nbase import count_unique
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


#from node import UNION, INTERSECTION, DIFFERENCE, desc
from node import Node, BOX, SPHERE, EMPTY
from node import trees
from node import root0, root1, root2, root3, root4
from node import ubo, lrsph_d1, lrsph_d2, lrsph_u, lrsph_i
from node import lrbox_u, lrbox_i, lrbox_d1, lrbox_d2

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


        if ctrl & CtrlReturnMiss:
            result = miss[:]
        elif ctrl & CtrlReturnLeft:
            result = left[:]
        elif ctrl & CtrlReturnRight: 
            result = right[:]
        elif ctrl & CtrlReturnFlipRight: 
            result = right[:]
            result.n = -result.n
        else:
            assert 0, desc_ctrl(ctrl)
        pass
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
                                log.error("lhs pop from empty")
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
                                log.error("rhs pop from empty")
                        pass
                    pass

                    ctrl = self.binary_ctrl(p.operation, left, right, tminL, tminR)    

                    if ctrl & (CtrlReturnMiss | CtrlReturnLeft | CtrlReturnRight | CtrlReturnFlipRight):
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
                    if reiterate_:  # despite loopers non-leaves need to break to traverse subtree
                        break 
                    pass
                    loopcount += 1 
                    assert loopcount < 10  
                    pass 
                pass  # end while ctrl loop   
          
                if reiterate_:  
                    log.info("break out of postorder traversal, for re-iteration of subtree ntranche %d " % len(tranche))
                    break 

                assert ctrl & (CtrlReturnLeft | CtrlReturnRight | CtrlReturnMiss | CtrlReturnFlipRight )
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
            log.error("lhs expected to end with zero isects, see %d " % len(lhs))

        if len(rhs) != 1:
            log.error("rhs expected to end with one isect, see %d " % len(rhs))

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
                pass
            pass 
            loopcount += 1 
            assert loopcount < 10  
        pass  #  end while ctrl loop

        assert ctrl & (CtrlReturnLeft | CtrlReturnRight | CtrlReturnMiss | CtrlReturnFlipRight )
        result = self.binary_result(ctrl, miss, left, right)
        return result
 

    @classmethod
    def trep_fmt(cls, tmin, tl, tr ):
        return "tmin/tl/tr %5.2f %5.2f %5.2f " % (tmin if tmin else -1, tl if tl else -1, tr if tr else -1 )

    def compare_intersects(self, tst, rr=[1,0]):
        nray = len(tst.rays)
        a = np.zeros((2,nray,4,4), dtype=np.float32 )
       
        iis = IIS(a)
        iis._ctrl_color = _ctrl_color

        self.iis = iis
        self.prob = []
        nerr = 0 

        for iray, ray in enumerate(tst.rays):
            for r in rr:
                self.typ = "RECURSIVE" if r else "ITERATIVE"
                self.reset(ray=ray, iray=iray, debug=tst.debug) 

                if r:
                    isect = self.recursive_intersect(tst.root) 
                else: 
                    isect = self.iterative_intersect(tst.root)
                pass
                self.iis[r,iray] = isect 

                if self.debug:
                    log.info("[%d] %s intersect tt %s nn %r " % (-1, self.typ, isect.t, isect.n ))
                pass
            pass

            t = self.iis.t[:,iray]  # intersect t 
            ok_t = np.allclose( t[0], t[1] )

            n = self.iis.n[:,iray]  # intersect normal
            ok_n = np.allclose( n[0], n[1] )

            o = self.iis.o[:,iray] # ray.origin
            ok_o = np.allclose( o[0], o[1] )

            d = self.iis.d[:,iray] # ray.direction
            ok_d = np.allclose( d[0], d[1] )

            p = self.iis.ipos[:,iray]
            ok_p = np.allclose( p[0], p[1] )

            ## hmm could compre all at once both within intersect 

            if not (ok_p and ok_n and ok_t and ok_d and ok_o):
                self.prob.append(iray)
            pass

        pass
        log.info("%10s %d/%d rays with intersect mismatches : %s  iterative nerr %d " % (tst.name, len(self.prob),nray,repr(self.prob), nerr))

        q = self.iis.seq
        iq = count_unique(q[0])
        rq = count_unique(q[1])
 
        try: 
            log.info(" iterative\n %s"  % desc_ctrl_cu(iq) )
            log.info(" recursive\n %s " % desc_ctrl_cu(rq) )
        except KeyError:
            print "q", q
            print "iq", iq
            print "rq", rq

    

    def plot_intersects(self, axs, normal=False, origin=True, xof=[0,300],yof=[0,0], rr=[1,0]):
        """
        """
        sc = 30 

        i = self.iis
        t = i.t
        q = i.seq
        n = i.n
        o = i.o
        d = i.d
        p = i.ipos
        c = i.cseq

        X,Y,Z = 0,1,2

        pr = self.prob

        for r in rr:
            ax = axs[r]

            if origin:
                ax.scatter( xof[r] + o[r][:,X] , yof[r] + o[r][:,Y]  )

            sel = t[r] > 0
            ax.scatter( xof[r] + p[r][sel,X] , yof[r] + p[r][sel,Y], c=c[r] )

            if normal:
                ax.scatter( xof[r] + p[r][sel,X] + n[r][sel,X]*sc, yof[r] + p[r][sel,Y] + n[r][sel,Y]*sc )

            if len(pr) > 0:
                sel = pr
                ax.scatter( xof[r] + p[r][sel,X] , yof[r] + p[r][sel,Y], c=c[r][sel] )
            pass
        
            #sel = slice(0,None)

            for _ in np.where(sel)[0]:
                x1 = o[r,_,X]
                y1 = o[r,_,Y]
                x2 = p[r,_,X]
                y2 = p[r,_,Y]
             
                ax.plot( [x1,x2], [y1,y2], c[r][_]+'-' )

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


    def _get_suptitle(self):
        return "%s" % self.name
    suptitle = property(_get_suptitle)

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


        if "qray" in self.source:
            s = 300
            r = range(-s,s+1,2)
            rays += [Ray(origin=[s,y,0], direction=[-1,0,0])  for y in r]
            rays += [Ray(origin=[-s,y,0], direction=[1,0,0])  for y in r]
            rays += [Ray(origin=[x,-s,0], direction=[0,1,0])  for x in r]
            rays += [Ray(origin=[x, s,0], direction=[0,-1,0]) for x in r]
 
        if "seray" in self.source:
            s = 300
            r = range(-s,s+1,2)
            rays += [Ray(origin=[v,v-s,0], direction=[-1,1,0]) for v in r]
 

        pass

        return rays

    rays = property(_get_rays)



class Renderer(object):
    def __init__(self, axs, axes=[0,1]):
        self.axs = axs
        self.axes = axes

    def limits(self, sx=200, sy=150):
        for ax in self.axs:
            ax.set_xlim(-sx,sx)
            ax.set_ylim(-sy,sy)

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
        for ax in self.axs:
            ax.add_artist(patch)
        pass
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

    #root = trees[1]
    #root = trees[3]
    #root = trees[4]
    #root = trees[5]

    for root in trees:
    #if 1:
 
        ## this should be cresent moon shape 

        #source, root = "lsquad", lrsph_d1    # left sphere intersects only, with propa outwards normals
        #source, root = "rsquad", lrsph_d2     # right sphere intersects only, with propa outwards normals

        #source, root = "lsquad", lrsph_d2     # incorrect inner intersects, with 
        #source, root = "rsquad", lrsph_d1  

        #source, root = "origlight", lrsph_i   # looks correct : half left/right
        #source, root = "aringlight", lrsph_i   # looks funny, 52 missers 
        #source, root = "aringlight", lrsph_u   # looks correct


        #source, root = "origlight", lrsph_u    # looks correct : half left/right
        #source, root = "aringlight", lrsph_u    # looks correct : half left/right

        #root = lrsph_u
        #source, root = "origlight", lrsph_i

        # lrsph_d1 : left sphere - right sphere  
        #source, root = "origlight", lrsph_d1    # get expected arc of the cresent moon to left, half CtrlReturnMiss, half CtrlReturnFlipRight

        #source, root, epsilon  = "aringlight", lrsph_d1, 1e-5    # unexpected left intersects that should be right flips
        #source, root, epsilon = "aringlight", lrsph_d1, 1e-4      # get expected with larger epsilon, cicle sqrt etc... require larger epsilon

        #source, root, epsilon = "aringlight", lrsph_d2, 1e-4      # get expected with larger epsilon, cicle sqrt etc... require larger epsilon


        #source, root = "qray", lrsph_d1         # still bad intersects, until applied epsilon to the advancers
        #source, root = "qray", lrsph_d2         # still bad intersects, until applied epsilon to advancers

        #source, root = "aringlight", lrbox_u    #  correct ~equal left/right
        #source, root = "origlight", lrbox_u    #  correct ~equal left/right

        #source, root = "aringlight", lrbox_i    #  unexpected misses
        #source, root = "qray",      lrbox_i      # getting expected intersects
        #source, root = "origlight", lrbox_i      #  correct ~equal l/r

        #source, root = "origlight", lrbox_d1   # expected ~equal CtrlReturnMiss, CtrlReturnFlipRight
        #source, root = "aringlight", lrbox_d1   # incorrect left intersects

        #source, root = "qray", lrbox_d1   # correct, huh...
        #source, root = "seray", lrbox_d1   # correct, huh... 
 

        root.annotate()


        fig = plt.figure()

        ax1 = fig.add_subplot(1,2,1, aspect='equal')
        ax2 = fig.add_subplot(1,2,2, aspect='equal')
        axs = [ax1,ax2]

        tst = T(root,level=3,debug=[0], num=100, source=source)
        csg = CSG(level=tst.level, epsilon=epsilon)

        rr = [0,1]   # recursive only [1], iterative only [0], or both [0,1] [1,0]
        xof = [0,0]        # offsets for iterative and recursive plotting
        #yof = [150, -150]   
        yof = [0, 0]


        csg.compare_intersects( tst, rr=rr )
        csg.plot_intersects( axs=axs, rr=rr, xof=xof, yof=yof, normal=False)

        rdr = Renderer(axs)
        rdr.limits(300,300)
        rdr.render(root)

        fig.suptitle(tst.suptitle) 

        fig.show()


        self = csg
        i = csg.iis

