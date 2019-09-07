#!/usr/bin/env python
#
# Copyright (c) 2019 Opticks Team. All Rights Reserved.
#
# This file is part of Opticks
# (see https://bitbucket.org/simoncblyth/opticks).
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and 
# limitations under the License.
#

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




X,Y,Z,W = 0,1,2,3



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




    def iterative_intersect(self, root, depth=0, tmin=0, debug=1, icheck=True):
        """
        :param root:
        :return isect: I instance

        0-based postorder p-indices from p0 at leftmost to p14 at root, for height 4 tree, 
        excluding leaf nodes to make height 3 internal nodes::

            In [78]: root4.txt
            Out[78]: 
            root4                                                                                                                            
                                                                         14                                                                
                                                                          o                                                                
                                          6                                                              13                                
                                          o                                                               o                                
                          2                               5                               9                              12                
                          o                               o                               o                               o                
                  0               1               3               4               7               8              10              11        
                  o               o               o               o               o               o               o               o        
                                                                                                                                           
              o       o       o       o       o       o       o       o       o       o       o       o       o       o       o       o    



        Consider evaluating p13 subtree, of sub-height 2 (from full-height - depth = 3 - 1 = 2 )  

        * reiterate p13.left   (p9) ->  repeat p7,  p8, p9  [with live tree Node.leftmost(p13.l)->p7 ]
        * reiterate p13.right (p12) ->  repeat p10,p11,p12  [with live tree Node.leftmost(p13.r)->p10 ]

        Need to derive the p-indices that need to repeat based on the depth of p13 and
        the height of the tree.

        * sub-nodes 7,   l-nodes = r-nodes = (sub-nodes - 1)/2 = (7 - 1)/2 = 3 

        *   13-3   +0, +1, +2 =  10, 11, 12 
        *   13-3*2 +0, +1, +2 =  7, 8, 9  

        *  beneath p13 are two sub-subtrees 


        Operation height is 3,    2^(h+1) = 2^(3+1) - 1               = 15  nodes in total
        Subtree starting p13 is at depth 1,   2^(h-d+1) = 2^(2+1) - 1 =  7  nodes 
        Subtree starting p9 is at depth 2,    2^(h-d+1) = 2^(1+1) - 1 =  3  nodes

        """
        self.check_tree(root)

        lhs = []
        rhs = []


        tranche = []
        tranche.append([tmin,Node.leftmost(root),None])

        if icheck:
            # check using 0-based postorder indices
            i_height = root.maxdepth - 1   # excluding leaves
            numInternalNodes = Node.NumNodes(i_height)

            itranche = []
            itranche.append([tmin, 0, numInternalNodes])

            i_postorder = Node.postorder_r(root, nodes=[], leaf=False)
            assert len(i_postorder) == numInternalNodes

            lmo = Node.leftmost(root) 
            assert lmo.pidx == 0 and lmo is i_postorder[0] 
            assert i_postorder[numInternalNodes-1].next_ == None

        else:
            itranche = None
        pass


        tcount = 0 

        while len(tranche) > 0:

            assert len(tranche) <= 4

            tmin, begin, end = tranche.pop()
            
            if icheck:
                i_tmin, i_begin, i_end = itranche.pop()

                #print "icheck (%d,%d) [%s,%s] " % (i_begin, i_end, begin, end ) 

                assert len(itranche) == len(tranche)
                assert i_tmin == tmin
                assert i_postorder[i_begin] is begin 
                assert i_postorder[i_end-1].next_ == end 
            pass


            #log.info("%s : start tranche %d begin %s end %s " % (self.pfx, tcount, begin, end))
            tcount += 1    
            assert tcount < 10

            p = begin

            if icheck:
               i_pindex = i_begin

            while p is not end:

                if icheck:
                    assert i_postorder[i_pindex] is p 
                    i_depth = p.cdepth
                    i_subNodes = Node.NumNodes( i_height, i_depth )
                    i_halfNodes = (i_subNodes - 1)/2
                pass

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
                        if self.debug:
                            print "CtrlLoopLeft %s " % p.l
                        pass
                        tminL = left.t + self.epsilon
                        if not p.l.is_leaf:
                            rhs.append(right)

                            tranche.append([tmin,p,None])  # should this be tminL ? its for continuation
                            tranche.append([tminL,Node.leftmost(p.l),p.l.next_])

                            if icheck:
                                itranche.append([tmin,  i_index, numInternalNodes ])
                                itranche.append([tminL, i_index - i_halfNodes*2, i_index - i_halfNodes ])  
                                print "icheck lhs %r " % itranche
                            pass
                            reiterate_ = True 
                        pass
                    elif ctrl == CtrlLoopRight: 

                        tminR = right.t + self.epsilon
                        if self.debug:
                            print "CtrlLoopRight %s " % p.r
                        pass
                        if not p.r.is_leaf:
                            lhs.append(left)
                            tranche.append([tmin,p,None])  # should this be tminR ?
                            tranche.append([tminR,Node.leftmost(p.r),p.r.next_])   
                            if icheck:
                                assert p.r.next_ is p  # next on right is always self
                                itranche.append([tmin,  i_index, numInternalNodes ])
                                itranche.append([tminR, i_index - i_halfNodes, i_index ])  
                                print "icheck rhs %r " % itranche
                            pass
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


                if icheck:
                    i_pindex += 1
                pass

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
 



if __name__ == '__main__':

    logformat = "%(asctime)s %(name)s %(levelname)-8s %(message)s"
    logging.basicConfig(level=logging.INFO,format=logformat)

    plt.rcParams['figure.figsize'] = 18,10.2 
    plt.ion()
    plt.close("all")

    #roots = trees
    roots = [root3, root4]
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
            if not Node.is_perfect_i(root):
                log.warning("skip imperfect tree %s " % root )
                continue  
            pass
            tsts.append(T(root,level=3,debug=[0], num=25, source="leaflight",origin=True,rayline=True, scale=0.1, sign=1))
        pass


    if 0: 
        t_root3a = T(root3, level=3, debug=[], num=100, source="aringlight" )
        t_root3b = T(root3, level=3, debug=[], num=100, source="origlight" )

        #t_root4a = T(root4, level=3, debug=[0], num=200, source="aringlight", notes="ok" )
        t_root4a = T(root4, level=3, debug=[0], num=20, source="leaflight", notes="ok", rayline=True, raytext=False, origin=True)
        t_root4b = T(root4, level=3, debug=[1], num=100, source="origlight", notes="pop from empty lhs" )
        tsts.append(t_root4a)
    pass
        
    # None: means use value from T ctor, or the default if not defined there
    source = None
    epsilon = None
    origin = None
    normal = None
    rayline = None

    for itst,tst in enumerate(tsts):
        csg = CSG(level=tst.level, epsilon=epsilon)
        partBuf = Node.serialize( tst.root )
        csg.ray = Ray()
        csg.iterative_intersect_slavish( partBuf )
 


    if 0:
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
                ax.axis('auto') 

            fig.suptitle(tst.suptitle, horizontalalignment='left', family='monospace', fontsize=10, x=0.1, y=0.99) 
            fig.show()

            i = tst.i


