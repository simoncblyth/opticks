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

"""
import logging
log = logging.getLogger(__name__)

import numpy as np

from node import Node, root0, root1, root2, root3, root4, UNION, INTERSECTION, DIFFERENCE, BOX, SPHERE, EMPTY, desc
from ctrl import CtrlLeft, CtrlRight, CtrlResumeFromLeft, CtrlResumeFromRight, CtrlReturn, CtrlBreak, desc_ctrl


def fake_pfx(ctrl):
    if ctrl & CtrlLeft and ctrl & CtrlRight:
        pfx = ""
    elif ctrl & CtrlLeft:
        pfx = "L"
    elif ctrl & CtrlRight:
        pfx = "R"
    elif ctrl & CtrlResumeFromLeft:
        pfx = "RL"
    elif ctrl & CtrlResumeFromRight:
        pfx = "RR"
    else:
        pfx = desc_ctrl(ctrl)
    pass
    return pfx

def fake_ctrl(p):
    if hasattr(p, "LoopL"): 
        log.info("fake_ctrl found node with LoopL %r " % p ) 
        delattr(p, "LoopL")
        ctrl = CtrlLeft
    else:
        ctrl = CtrlReturn
    pass
    return ctrl

def fake_binary_calc(node, left=None, right=None, ctrl=None):
    assert hasattr(node,'depth')
    assert left and right, (left, right)
    return "%s:[%s;%s](%s,%s)" % ( fake_pfx(ctrl),node.idx, node.depth, left, right )

def fake_primitive_calc(node, ctrl=None):
    return "%s:%s;%s" % (fake_pfx(ctrl),node.idx, node.depth )





def postordereval_i2t(root, debug=0): 
    """
    Iterative binary tree evaluation, using postorder threading to avoid 
    the stack manipulations of _i2 that repeatly "discover" the postorder.
    However intermediate evaluation steps still require 
    lhs and rhs stacks, that grow to a maximum of one less than the tree height.
    ie the stacks are small

    * NB this assumes a COMPLETE BINARY TREE, ie every node above the leaves has
      non None left and right children, and leaves have left and right None

    * note that the postorder traversal is operator only starting from bottom left
      with bileaf nodes at lowest level, ie operator nodes with left and right 
      primitives  

    * NB initial traverse always starts on a bileaf operator
      so p.l and p.r are leaves and lhs/rhs stacks will be appended
      before reaching operator where they will be popped  

    * non-bileaf LoopL/R forces reiteration of left/right subtree, 
      so push er/el onto rhs/lhs stack, as was just popped but the 
      so will return to same position after the reiteration
    
    * non-bileaf LoopL/R just popped lhs and rhs, but looping means reiterating 
      while leaving the other side unchanged, so must push to opposite side
      so when resume after the reiteration are back to the same other side state 
      as before it
    
    * bileaf loopers with p.l or p.r leaves just 
      go for an immediate while loop spin repeating one side
      primitive_calc with different act, they do not tee up tranches 
   
    * non-bileaf loopers have to tee up pair of tranches to repeat the loopside subtree
      and then resume from where left off

    Three while loop structure

    * begin/end tranches  
    * postorder operator nodes between begin and end
    * act controlled inner loop, repeats calc for immediate (bileaf) loopers
 
    """
    leftop = Node.leftmost(root)
    assert leftop.is_bileaf 
    assert not leftop.is_primitive
    assert leftop.next_ is not None, "threaded postorder requires Node.postorder_threading_r "

    debug = 1
    lhs = []
    rhs = []

    tranche = []
    tranche.append([leftop,None,CtrlLeft|CtrlRight])

    while len(tranche) > 0:
        begin, end, ctrl = tranche.pop()
        #print "start tranche %s begin %s end %s " % (desc_ctrl(ctrl), begin, end) 
        
        if debug > 3:
            p = begin
            while p is not end:
                print "pre-traverse ", p
                p = p.next_ 
            pass
        pass
        p = begin

        while p is not end:
            #ctrl = ctrl & ~CtrlReturn 
            ctrl = (CtrlLeft|CtrlRight)

            #print "p loop %s : %s " % (desc_ctrl(ctrl), p)

            while ctrl & (CtrlLeft | CtrlRight ):

                if ctrl & CtrlLeft: 
                    if p.l.is_leaf:
                        el = fake_primitive_calc(p.l,ctrl=ctrl) 
                        assert el
                    else:
                        el = lhs.pop()
                    pass
                pass

                if ctrl & CtrlRight: 
                    if p.r.is_leaf:
                        er = fake_primitive_calc(p.r,ctrl=ctrl) 
                        assert er
                    else:
                        er = rhs.pop()
                    pass
                pass

                ep = fake_binary_calc(p,el,er,ctrl=ctrl)

                ctrl = fake_ctrl(p)

                if ctrl & CtrlLeft: 
                    if not p.l.is_leaf:
                        rhs.append(er)
                        tranche.append([p,None,CtrlResumeFromLeft])  
                        tranche.append([Node.leftmost(p.l),p.l.next_,CtrlLeft])
                        ctrl |= CtrlBreak
                    pass
                pass
                if ctrl & CtrlRight: 
                    if not p.r.is_leaf:
                        lhs.append(el)
                        tranche.append([p,None,CtrlResumeFromRight])  
                        tranche.append([Node.leftmost(p.r),p.r.next_,CtrlRight])
                        ctrl |= CtrlBreak
                    pass
                pass
                if ctrl & CtrlBreak:
                    break
                pass
            pass  

            if ctrl & CtrlBreak:
                ctrl = ctrl & ~CtrlBreak 
                if debug > 0:
                    log.info("_i2t post ctrl-while (after scrubbed CtrlBreak): %s " % desc_ctrl(ctrl))
                break
            pass

            if p.is_left:
                lhs.append(ep)
            else:
                rhs.append(ep)
            pass

            p = p.next_ 
        pass
    pass
    assert len(lhs) == 0, lhs
    assert len(rhs) == 1, rhs   # end with p.idx = 1 for the root
    return rhs[0]
 


def postordereval_r(p, ctrl=CtrlLeft|CtrlRight, debug=0):
    """
    * CtrlLeft CtrlRight distinction only has teeth at 
      the single recursion level, the ctrl is passed to
      other levels for tree annotation purposes but its
      power is removed at the sub-levels via ctrl|CtrlBoth

    * note that when a CtrlLeft or CtrlRight is raised the 
      current node and its subtree gets repeated

    """
    assert p
    el, er, ep = None, None, None
    debug = 0

    xctrl = ctrl

    loopcount = 0 
    while ctrl & (CtrlLeft | CtrlRight):

        loopcount += 1
        assert loopcount < 10 

        if ctrl & CtrlLeft: 
            if p.l.is_leaf:
                el = fake_primitive_calc(p.l, ctrl=ctrl) 
            else:
                el = postordereval_r(p.l, ctrl=ctrl|CtrlRight, debug=debug) 
            pass
        pass

        if ctrl & CtrlRight: 
            if p.r.is_leaf:
                er = fake_primitive_calc(p.r, ctrl=ctrl) 
            else:
                er = postordereval_r(p.r, ctrl=ctrl|CtrlLeft, debug=debug) 
            pass
        pass

        ep = fake_binary_calc(p, el, er, xctrl)
        ctrl = fake_ctrl(p)

        if ctrl & CtrlLeft:
            xctrl = CtrlResumeFromLeft
        elif ctrl & CtrlRight:
            xctrl = CtrlResumeFromRight
        else:
            xctrl = ctrl
        pass
    pass
    assert ep
    return ep






if __name__ == '__main__':

    logformat = "%(asctime)s %(name)s %(levelname)-8s %(message)s"
    logging.basicConfig(level=logging.INFO,format=logformat)

    root = root2
    root.tree_labelling()
    Node.dress(root)

    ret0 = None
    fns = [postordereval_r,postordereval_i2t]

    for fn in fns:
        Node.label_r(root, 2, "LoopL")   # label is popped, so relabel for each imp

        ret = fn(root, debug=0) 
        print "%20s : %s " % ( fn.__name__, ret )
        if ret0 is None:
            ret0 = ret
        else:
            assert ret == ret0, (ret, ret0)
        pass
    pass




