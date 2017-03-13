#!/usr/bin/env python
"""
A slavish translation back to python from the CUDA/OptiX implementation 
of iterative CSG in intersect_boolean.h  in slavish_intersect and
a recursive riff on that in slavish_intersect_recursive.

Objectives:

* instrument code paths and make visual rep of them, especially 
  the non-bileaf looping ones that do the reiteration backtracking 
  
  * did not viz, but find for the standard perfect trees 
    that the recursive and iterative code paths are matching perfectly  

* extend code to handle partBuffer with empty nodes ie 
  levelorder serializations of non-perfect trees 

* extend to handle op-node transforms 

* check csg tree serialization/persisting and use to 
  bring python trees to GPU and vice versa

  * DONE: have brought tboolean source trees to use here with using GParts.py 
  * TODO: go from py to GPU too, more involved : needs container box and boundary hookups,
    the analytic PMT did this already, see how that worked 

* investigate pop empty errors seen with the OptiX imp

  * bringing the geometry to py did not reproduce the error, but in python are 
    just using rays in 2D, 
    perhaps persist the torch configuration too ? so can repeat the same py side
  
"""

import numpy as np
import logging
from opticks.ana.base import opticks_main

log = logging.getLogger(__name__)


import math
signbit_ = lambda f:math.copysign(1., f) < 0

def signbit(f):
    """
    In [9]: map(signbit_, [-3, -2.,-1.,-0.,0.,1.,2.,3])
    Out[9]: [True, True, True, True, False, False, False, False]
    """
    return signbit_(f)



from intersectTest import T

from opticks.bin.ffs import ffs_, clz_
from intersect import intersect_primitive
from node import Node, root3, root4, lrsph_u, trees, lrsph_d1, lrsph_d2
from node import Q0, Q1, Q2, Q3, X, Y, Z, W
from GParts import GParts

from opticks.optixrap.cu.boolean_solid_h import CTRL_RETURN_MISS, CTRL_RETURN_A, CTRL_RETURN_B, CTRL_RETURN_FLIP_B, CTRL_LOOP_A, CTRL_LOOP_B
from opticks.optixrap.cu.boolean_solid_h import ERROR_POP_EMPTY, ERROR_LHS_POP_EMPTY, ERROR_RHS_POP_EMPTY, ERROR_LHS_END_NONEMPTY, ERROR_RHS_END_EMPTY, ERROR_BAD_CTRL, ERROR_LHS_OVERFLOW, ERROR_RHS_OVERFLOW, ERROR_LHS_TRANCHE_OVERFLOW
from opticks.optixrap.cu.boolean_h import desc_state, desc_ctrl, Enter, Exit, Miss


LHS, RHS = 0, 1
MISS, LEFT, RIGHT, RFLIP = 0,1,2,3



def CSG_CLASSIFY(ise, dir_, tmin):
     assert ise.shape == (4,)
     #if ise[W] > tmin:
     if abs(ise[W]) > tmin:
         return Enter if np.dot( ise[:3], dir_ ) < 0 else Exit 
     else:
         return Miss


TRANCHE_STACK_SIZE = 4
CSG_STACK_SIZE = 16

class Error(Exception):
    def __init__(self, msg):
        super(Error, self).__init__(msg)


class Tranche(object):
    def __init__(self):
        self.tmin = np.zeros( (TRANCHE_STACK_SIZE), dtype=np.float32 )
        self.slice_  = np.zeros( (TRANCHE_STACK_SIZE), dtype=np.uint32 )
        self.curr = -1
        pass

    def push(self, slice_, tmin):
        if self.curr >= TRANCHE_STACK_SIZE - 1:
            raise Error("Tranche push overflow")
        else:
            self.curr += 1 
            self.tmin[self.curr] = tmin      
            self.slice_[self.curr] = slice_ 
        pass

    def pop(self):
        if self.curr < 0:
            raise Error("ERROR_POP_EMPTY")  
        pass
        slice_ = self.slice_[self.curr] 
        tmin = self.tmin[self.curr] ;  
        self.curr -= 1  
        return slice_, tmin  
 


class CSGD(object):
    def __init__(self):
        self.data = np.zeros( (CSG_STACK_SIZE, 4,4), dtype=np.float32 )
        self.idx = np.zeros( (CSG_STACK_SIZE,), dtype=np.uint32 )
        self.curr = -1
        pass

    def __repr__(self):
        return "csg  %2d:{%s} " % (self.curr, ",".join(map(lambda _:"%d" % _, self.nodes)))

    def pop(self):
        if self.curr < 0:
            raise Error("CSG_ pop from empty")
        else:
            data = self.data[self.curr] 
            idx = self.idx[self.curr] 
            self.curr -= 1 
        pass
        return data,idx

    nodes = property(lambda self:self.idx[0:self.curr+1])

    def push(self, ise, nodeIdx):
        if self.curr+1 >= CSG_STACK_SIZE:
            raise Error("CSG_ push overflow")
        else:
            self.curr += 1 
            self.data[self.curr] = ise        
            self.idx[self.curr] = nodeIdx 
        pass



class CSG_(object):
    def __init__(self):
        self.data = np.zeros( (CSG_STACK_SIZE, 4,4), dtype=np.float32 )
        self.curr = -1

    def pop(self):
        if self.curr < 0:
            raise Error("CSG_ pop from empty")
        else:
            ret = self.data[self.curr] 
            self.curr -= 1 
        pass
        return ret

    def push(self, ise):
        if self.curr+1 >= CSG_STACK_SIZE:
            raise Error("CSG_ push overflow")
        else:
            self.curr += 1 
            self.data[self.curr] = ise        
        pass


def TRANCHE_PUSH0( _stacku, _stackf, stack, valu, valf ):
    stack += 1 
    _stacku[stack] = valu 
    _stackf[stack] = valf
    return stack 

def TRANCHE_PUSH( _stacku, _stackf, stack, valu, valf ):
    if stack+1 >= TRANCHE_STACK_SIZE:
        raise Error("Tranche push overflow")
    else:
        stack += 1 
        _stacku[stack] = valu 
        _stackf[stack] = valf
    pass
    return stack 


def TRANCHE_POP0( _stacku, _stackf, stack):
    valu = _stacku[stack] 
    valf = _stackf[stack] 
    stack -= 1 
    return stack, valu, valf


POSTORDER_SLICE = lambda begin, end:( (((end) & 0xff) << 8) | ((begin) & 0xff)  )
POSTORDER_BEGIN = lambda tmp:( ((tmp) & (0xff << 0)) >> 0 )
POSTORDER_END   = lambda tmp:( ((tmp) & (0xff << 8)) >> 8 )

def test_POSTORDER():
    for begin in range(0x100):
        for end in range(0x100):

            tmp = POSTORDER_SLICE(begin, end, swap)
            begin2 = POSTORDER_BEGIN(tmp)
            end2 = POSTORDER_END(tmp)

            print "  %2x %2x -> %6x ->  %2x %2x " % (begin,end,tmp,begin2, end2)

            assert begin2 == begin
            assert end2 == end



POSTORDER_NODE = lambda postorder, i: (((postorder) & (0xF << (i)*4 )) >> (i)*4 )

TREE_HEIGHT = lambda numNodes:( ffs_((numNodes) + 1) - 2)
TREE_NODES = lambda height:( (0x1 << (1+(height))) - 1 )
TREE_DEPTH = lambda nodeIdx:( 32 - clz_((nodeIdx)) - 1 )

def test_tree():
    """
    ::

        In [421]: numNodes = [TREE_NODES(_) for _ in range(5)]
        In [422]: numNodes
        Out[422]: [1, 3, 7, 15, 31]
        In [423]: map(TREE_HEIGHT, numNodes)
        Out[423]: [0, 1, 2, 3, 4]
    """
    numNodes = [TREE_NODES(_) for _ in range(5)]
    print map(TREE_HEIGHT, numNodes)



# generated from /Users/blyth/opticks/optixrap/cu by boolean_h.py on Sat Mar  4 20:37:03 2017 
packed_boolean_lut_ACloser = [ 0x22121141, 0x00014014, 0x00141141, 0x00000000 ]
packed_boolean_lut_BCloser = [ 0x22115122, 0x00022055, 0x00133155, 0x00000000 ]
             
def boolean_ctrl_packed_lookup(operation, stateA, stateB, ACloser ):
    lut = packed_boolean_lut_ACloser if ACloser else  packed_boolean_lut_BCloser 
    offset = 3*stateA + stateB ;
    return (lut[operation] >> (offset*4)) & 0xf if offset < 8 else CTRL_RETURN_MISS 

propagate_epsilon = 1e-3





def log_info(msg):
    print msg 

def recursive_intersect(partBuffer, ray, tst):
    """
    """
    debug = tst.debug
    instrument = True
    numParts = len(partBuffer)  
    partOffset = 0 
    fullHeight = TREE_HEIGHT(numParts)
    height = fullHeight - 1
    numInternalNodes = TREE_NODES(height)

    def recursive_intersect_r(nodeIdx, tmin):
        """  
        :param nodeIdx: 1-based levelorder tree index
        """
        leftIdx = nodeIdx*2 
        rightIdx = nodeIdx*2 + 1
        bileaf = leftIdx > numInternalNodes

        isect = np.zeros( [4, 4, 4], dtype=np.float32 )
        tX_min = np.zeros( 2, dtype=np.float32 )
        tX_min[LHS] = tmin
        tX_min[RHS] = tmin
        x_state = np.zeros( 2, dtype=np.uint32 )

        if bileaf:
            isect[LEFT,0,W] = 0.
            isect[RIGHT,0,W] = 0.
            isect[LEFT] = intersect_primitive( Node.fromPart(partBuffer[partOffset+leftIdx-1]), ray, tX_min[LHS])
            isect[RIGHT] = intersect_primitive( Node.fromPart(partBuffer[partOffset+rightIdx-1]), ray, tX_min[RHS])
        else:
            isect[LEFT]  = recursive_intersect_r( leftIdx, tX_min[LHS] )
            isect[RIGHT] = recursive_intersect_r( rightIdx, tX_min[RHS] )
        pass

        x_state[LHS] = CSG_CLASSIFY( isect[LEFT][0], ray.direction, tX_min[LHS] )
        x_state[RHS] = CSG_CLASSIFY( isect[RIGHT][0], ray.direction, tX_min[RHS] )

        operation = partBuffer[nodeIdx-1,Q1,W].view(np.uint32)
        ctrl = boolean_ctrl_packed_lookup( operation, x_state[LHS], x_state[RHS], isect[LEFT][0][W] <= isect[RIGHT][0][W] )
        #if debug:log_info("%s :   %s   " % (pfx("R0",tst.iray,nodeIdx,bileaf), fmt(isect[LEFT],isect[RIGHT],x_state[LHS],x_state[RHS],ctrl)))

        if instrument:
            ray.addseq(ctrl)

        side = ctrl - CTRL_LOOP_A 
 
        loop = -1
        while side > -1 and loop < 10:
            loop += 1
            THIS = side + LEFT 
            tX_min[side] = isect[THIS][0][W] + propagate_epsilon 
            if bileaf:               
                isect[THIS] = intersect_primitive( Node.fromPart(partBuffer[partOffset+leftIdx+side-1]) , ray, tX_min[side])
            else:
                isect[THIS] = slavish_intersect_r( leftIdx+side, tX_min[side] )
            pass
            x_state[side] = CSG_CLASSIFY( isect[THIS][0], ray.direction, tX_min[side] )
            ctrl = boolean_ctrl_packed_lookup( operation, x_state[LHS], x_state[RHS], isect[LEFT][0][W] <= isect[RIGHT][0][W] )
            if instrument:
                ray.addseq(ctrl)
            pass
            side = ctrl - CTRL_LOOP_A ## NB possibly changed side            
        pass  

        isect[RFLIP] = isect[RIGHT]
        isect[RFLIP,0,X] = -isect[RFLIP,0,X]
        isect[RFLIP,0,Y] = -isect[RFLIP,0,Y]
        isect[RFLIP,0,Z] = -isect[RFLIP,0,Z]
 
        assert ctrl < CTRL_LOOP_A 

        # recursive return is equivalent to what gets popped ?
        if debug:log_info("%s :   %s   " % (pfx("R1",tst.iray,nodeIdx,bileaf), fmt(isect[LEFT],isect[RIGHT],x_state[LHS],x_state[RHS],ctrl)))
        return isect[ctrl]
    pass
    return recursive_intersect_r( 1, ray.tmin )




def f4(a,j=0,nfmt="%5.2f",tfmt="%7.3f"):
    assert a.shape == (4,4), a.shape
    return " ".join( map(lambda _:nfmt %  _, a[j][:3]) + [tfmt % a[j][3]]  )

def fmt(left,right,lst, rst, ctrl):
    assert left.shape == (4,4), left.shape
    assert right.shape == (4,4), right.shape
    return "L %s R %s   (%5s,%5s) -> %-20s " % ( f4(left), f4(right), desc_state(lst),desc_state(rst), desc_ctrl(ctrl))

def pfx(alg,iray,nodeIdx,bileaf):
    return "%s %6d [%2d] %2s " % (alg,iray,nodeIdx,"BI" if bileaf else "  ")

def stk(lhs, rhs):
    return "lhs %2d rhs %d " % (lhs.curr, rhs.curr) 



def evaluative_intersect(partBuffer, ray, tst):
    debug = tst.debug

    partOffset = 0 
    numParts = len(partBuffer)
    primIdx = -1

    fullHeight = TREE_HEIGHT(numParts)
    height = fullHeight - 1 
    numInternalNodes = TREE_NODES(height)       
    numNodes = TREE_NODES(fullHeight)      

    postorder_sequence = [ 0x1, 0x132, 0x1376254, 0x137fe6dc25ba498 ] 
    postorder = postorder_sequence[fullHeight] 


    tr = Tranche()
    tr.push( POSTORDER_SLICE(0, numNodes ), ray.tmin )

    csg = CSGD()
    csg.curr = -1

    tloop = -1 

    while tr.curr > -1:
        tloop += 1 
        assert tloop < 20   # wow root3 needs lots of loops

        slice_, tmin = tr.pop()
        begin = POSTORDER_BEGIN(slice_)
        end = POSTORDER_END(slice_)
        beginIdx = POSTORDER_NODE(postorder, begin)
        endIdx = POSTORDER_NODE(postorder, end-1)

        #if debug:log.info("%6d E : tranche begin %d end %d  (nodeIdx %d:%d)tmin %5.2f tloop %d  %r " % (tst.iray, begin, end, beginIdx, endIdx,  tmin, tloop, csg))

        i = begin
        while i < end:
            nodeIdx = POSTORDER_NODE(postorder, i)

            depth = TREE_DEPTH(nodeIdx)
            subNodes = TREE_NODES(fullHeight-depth) 
            halfNodes = (subNodes - 1)/2 
            primitive = nodeIdx > numInternalNodes 

            operation = partBuffer[nodeIdx-1,Q1,W].view(np.uint32)

            #print "(%d) depth %d subNodes %d halfNodes %d " % (nodeIdx, depth, subNodes, halfNodes )

            if primitive:
                isect = intersect_primitive( Node.fromPart(partBuffer[partOffset+nodeIdx-1]), ray, tmin)
                isect[0,W] = math.copysign(isect[0,W], -1. if nodeIdx % 2 == 0 else 1. )
                csg.push(isect,nodeIdx)
            else:
                if csg.curr < 1:
                   raise Error("ERROR_POP_EMPTY : csg.curr < 1 when need two items to combine")

                firstLeft = signbit(csg.data[csg.curr,0,W])
                secondLeft = signbit(csg.data[csg.curr-1,0,W])

                if not firstLeft ^ secondLeft:
                    raise Error("ERROR_XOR_SIDE")

                left  = csg.curr if firstLeft else csg.curr - 1
                right = csg.curr-1 if firstLeft else csg.curr 

                l_state = CSG_CLASSIFY( csg.data[left,0], ray.direction, tmin )
                r_state = CSG_CLASSIFY( csg.data[right,0], ray.direction, tmin )

                t_left = abs(csg.data[left,0,W])
                t_right = abs(csg.data[right,0,W])

                ctrl = boolean_ctrl_packed_lookup( operation, l_state, r_state, t_left <= t_right  )
                ray.addseq(ctrl)

                if debug:log_info("%s :   %s   " % (pfx("E1",tst.iray,nodeIdx,None), fmt(csg.data[left],csg.data[right],l_state,r_state,ctrl)))

                UNDEFINED = 0
                CONTINUE = 1
                BREAK = 2

                if ctrl < CTRL_LOOP_A:
                    result = np.zeros((4,4), dtype=np.float32)
                    if not ctrl == CTRL_RETURN_MISS:
                        result[:] = csg.data[left if ctrl == CTRL_RETURN_A else right]
                    pass
                    if ctrl == CTRL_RETURN_FLIP_B:
                        result[0,X] = -result[0,X]
                        result[0,Y] = -result[0,Y]
                        result[0,Z] = -result[0,Z]
                    pass
                    result[0,W] = math.copysign( result[0,W], -1. if nodeIdx %2 == 0 else 1.)

                    if debug:log_info("%s :   %s " % (pfx("E2",tst.iray,nodeIdx, None), f4(result) ))


                    csg.pop()
                    csg.pop()
                    csg.push(result, nodeIdx)

                    act = CONTINUE
                else:
                    loopside = left if ctrl == CTRL_LOOP_A else right 
                    otherside = right if ctrl == CTRL_LOOP_A else left 
 
                    leftIdx = 2*nodeIdx
                    rightIdx = leftIdx + 1
                    otherIdx = rightIdx if ctrl == CTRL_LOOP_A else leftIdx

                    tminAdvanced = abs(csg.data[loopside,0,W]) + propagate_epsilon 

                    other = np.zeros((4,4), dtype=np.float32)  #  need tmp as pop about to invalidate indices
                    other[:] = csg.data[otherside]

                    csg.pop()
                    csg.pop()
                    csg.push( other, otherIdx ) 
 
                    endTree   = POSTORDER_SLICE(i, end )   # fix numNodes -> end
                    leftTree  = POSTORDER_SLICE(i-2*halfNodes, i-halfNodes)
                    rightTree = POSTORDER_SLICE(i-halfNodes, i) 
                    loopTree  = leftTree if ctrl == CTRL_LOOP_A else rightTree

                    tr.push(endTree, tmin) 
                    tr.push(loopTree, tminAdvanced) 
                    
                    act = BREAK 
                pass     # return or "recursive" call
                if act == BREAK:
                    break 
                pass
            pass
            i += 1   # next postorder node in the tranche
        pass         # end traversal loop
    pass             # end tranch loop
    assert csg.curr == 0, csg.curr
    ret = csg.data[0]
    assert ret.shape == (4,4)
    ret[0,W] = abs(ret[0,W])
    return ret




def iterative_intersect(partBuffer, ray, tst):
    """  
    For following code paths its simpler to instrument rays, not intersects
    as isects keep getting created, pushed, popped, etc..
    """
    debug = tst.debug
    postorder_sequence = [ 0x1, 0x132, 0x1376254, 0x137fe6dc25ba498 ] 
    ierr = 0
    abort_ = False
    instrument = True 

    partOffset = 0 
    numParts = len(partBuffer)
    primIdx = -1

    fullHeight = ffs_(numParts + 1) - 2 
    height = fullHeight - 1 

    postorder = postorder_sequence[height] 
    numInternalNodes = (0x1 << (1+height)) - 1       

    _tmin = np.zeros( (TRANCHE_STACK_SIZE), dtype=np.float32 )
    _tranche = np.zeros( (TRANCHE_STACK_SIZE), dtype=np.uint32 )
    tranche = -1

    csg_ = [CSG_(), CSG_()] 
    lhs = csg_[LHS] 
    rhs = csg_[RHS] 
    lhs.curr = -1
    rhs.curr = -1

    isect = np.zeros( [4, 4, 4], dtype=np.float32 )

    tranche = TRANCHE_PUSH0( _tranche, _tmin, tranche, POSTORDER_SLICE(0, numInternalNodes), ray.tmin )


    while tranche >= 0:
        tranche, tmp, tmin = TRANCHE_POP0( _tranche, _tmin,  tranche )
        begin = POSTORDER_BEGIN(tmp)
        end = POSTORDER_END(tmp)

        i = begin
        while i < end:

            nodeIdx = POSTORDER_NODE(postorder, i)
            leftIdx = nodeIdx*2 
            rightIdx = nodeIdx*2 + 1
            depth = TREE_DEPTH(nodeIdx)
            subNodes = (0x1 << (1+height-depth)) - 1
            halfNodes = (subNodes - 1)/2 
            bileaf = leftIdx > numInternalNodes

            operation = partBuffer[nodeIdx-1,Q1,W].view(np.uint32)

            tX_min = np.zeros( 2, dtype=np.float32 )
            tX_min[LHS] = tmin
            tX_min[RHS] = tmin

            x_state = np.zeros( 2, dtype=np.uint32 )

            if bileaf:
                isect[LEFT,0,W] = 0.
                isect[RIGHT,0,W] = 0.
                isect[LEFT] = intersect_primitive( Node.fromPart(partBuffer[partOffset+leftIdx-1]), ray, tX_min[LHS])
                isect[RIGHT] = intersect_primitive( Node.fromPart(partBuffer[partOffset+rightIdx-1]), ray, tX_min[RHS])
            else:
                try:
                    isect[LEFT] = lhs.pop()
                except Error:
                    log_info("%s : ERROR_LHS_POP_EMPTY : ABORT  " % (pfx("I0",tst.iray,nodeIdx,bileaf)))
                    ierr |= ERROR_LHS_POP_EMPTY
                    abort_ = True
                    break
                pass
                try:
                    isect[RIGHT] = rhs.pop()
                except Error:
                    log_info("%s : ERROR_RHS_POP_EMPTY : ABORT  " % (pfx("I0",tst.iray,nodeIdx,bileaf)))
                    ierr |= ERROR_RHS_POP_EMPTY
                    abort_ = True
                    break
                pass
            pass


            x_state[LHS] = CSG_CLASSIFY( isect[LEFT][0], ray.direction, tX_min[LHS] )
            x_state[RHS] = CSG_CLASSIFY( isect[RIGHT][0], ray.direction, tX_min[RHS] )
            ctrl = boolean_ctrl_packed_lookup( operation, x_state[LHS], x_state[RHS], isect[LEFT][0][W] <= isect[RIGHT][0][W] )

            if debug:log_info("%s :   %s   " % (pfx("I0",tst.iray,nodeIdx,bileaf), fmt(isect[LEFT],isect[RIGHT],x_state[LHS],x_state[RHS],ctrl)))
            if instrument:
                ray.addseq(ctrl)
            pass

            reiterate = False
            if ctrl < CTRL_LOOP_A:
                ## recursive return after first classify, needs an avenue to push 
                pass
            else:
                side = ctrl - CTRL_LOOP_A 
                loop = -1

                while side > -1 and loop < 10:
                    if debug:log_info("%s :  side %s loop %d " % (pfx("I",tst.iray,nodeIdx,bileaf), side, loop )) 
                    loop += 1
                    THIS = side + LEFT 
                    tX_min[side] = isect[THIS][0][W] + propagate_epsilon 

                    if bileaf:               
                        isect[THIS] = intersect_primitive( Node.fromPart(partBuffer[partOffset+leftIdx+side-1]) , ray, tX_min[side])
                    else:
                        try:
                            isect[THIS] = csg_[side].pop()
                        except Error:
                            log_info("%s : ERROR_POP_EMPTY : ABORT : %s " % (pfx("I0",tst.iray,nodeIdx,bileaf),stk(csg_[LHS],csg_[RHS])))
                            ierr |= ERROR_POP_EMPTY
                            abort_ = True
                            break
                        pass
                    pass

                    x_state[side] = CSG_CLASSIFY( isect[THIS][0], ray.direction, tX_min[side] )
                    ctrl = boolean_ctrl_packed_lookup( operation, x_state[LHS], x_state[RHS], isect[LEFT][0][W] <= isect[RIGHT][0][W] )
                    if debug:log_info("%s :   %s   " % (pfx("I1",tst.iray,nodeIdx,bileaf), fmt(isect[LEFT],isect[RIGHT],x_state[LHS],x_state[RHS],ctrl)))

                    if instrument:
                        ray.addseq(ctrl)
                    pass
                    side = ctrl - CTRL_LOOP_A    ## NB possibly changed side

                    if side > -1 and not bileaf:
                        other = 1 - side
                        tX_min[side] = isect[THIS][0][W] + propagate_epsilon 
                        csg_[other].push( isect[other+LEFT] ) 

                        leftTree = POSTORDER_SLICE(i-2*halfNodes, i-halfNodes) 
                        rightTree = POSTORDER_SLICE(i-halfNodes, i) 
                        endTree = POSTORDER_SLICE(i, end)  # FIX numInternalNodes -> end
                        sideTree = leftTree if side == LHS else rightTree
            
                        tranche = TRANCHE_PUSH( _tranche, _tmin, tranche, endTree, tmin )
                        tranche = TRANCHE_PUSH( _tranche, _tmin, tranche, sideTree, tX_min[side] )
                        reiterate = True
                        break
                    pass
                pass   # side loop
            pass

            if reiterate or abort_:
                break
            pass

            isect[RFLIP] = isect[RIGHT]
            isect[RFLIP,0,X] = -isect[RFLIP,0,X]
            isect[RFLIP,0,Y] = -isect[RFLIP,0,Y]
            isect[RFLIP,0,Z] = -isect[RFLIP,0,Z]
 
            assert ctrl < CTRL_LOOP_A 
            result = isect[ctrl] 
            nside = LHS if nodeIdx % 2 == 0 else RHS  # even on left

            csg_[nside].push(result)



            i += 1   # next postorder node in the tranche
        pass         # end traversal loop
        if abort_:break  
    pass             # end tranch loop


    LHS_curr = csg_[LHS].curr
    RHS_curr = csg_[RHS].curr 

    ierr |=  ERROR_LHS_END_NONEMPTY if LHS_curr != -1 else 0
    ierr |=  ERROR_RHS_END_EMPTY    if RHS_curr != 0 else 0

    assert RHS_curr == 0 and ierr == 0
    assert RHS_curr == 0, RHS_curr

    ret = csg_[RHS].data[0]
    assert ret.shape == (4,4)
    ret[0,W] = abs(ret[0,W])
    return ret





if __name__ == '__main__':
#if 0:
    args = opticks_main(doc=__doc__) 

    from nodeRenderer import Renderer
    import matplotlib.pyplot as plt

    plt.rcParams['figure.figsize'] = 18,10.2 
    plt.ion()
    plt.close("all")

    #roots = [lrsph_d1, lrsph_u]
    #roots = [lrsph_u]
    #roots = [lrsph_d1]
    roots = trees
    #roots = [root3]
    #roots = ["$TMP/tboolean-csg-two-box-minus-sphere-interlocked"]
    #roots = ["$TMP/tboolean-csg-four-box-minus-sphere"]

 
    skips = []
    tsts = []
    disc = []

    if 1:
        for root in roots:
            if type(root) is str:
                source = "ringlight,origlight"  # TODO: support partBuf extraction of prim nodes in leaflight
            else:
                if not Node.is_perfect_i(root):
                    log.warning("skip imperfect tree %s " % root )
                    continue  
                pass
                if root.name in skips:
                    log.warning("skipping tree %s " % root )
                    continue  
                pass
                source = "leaflight"
            pass
            tsts.append(T(root,level=3,irays=disc, num=100, source=source,origin=True,rayline=True, scale=0.1, sign=1))
        pass
    pass

    #irays = [785,1972,3546,7119,7325,8894]
    #irays_ok = [785,7119,7325]
    #irays_nok = [1972,3546,8894]
    #irays = irays_nok
    #tsts.append(T("$TMP/tboolean-csg-four-box-minus-sphere", seed=0, num=10000, source="randbox",irays=irays,origin=True, rayline=True,scale=0.1, sign=1))


    for tst in tsts:
        root = tst.root
        #log.info("tst %s " % tst.name )

        if type(root) is Node:        
            partBuffer = Node.serialize( root )
        elif type(root) is str:
            gp = GParts(root)
            partBuffer = gp[1]
        else:
            assert 0, (root, type(root)) 
        pass

        imps = {
           "evaluative":lambda ray,tst:evaluative_intersect(partBuffer, ray,tst),
           "iterative":lambda ray,tst:iterative_intersect(partBuffer, ray,tst),
           "recursive":lambda ray,tst:recursive_intersect(partBuffer, ray,tst)
        }

        keys = "evaluative recursive iterative".split()

        tst.run( imps, keys=keys)
        tst.compare()

        nx = len(keys)
        fig = plt.figure()

        axs = {}
        for k,key in enumerate(keys):
            axs[key] = fig.add_subplot(1,nx,k+1, aspect='equal')
        pass

        tst.plot_intersects( axs=axs, keys=keys)

        if type(root) is Node:
            for key in keys:
                ax = axs[key]
                rdr = Renderer(ax)
                rdr.render(tst.root)
                #rdr.limits(400,400)
                ax.axis('auto')
                ax.set_xlabel(key)
            pass
        else:
            pass  # TODO: support rendering basis shapes from partBuffer

        fig.suptitle(tst.suptitle, horizontalalignment='left', family='monospace', fontsize=10, x=0.1, y=0.99) 
        fig.show()
    pass


    i = tst.i
    r0 = tst.rays[0]
    r1 = tst.rays[1]


 
