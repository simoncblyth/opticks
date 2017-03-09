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
     if ise[W] > tmin:
         return Enter if np.dot( ise[:3], dir_ ) < 0 else Exit 
     else:
         return Miss


TRANCHE_STACK_SIZE = 4
CSG_STACK_SIZE = 16

class Error(Exception):
    def __init__(self, msg):
        super(Error, self).__init__(msg)



class CSGD(object):
    def __init__(self):
        self.data = np.zeros( (CSG_STACK_SIZE, 4,4), dtype=np.float32 )
        self.idx = np.zeros( (CSG_STACK_SIZE,), dtype=np.uint32 )
        self.curr = -1

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


POSTORDER_SLICE = lambda begin, end, swap:( (((swap) & 0xff) << 16) | (((end) & 0xff) << 8) | ((begin) & 0xff)  )
POSTORDER_BEGIN = lambda tmp:( ((tmp) & (0xff << 0)) >> 0 )
POSTORDER_END   = lambda tmp:( ((tmp) & (0xff << 8)) >> 8 )
POSTORDER_SWAP  = lambda tmp:( ((tmp) & (0xff << 16)) >> 16 )

def test_POSTORDER():
    for swap in range(0x2):
        for begin in range(0x100):
            for end in range(0x100):

                tmp = POSTORDER_SLICE(begin, end, swap)
                swap2 = POSTORDER_SWAP(tmp)
                begin2 = POSTORDER_BEGIN(tmp)
                end2 = POSTORDER_END(tmp)

                print "%d %2x %2x -> %6x -> %d %2x %2x " % (swap,begin,end,tmp, swap2, begin2, end2)

                assert begin2 == begin
                assert end2 == end
                assert swap2 == swap



POSTORDER_NODE = lambda postorder, i: (((postorder) & (0xF << (i)*4 )) >> (i)*4 )



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


def slavish_intersect_recursive(partBuffer, ray, tst):
    """
    """
    debug = tst.debug
    instrument = True
    numParts = len(partBuffer)  
    partOffset = 0 
    fullHeight = ffs_(numParts + 1) - 2
    height = fullHeight - 1
    numInternalNodes = (0x1 << (1+height)) - 1

    def slavish_intersect_r(nodeIdx, tmin):
        """  
        :param nodeIdx: 1-based levelorder tree index

        For supporting partial trees, 

        * bileaf, must be based on partBuffer contents not leftIdx value

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
            isect[LEFT]  = slavish_intersect_r( leftIdx, tX_min[LHS] )
            #if debug and nodeIdx == 1:
            #    log_info("%s : initial left subtree recursion completed" % pfx("R0",tst.iray,nodeIdx,bileaf))

            isect[RIGHT] = slavish_intersect_r( rightIdx, tX_min[RHS] )
            #if debug and nodeIdx == 1:
            #    log_info("%s : initial right subtree recursion completed" % pfx("R0",tst.iray,nodeIdx,bileaf))
        pass
        # NB left and right subtree recursions complete before classification can happen at this level
        # ie the above block of code is repeated in every stack instance until terminations are 
        # reached and the unwind gets back to here 
       
        #if debug:
        #    log_info("%s : middle  reached " % pfx("R0",tst.iray,nodeIdx,bileaf))


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
    return slavish_intersect_r( 1, ray.tmin )



def f4(a,j=0,nfmt="%5.2f",tfmt="%7.3f"):
    return " ".join( map(lambda _:nfmt %  _, a[j][:3]) + [tfmt % a[j][3]]  )

def fmt(left,right,lst, rst, ctrl):
    return "L %s R %s   (%5s,%5s) -> %-20s " % ( f4(left), f4(right), desc_state(lst),desc_state(rst), desc_ctrl(ctrl))

def pfx(alg,iray,nodeIdx,bileaf):
    return "%s %6d [%2d] %2s " % (alg,iray,nodeIdx,"BI" if bileaf else "  ")

def stk(lhs, rhs):
    return "lhs %2d rhs %d " % (lhs.curr, rhs.curr) 



def evaluate(operation, isect, ray, tmin):
    l_state = CSG_CLASSIFY( isect[LEFT][0], ray.direction, tmin )
    r_state = CSG_CLASSIFY( isect[RIGHT][0], ray.direction, tmin )
    ctrl = boolean_ctrl_packed_lookup( operation, l_state, r_state, isect[LEFT][0][W] <= isect[RIGHT][0][W] )
    if ctrl < CTRL_LOOP_A:
        isect[RFLIP] = isect[RIGHT]
        isect[RFLIP,0,X] = -isect[RFLIP,0,X]
        isect[RFLIP,0,Y] = -isect[RFLIP,0,Y]
        isect[RFLIP,0,Z] = -isect[RFLIP,0,Z]
    pass
    return ctrl, l_state, r_state


def evaluative_intersect(partBuffer, ray, tst):
    debug = tst.debug
    postorder_sequence = [ 0x1, 0x132, 0x1376254, 0x137fe6dc25ba498 ] 

    partOffset = 0 
    numParts = len(partBuffer)
    primIdx = -1

    fullHeight = ffs_(numParts + 1) - 2 
    height = fullHeight - 1 
    numInternalNodes = (0x1 << (1+height)) - 1       
    numNodes = (0x1 << (1+fullHeight)) - 1       

    postorder = postorder_sequence[fullHeight] 

    if debug:log_info("%s %d ray.tmin %5.2f postorder %8x " % ("EO",tst.iray, ray.tmin, postorder))

    _tmin = np.zeros( (TRANCHE_STACK_SIZE), dtype=np.float32 )
    _tranche = np.zeros( (TRANCHE_STACK_SIZE), dtype=np.uint32 )
    tranche = -1
    isect = np.zeros( [4, 4, 4], dtype=np.float32 )

    csg = CSGD()
    csg.curr = -1

    tranche = TRANCHE_PUSH0( _tranche, _tmin, tranche, POSTORDER_SLICE(0, numNodes,0), ray.tmin )
    tloop = -1 
    while tranche >= 0:
        tloop += 1 
        assert tloop < 20   # wow root3 needs lots of loops
        tranche, tmp, tmin = TRANCHE_POP0( _tranche, _tmin,  tranche )
        begin = POSTORDER_BEGIN(tmp)
        end = POSTORDER_END(tmp)
        swap = POSTORDER_SWAP(tmp)

        if debug:log.info("%6d E : tranche begin %d end %d swap %d (nodeIdx %d:%d)tmin %5.2f tloop %d  %r " % (tst.iray, begin, end, swap, POSTORDER_NODE(postorder,begin), POSTORDER_NODE(postorder,end),  tmin, tloop, csg))

        i = begin
        while i < end:
            nodeIdx = POSTORDER_NODE(postorder, i)
            depth = 32 - clz_(nodeIdx)-1
            subNodes = (0x1 << (1+fullHeight-depth)) - 1
            halfNodes = (subNodes - 1)/2 
            primitive = nodeIdx > numInternalNodes 
            operation = partBuffer[nodeIdx-1,Q1,W].view(np.uint32)

            #print "(%d) depth %d subNodes %d halfNodes %d " % (nodeIdx, depth, subNodes, halfNodes )

            if primitive:
                ise = intersect_primitive( Node.fromPart(partBuffer[partOffset+nodeIdx-1]), ray, tmin)
                csg.push(ise,nodeIdx)
                #print "(%2d) prim-push  %r " % (nodeIdx, csg)
            else:
                print "(%2d) bef-op-pop %r " % (nodeIdx, csg)

                a,a_idx = csg.pop()
                b,b_idx = csg.pop()

                #a_left = a_idx % 2 == 0          
                #b_left = b_idx % 2 == 0          
                #assert a_left ^ b_left, (a_left, b_left, a_idx, b_idx)
                #isect[LEFT][:]  = b if b_left else a 
                #isect[RIGHT][:] = a if b_left else b 
                
                if a_idx < b_idx:
                    isect[LEFT][:]  = a
                    isect[RIGHT][:]  = b
                else:
                    isect[LEFT][:]  = b
                    isect[RIGHT][:]  = a
                pass

                # can tranche swap bit avoid keeping nodeIdx in stack for all isects ?
                # it doesnt look like it 
                #if swap == 0:
                #    # normally postorder means that left subtree gets pushed before right
                #    # so pop order normally right then left, but loop-left means that 
                #    # the unchanged right gets pushed before the left subtree completes so order gets reversed
                #    assert np.all(isect[RIGHT] == a) 
                #    assert np.all(isect[LEFT] == b) 
                #else:  
                #    assert np.all(isect[RIGHT] == b), (isect[RIGHT], b )
                #    assert np.all(isect[LEFT] == a) 
                #pass
                #print "(%2d) eval after pop %r " % (nodeIdx, csg )

                
                # hmm keeping nodeIdx with the isects means evaluate could directly look at the csg stack
                # and determine the ctrl prior to moving any data
                #  
                # 
                ctrl,l_state,r_state = evaluate(operation, isect, ray, tmin )

                #if debug:log_info("%s :   %s   " % (pfx("E0",tst.iray,nodeIdx,primitive), fmt(isect[LEFT],isect[RIGHT],l_state,r_state,ctrl)))
                ray.addseq(ctrl)
 
                if ctrl < CTRL_LOOP_A:
                    csg.push(isect[ctrl], nodeIdx)
                    print "(%2d) after return push  %r " % (nodeIdx, csg)
                else:
                    loopside = ctrl - CTRL_LOOP_A   
                    otherside = 1 - loopside

                    csg.push(isect[LEFT+otherside], 2*nodeIdx+otherside )

                    tminAdvanced = isect[LEFT+loopside][0][W] + propagate_epsilon 

                    onwards  = POSTORDER_SLICE(i, numNodes,1 if loopside == LHS else 0)
                    sideTree = POSTORDER_SLICE(i-2*halfNodes, i-halfNodes, 0) if loopside == LHS else POSTORDER_SLICE(i-halfNodes, i, 0) 

                    tranche = TRANCHE_PUSH( _tranche, _tmin, tranche, onwards, tmin )
                    tranche = TRANCHE_PUSH( _tranche, _tmin, tranche, sideTree, tminAdvanced )
                    break
                pass
            pass
            i += 1   # next postorder node in the tranche
        pass         # end traversal loop
    pass             # end tranch loop
    if csg.curr != 0:
        print "ERROR over csg %d " % csg.curr
    return csg.data[0]





def slavish_intersect(partBuffer, ray, tst):
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

    tranche = TRANCHE_PUSH0( _tranche, _tmin, tranche, POSTORDER_SLICE(0, numInternalNodes,0), ray.tmin )


    while tranche >= 0:
        tranche, tmp, tmin = TRANCHE_POP0( _tranche, _tmin,  tranche )
        begin = POSTORDER_BEGIN(tmp)
        end = POSTORDER_END(tmp)
        if debug:log.info("%6d I : tranche begin %d end %d " % (tst.iray, begin, end))

        i = begin
        while i < end:

            nodeIdx = POSTORDER_NODE(postorder, i)
            leftIdx = nodeIdx*2 
            rightIdx = nodeIdx*2 + 1
            depth = 32 - clz_(nodeIdx)-1
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
                        sideTree = POSTORDER_SLICE(i-2*halfNodes, i-halfNodes,1) if side == LHS else POSTORDER_SLICE(i-halfNodes, i,0) 
                        tranche = TRANCHE_PUSH( _tranche, _tmin, tranche, POSTORDER_SLICE(i, numInternalNodes,0), tmin )
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


    if RHS_curr == 0 and ierr == 0:
        ret = csg_[RHS].data[RHS_curr]
    else:
        ret = None ## hmm need some kind error holding ret
    pass
    if ierr is not 0:
        print "ierr: 0x%.8x tst.iray:%6d " % (ierr, tst.iray)
    pass
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
    #roots = trees
    roots = [root3]
    #roots = ["$TMP/tboolean-csg-two-box-minus-sphere-interlocked"]
    #roots = ["$TMP/tboolean-csg-four-box-minus-sphere"]

    skips = []
    tsts = []

    if 0:
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
            tsts.append(T(root,level=3,debug=[0], num=100, source=source,origin=True,rayline=True, scale=0.1, sign=1))
        pass
    pass

    irays = [785,1972,3546,7119,7325,8894]
    #irays_ok = [785,7119,7325]
    #irays_nok = [1972,3546,8894]
    #irays = irays_nok
    tsts.append(T("$TMP/tboolean-csg-four-box-minus-sphere", seed=0, num=10000, source="randbox",irays=irays,origin=True, rayline=True,scale=0.1, sign=1))


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

        evaluative_ = lambda ray,tst:evaluative_intersect(partBuffer, ray, tst) 
        iterative_ = lambda ray,tst:slavish_intersect(partBuffer, ray, tst) 
        recursive_ = lambda ray,tst:slavish_intersect_recursive(partBuffer, ray, tst) 
        rr = [1,0]
        #tst.run( [iterative_, recursive_], rr)
        tst.run( [evaluative_, recursive_], rr)
        tst.compare()

        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1, aspect='equal')
        ax2 = fig.add_subplot(1,2,2, aspect='equal')
        axs = [ax1,ax2]

        #tst.plot_intersects( axs=axs, rr=rr, normal=False, origin=False, rayline=False)
        tst.plot_intersects( axs=axs, rr=rr)

        if type(root) is Node:
            for ax in axs:
                rdr = Renderer(ax)
                rdr.render(tst.root)
                #rdr.limits(400,400)
                ax.axis('auto')
            pass
        else:
            pass  # TODO: support rendering basis shapes from partBuffer

        fig.suptitle(tst.suptitle, horizontalalignment='left', family='monospace', fontsize=10, x=0.1, y=0.99) 
        fig.show()
    pass


    i = tst.i
    r0 = tst.rays[0]
    r1 = tst.rays[1]


 
