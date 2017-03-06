#!/usr/bin/env python

import numpy as np

from opticks.bin.ffs import ffs_, clz_
from intersect import Ray, intersect_primitive
from node import Node, root4
from node import Q0, Q1, Q2, Q3, X, Y, Z, W
from slavish import CSG_, TRANCHE_STACK_SIZE, TRANCHE_PUSH0, TRANCHE_POP0
from slavish import POSTORDER_SLICE, POSTORDER_BEGIN, POSTORDER_END, POSTORDER_NODE


from opticks.optixrap.cu.boolean_solid_h import CTRL_RETURN_MISS
from opticks.optixrap.cu.boolean_solid_h import ERROR_LHS_POP_EMPTY, ERROR_RHS_POP_EMPTY, ERROR_LHS_END_NONEMPTY, ERROR_RHS_END_EMPTY, ERROR_BAD_CTRL, ERROR_LHS_OVERFLOW, ERROR_RHS_OVERFLOW, ERROR_LHS_TRANCHE_OVERFLOW
from opticks.optixrap.cu.boolean_h import desc_state, Enter, Exit, Miss


def CSG_CLASSIFY(ise, dir_, tmin):
     if ise[W] > tmin:
         return Enter if np.dot( ise[:3], dir_ ) < 0 else Exit 
     else:
         return Miss


TRANCHE_STACK_SIZE = 4
CSG_STACK_SIZE = 4

class CSG_(object):
    def __init__(self):
        self.data = np.zeros( (CSG_STACK_SIZE), dtype=np.float32 )
        self.curr = -1

    def pop(self):
        if self.curr < 0:
            raise IndexError
        else:
            ret = self.data[self.curr] 
            self.curr -= 1 
        pass
        return ret


def TRANCHE_PUSH0( _stacku, _stackf, stack, valu, valf ):
    stack += 1 
    _stacku[stack] = valu 
    _stackf[stack] = valf
    return stack 

def TRANCHE_POP0( _stacku, _stackf, stack):
    valu = _stacku[stack] 
    valf = _stackf[stack] 
    stack -= 1 
    return stack, valu, valf







POSTORDER_SLICE = lambda begin, end:((((end) & 0xffff) << 16) | ((begin) & 0xffff)  )

POSTORDER_BEGIN = lambda tmp:( (tmp) & 0xffff )
POSTORDER_END = lambda tmp:( (tmp) >> 16 )
POSTORDER_NODE = lambda postorder, i: (((postorder) & (0xF << (i)*4 )) >> (i)*4 )



# generated from /Users/blyth/opticks/optixrap/cu by boolean_h.py on Sat Mar  4 20:37:03 2017 
packed_boolean_lut_ACloser = [ 0x22121141, 0x00014014, 0x00141141, 0x00000000 ]
packed_boolean_lut_BCloser = [ 0x22115122, 0x00022055, 0x00133155, 0x00000000 ]
             
def boolean_ctrl_packed_lookup(operation, stateA, stateB, ACloser ):
    lut = packed_boolean_lut_ACloser if ACloser else  packed_boolean_lut_BCloser 
    offset = 3*stateA + stateB ;
    return (lut[operation] >> (offset*4)) & 0xf if offset < 8 else CTRL_RETURN_MISS 




class CSG(object):pass



if __name__ == '__main__':

    root = root4
    root.annotate() 

    self = CSG()
    self.ray = Ray()

    ### params

    tmin = 0 
    partBuffer = Node.serialize( root )

    ######## start intersect 

    postorder_sequence = [ 0x1, 0x132, 0x1376254, 0x137fe6dc25ba498 ] 
    ierr = 0
    abort_ = False

    partOffset = 0 
    numParts = len(partBuffer)
    primIdx = -1

    fullHeight = ffs_(numParts + 1) - 2 
    height = fullHeight - 1 

    postorder = postorder_sequence[height] 
    numInternalNodes = (0x1 << (1+height)) - 1       

    print "postorder %16x numInternalNodes %d " % (postorder, numInternalNodes)


    _tmin = np.zeros( (TRANCHE_STACK_SIZE), dtype=np.float32 )
    _tranche = np.zeros( (TRANCHE_STACK_SIZE), dtype=np.uint32 )
    tranche = -1

    LHS, RHS = 0, 1
    csg_ = [CSG_(), CSG_()] 
    lhs = csg_[LHS] 
    rhs = csg_[RHS] 
    lhs.curr = -1
    rhs.curr = -1

    MISS, LEFT, RIGHT, RFLIP = 0,1,2,3
    isect = np.zeros( [4, 4], dtype=np.float32 )


    tranche = TRANCHE_PUSH0( _tranche, _tmin, tranche, POSTORDER_SLICE(0, numInternalNodes), tmin )

    print "tranche %16x " % tranche

    while tranche >= 0:
        tranche, tmp, tmin = TRANCHE_POP0( _tranche, _tmin,  tranche )
        begin = POSTORDER_BEGIN(tmp)
        end = POSTORDER_END(tmp)

        print " begin %d end %d " % (begin, end)
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

            print i, nodeIdx, depth, subNodes, halfNodes , operation

            if bileaf:
                isect[LEFT][W] = 0.
                isect[RIGHT][W] = 0.
                isect[LEFT] = intersect_primitive( Node.fromPart(partBuffer[partOffset+leftIdx-1]), self.ray, tX_min[LHS])[0]
                isect[RIGHT] = intersect_primitive( Node.fromPart(partBuffer[partOffset+rightIdx-1]), self.ray, tX_min[RHS])[0]
            else:
                try:
                    isect[LEFT] = lhs.pop()
                except IndexError:
                    ierr |= ERROR_LHS_POP_EMPTY
                pass
                try:
                    isect[RIGHT] = rhs.pop()
                except IndexError:
                    ierr |= ERROR_RHS_POP_EMPTY
                pass
            pass

            x_state[LHS] = CSG_CLASSIFY( isect[LEFT], self.ray.direction, tX_min[LHS] )
            x_state[RHS] = CSG_CLASSIFY( isect[RIGHT], self.ray.direction, tX_min[RHS] )

            ctrl = boolean_ctrl_packed_lookup( operation, x_state[LHS], x_state[RHS], isect[LEFT][W] <= isect[RIGHT][W] )

            print "ctrl", ctrl


            i += 1 
        pass     








 
