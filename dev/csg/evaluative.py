


from slavish import * 


def evaluative_intersect(partBuffer, ray, tst):
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

    numInternalNodes = (0x1 << (1+height)) - 1       
    numNodes = (0x1 << (1+fullHeight)) - 1       


    postorder = postorder_sequence[fullHeight] 

    _tmin = np.zeros( (TRANCHE_STACK_SIZE), dtype=np.float32 )
    _tranche = np.zeros( (TRANCHE_STACK_SIZE), dtype=np.uint32 )
    tranche = -1

    LHS, RHS = 0, 1
    csg = CSG_()
    csg.curr = -1

    MISS, LEFT, RIGHT, RFLIP = 0,1,2,3
    isect = np.zeros( [4, 4, 4], dtype=np.float32 )

    tranche = TRANCHE_PUSH0( _tranche, _tmin, tranche, POSTORDER_SLICE(0, numInternalNodes), ray.tmin )

    while tranche >= 0:
        tranche, tmp, tmin = TRANCHE_POP0( _tranche, _tmin,  tranche )
        begin = POSTORDER_BEGIN(tmp)
        end = POSTORDER_END(tmp)
        if debug:log.info("%6d E : tranche begin %d end %d " % (tst.iray, begin, end))

        i = begin
        while i < end:

            nodeIdx = POSTORDER_NODE(postorder, i)
            depth = 32 - clz_(nodeIdx)-1
            subNodes = (0x1 << (1+height-depth)) - 1
            halfNodes = (subNodes - 1)/2 
            primitive = nodeIdx > numInternalNodes 

            operation = partBuffer[nodeIdx-1,Q1,W].view(np.uint32)

            tX_min = np.zeros( 2, dtype=np.float32 )
            tX_min[LHS] = tmin
            tX_min[RHS] = tmin

            if primitive:
                ise = intersect_primitive( Node.fromPart(partBuffer[partOffset+nodeIdx-1]), ray, tmin)
                csg.push(ise)
            else:
                isect[RIGHT] = csg.pop()  # NB order reversed
                isect[LEFT]  = csg.pop()

                x_state[LHS] = CSG_CLASSIFY( isect[LEFT][0], ray.direction, tmin )
                x_state[RHS] = CSG_CLASSIFY( isect[RIGHT][0], ray.direction, tmin )
                ctrl = boolean_ctrl_packed_lookup( operation, x_state[LHS], x_state[RHS], isect[LEFT][0][W] <= isect[RIGHT][0][W] )

                if ctrl < CTRL_LOOP_A:
                    isect[RFLIP] = isect[RIGHT]
                    isect[RFLIP,0,X] = -isect[RFLIP,0,X]
                    isect[RFLIP,0,Y] = -isect[RFLIP,0,Y]
                    isect[RFLIP,0,Z] = -isect[RFLIP,0,Z]
                    csg.push(isect[ctrl])
                else:
                    side = ctrl - CTRL_LOOP_A   

                    tX_min[side] = isect[THIS][0][W] + propagate_epsilon 
                    subtree = POSTORDER_SLICE(i-2*halfNodes, i-halfNodes) if side == LHS else POSTORDER_SLICE(i-halfNodes, i) 
                    tranche = TRANCHE_PUSH( _tranche, _tmin, tranche, POSTORDER_SLICE(i, numInternalNodes), tmin )
                    tranche = TRANCHE_PUSH( _tranche, _tmin, tranche, subtree, tX_min[side] )
                pass  
            pass

            i += 1   # next postorder node in the tranche
        pass         # end traversal loop
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



