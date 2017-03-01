#pragma once

// see opticks/dev/csg/node.py:Node.postOrderSequence
rtDeclareVariable(unsigned long long, postorder_seq2, , ) =  0x132 ;
rtDeclareVariable(unsigned long long, postorder_seq3, , ) =  0x1376254 ;
rtDeclareVariable(unsigned long long, postorder_seq4, , ) =  0x137fe6dc25ba498 ;



static __device__
void intersect_boolean_only_first( const uint4& prim, const uint4& identity )
{
    unsigned a_partIdx = prim.x + 1 ;  

    float3 a_normal = make_float3(0.f,0.f,1.f);

    float tA_min = propagate_epsilon ;  
    float tA     = tA_min ;  

    IntersectionState_t a_state = intersect_part( a_partIdx , tA_min, a_normal, tA ) ;


    if(a_state != Miss)
    {
        if(rtPotentialIntersection(tA))
        {
            shading_normal = geometric_normal = a_normal;
            instanceIdentity = identity ;

#ifdef BOOLEAN_DEBUG
            instanceIdentity.x = dot(a_normal, ray.direction) < 0.f ? 1 : 2 ;
#endif
            rtReportIntersection(0);
        }
    }
}




/*
static __device__
void intersect_csg( const uint4& prim, const uint4& identity )
{
   // hmm need to thread the tree before can start with this, and provide leftmost operation jumpoff point
}
*/



static __device__
void intersect_boolean( const uint4& prim, const uint4& identity )
{
    // NB LIMITED TO SINGLE BOOLEAN OPERATION APPLIED TO TWO BASIS SOLIDS, ie triplet trees

    // primFlags only available for root of tree,
    // operate from partBuffer for other nodes

    unsigned partOffset = prim.x ; 
    unsigned primIdx_   = prim.z ; 

    unsigned n_partIdx = partOffset ;    
    unsigned a_partIdx = partOffset + 1 ;   // SIMPLIFYING TRIPLET ASSUMPTION
    unsigned b_partIdx = partOffset + 2 ;  

    quad q1 ; 
    q1.f = partBuffer[4*n_partIdx+1];
    OpticksShape_t operation = (OpticksShape_t)q1.u.w ;

    rtPrintf("intersect_boolean primIdx_:%u n:%u a:%u b:%u operation:%u \n", primIdx_, n_partIdx, a_partIdx, b_partIdx, operation );

    float3 a_normal = make_float3(0.f,0.f,1.f);
    float3 b_normal = make_float3(0.f,0.f,1.f);

    float tA_min = ray.tmin ; // formerly propagate_epsilon and before that 0.f
    float tB_min = ray.tmin ;
    float tA     = 0.f ;
    float tB     = 0.f ;

    int ctrl = CTRL_LOOP_A | CTRL_LOOP_B ; 

    IntersectionState_t a_state = Miss ; 
    IntersectionState_t b_state = Miss ; 

    int count(0) ;  
    while((ctrl & (CTRL_LOOP_A | CTRL_LOOP_B)) && count < 4 )
    {
        count++ ; 

        a_state = (ctrl & CTRL_LOOP_A) ? intersect_part( a_partIdx , tA_min, a_normal, tA ) : a_state ;
        b_state = (ctrl & CTRL_LOOP_B) ? intersect_part( b_partIdx , tB_min, b_normal, tB ) : b_state ;

        int actions = boolean_actions( operation , a_state, b_state );
        int act = boolean_decision( actions, tA <= tB );
        ctrl = boolean_ctrl( act );

        if(     ctrl == CTRL_LOOP_A) tA_min = tA ; 
        else if(ctrl == CTRL_LOOP_B) tB_min = tB ; 
    } 


    // hmm below passing to OptiX should probably be done in caller ?
    if( ctrl & (CTRL_RETURN_A | CTRL_RETURN_B | CTRL_RETURN_FLIP_B  ))
    {
        if(rtPotentialIntersection( ctrl == CTRL_RETURN_A ? tA : tB))
        {
            shading_normal = geometric_normal = ctrl == CTRL_RETURN_A ? 
                                                                           a_normal
                                                                      :
                                                                          ( ctrl == CTRL_RETURN_FLIP_B ? -b_normal : b_normal )
                                                                      ;
            instanceIdentity = identity ;
            rtReportIntersection(0);
        }
    } 

}



