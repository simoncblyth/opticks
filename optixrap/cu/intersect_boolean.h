#pragma once


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


static __device__
void intersect_boolean( const uint4& prim, const uint4& identity )
{
   // NB LIMITED TO SINGLE OPERATION APPLIED TO TWO BASIS SOLIDS
   //
   // hmm to work with boolean CSG tree primitives this
   // needs to have the same signature as intersect_part 
   // ie with deferring the reporting to OptiX to the caller

    unsigned primFlags  = prim.w ;  


    // TODO: pass "operation" enum from CPU side, instead of wishy-washy flags   
    enum { INTERSECT, UNION, DIFFERENCE  };
    int bop = primFlags & SHAPE_INTERSECTION ? 
                                                  INTERSECT 
                                             :
                                                  ( primFlags & SHAPE_DIFFERENCE ? DIFFERENCE : UNION ) 
                                             ;

    unsigned a_partIdx = prim.x + 1 ;  
    unsigned b_partIdx = prim.x + 2 ;  

    float3 a_normal = make_float3(0.f,0.f,1.f);
    float3 b_normal = make_float3(0.f,0.f,1.f);

    // _min 0.f rather than propagate_epsilon 
    // leads to missed boundaries when start photons on a boundary, 
    // see boolean_csg_on_gpu.rst

    //float tA_min = propagate_epsilon ;  
    //float tB_min = propagate_epsilon ;

    float tA_min = ray.tmin ;
    float tB_min = ray.tmin ;
    float tA     = 0.f ;
    float tB     = 0.f ;

    enum { 
             LIVE_A = 0x1 << 0,  
             LIVE_B = 0x1 << 1
           };  

    int ctrl = LIVE_A | LIVE_B ; 

    IntersectionState_t a_state = Miss ; 
    IntersectionState_t b_state = Miss ; 

    int count(0) ;  

    //rtPrintf("boolean_intersect t_parameter %f ", t_parameter);

#ifdef BOOLEAN_DEBUG
    //int debugA = 1 ; 
    //int debugB = 1 ; 
#endif

    while(ctrl != 0 && count < 4 )
    {
        count++ ; 

        a_state = (ctrl & LIVE_A) ? intersect_part( a_partIdx , tA_min, a_normal, tA ) : a_state ;
        b_state = (ctrl & LIVE_B) ? intersect_part( b_partIdx , tB_min, b_normal, tB ) : b_state ;

        int action = 0 ; 
        switch(bop)
        {
            case INTERSECT:    action = intersection_action(a_state, b_state) ; break ;
            case UNION:        action = union_action(a_state, b_state)        ; break ;
            case DIFFERENCE:   action = difference_action(a_state, b_state)   ; break ;
        }


        bool ACloser = tA <= tB ; 
        bool AFarther = !ACloser ; 
        bool BCloser = !ACloser ; 
        bool BFarther = ACloser ; 


        if(action & ReturnMiss)
        {
            ctrl = 0 ; 
        }
       else if( 
                   (action & ReturnA) 
                || 
                   ((action & ReturnAIfCloser) && ACloser )
                || 
                   ((action & ReturnAIfFarther) && AFarther )
                 )
        {
            ctrl = 0 ; 
            if(rtPotentialIntersection(tA))
            {
                shading_normal = geometric_normal = a_normal;
                instanceIdentity = identity ;
#ifdef BOOLEAN_DEBUG
                //if((action & ReturnA))                      instanceIdentity.x = 1 ; 
                //if((action & ReturnAIfCloser)  && tA <= tB) instanceIdentity.x = 2 ; 
                //if((action & ReturnAIfFarther) && tA > tB)  instanceIdentity.x = 3 ; 
                //
                // difference(box-sphere)
                //     * red   (ReturnA) box periphery where MissB the sphere
                //     * green (ReturnAIfCloser..) box hits in shape of sphere
                //     * no blue seen 
                //
                //instanceIdentity.x = debugA ; 
#endif
                rtReportIntersection(0);
            }
        }
        else if( 
                   (action & ReturnB) 
                || 
                   ((action & ReturnBIfCloser) && BCloser )
                || 
                   ((action & ReturnFlipBIfCloser) && BCloser )
                || 
                   ((action & ReturnBIfFarther) && BFarther )
                 )
        {
            ctrl = 0 ; 
            if(rtPotentialIntersection(tB))
            {
                shading_normal = geometric_normal = action & ReturnFlipBIfCloser ? -b_normal : b_normal ;
                instanceIdentity = identity ;
#ifdef BOOLEAN_DEBUG
                //if((action & ReturnB))                       instanceIdentity.x = 1 ; 
                //if((action & ReturnBIfCloser)  && BCloser)   instanceIdentity.x = 2 ; 
                //if((action & ReturnBIfFarther) && BFarther)  instanceIdentity.x = 3 ; 
                // difference(box-sphere) 
                //    no coloring apparent from outside (makes sense as sphere is "subtracted"),
                //     hint of green(ReturnBIfCloser) the sphere from inside
                //
#endif
                rtReportIntersection(0);
            }
        }
        else if(
                     (action & AdvanceAAndLoop)
                  ||  
                     ((action & AdvanceAAndLoopIfCloser) && tA <= tB )
                )
        {

#ifdef BOOLEAN_DEBUG
            //if( (action & AdvanceAAndLoop) )                     debugA = 2 ; 
            //if( (action & AdvanceAAndLoopIfCloser) && ACloser ) debugA = 3 ; 
#endif

            //ctrl = ctrl & ~LIVE_B  ;   // CAUSES INVISIBLE INSIDES 
            ctrl = LIVE_A  ; 
            tA_min = tA ; 
        }
        else if(
                     (action & AdvanceBAndLoop)
                  ||  
                     ((action & AdvanceBAndLoopIfCloser) && BCloser )
                )
        {
            //ctrl = ctrl & ~LIVE_A  ;   // CAUSES INVISIBLE INSIDES
            ctrl = LIVE_B ; 
            tB_min = tB ; 
        }

     }     // while loop 
}




