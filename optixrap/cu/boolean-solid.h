#pragma once
/**
boolean-solid.h
=================

Boolean solid (union, intersection and difference) 
state tables obtained from the Andrew Kensler paper entitled 
"Ray Tracing CSG Objects Using Single Hit Intersections"

* http://xrt.wdfiles.com/local--files/doc%3Acsg/CSG.pdf

With corrections and additions from the XRT Renderer webpage 

* http://xrt.wikidot.com/doc:csg

Notes can be found in env/csg-/csg-vi

Changes compare to XRT source

* replaced "ReturnBIfCloser | FlipB" with merged ReturnFlipBIfCloser
  for simpler recording of algorithm actions

**/


enum 
{
    ReturnMiss              = 0x1 << 0,
    ReturnAIfCloser         = 0x1 << 1,
    ReturnAIfFarther        = 0x1 << 2,
    ReturnA                 = 0x1 << 3,
    ReturnBIfCloser         = 0x1 << 4,
    ReturnBIfFarther        = 0x1 << 5,
    ReturnB                 = 0x1 << 6,
    ReturnFlipBIfCloser     = 0x1 << 7,
    AdvanceAAndLoop         = 0x1 << 8,
    AdvanceBAndLoop         = 0x1 << 9,
    AdvanceAAndLoopIfCloser = 0x1 << 10,    
    AdvanceBAndLoopIfCloser = 0x1 << 11
};


enum { 
     CTRL_LOOP_A        = 0x1 << 0,  
     CTRL_LOOP_B        = 0x1 << 1,
     CTRL_RETURN_MISS   = 0x1 << 2,
     CTRL_RETURN_A      = 0x1 << 3,
     CTRL_RETURN_B      = 0x1 << 4,
     CTRL_RETURN_FLIP_B = 0x1 << 5,
     CTRL_ERROR         = 0x1 << 6
};  

enum {
     ERROR_LHS_POP_EMPTY         = 0x1 << 0, 
     ERROR_RHS_POP_EMPTY         = 0x1 << 1, 
     ERROR_LHS_END_NONEMPTY      = 0x1 << 2, 
     ERROR_RHS_END_EMPTY         = 0x1 << 3,
     ERROR_BAD_CTRL              = 0x1 << 4,
     ERROR_LHS_OVERFLOW          = 0x1 << 5,
     ERROR_RHS_OVERFLOW          = 0x1 << 6,
     ERROR_LHS_TRANCHE_OVERFLOW  = 0x1 << 7,
     ERROR_RHS_TRANCHE_OVERFLOW  = 0x1 << 8
};




enum 
{
    Union_EnterA_EnterB = ReturnAIfCloser | ReturnBIfCloser,
    Union_EnterA_ExitB  = ReturnBIfCloser | AdvanceAAndLoop,
    Union_EnterA_MissB  = ReturnA, 
    Union_ExitA_EnterB  = ReturnAIfCloser | AdvanceBAndLoop,
    Union_ExitA_ExitB   = ReturnAIfFarther | ReturnBIfFarther,
    Union_ExitA_MissB   = ReturnA ,
    Union_MissA_EnterB  = ReturnB ,
    Union_MissA_ExitB   = ReturnB ,
    Union_MissA_MissB   = ReturnMiss 
};

enum 
{
    Difference_EnterA_EnterB =  ReturnAIfCloser | AdvanceBAndLoop,
    Difference_EnterA_ExitB  =  AdvanceAAndLoopIfCloser | AdvanceBAndLoopIfCloser,
    Difference_EnterA_MissB  =  ReturnA,
    Difference_ExitA_EnterB  =  ReturnAIfCloser | ReturnFlipBIfCloser,
    Difference_ExitA_ExitB   =  ReturnFlipBIfCloser | AdvanceAAndLoop,
    Difference_ExitA_MissB   =  ReturnA,
    Difference_MissA_EnterB  =  ReturnMiss,
    Difference_MissA_ExitB   =  ReturnMiss,
    Difference_MissA_MissB   =  ReturnMiss
};

enum 
{
    Intersection_EnterA_EnterB = AdvanceAAndLoopIfCloser | AdvanceBAndLoopIfCloser,
    Intersection_EnterA_ExitB  = ReturnAIfCloser | AdvanceBAndLoop,
    Intersection_EnterA_MissB  = ReturnMiss,
    Intersection_ExitA_EnterB  = ReturnBIfCloser | AdvanceAAndLoop,
    Intersection_ExitA_ExitB   = ReturnAIfCloser | ReturnBIfCloser,
    Intersection_ExitA_MissB   = ReturnMiss,
    Intersection_MissA_EnterB  = ReturnMiss, 
    Intersection_MissA_ExitB   = ReturnMiss,
    Intersection_MissA_MissB   = ReturnMiss 
};

typedef enum { Enter, Exit, Miss } IntersectionState_t ;




/*
Perhaps could treat the LUT like a matrix and get into optix that way 

rtDeclareVariable(optix::Matrix4x4, textureMatrix0, , );

* https://devtalk.nvidia.com/default/topic/739954/optix/emulating-opengl-texture-matrix/
* https://devtalk.nvidia.com/default/topic/767650/optix/matrix4x4-assignment-not-working/

Or as tiny 3D buffer/texture 
*/

#ifdef __CUDACC__
__host__
__device__
#endif
int boolean_lookup( OpticksCSG_t op, IntersectionState_t stateA, IntersectionState_t stateB )
{
    //
    // assumes OpticksCSG_t enum values 0,1,2 for CSG_UNION, CSG_INTERSECTION, CSG_DIFFERENCE
    // and IntersectionState_t enum values 0,1,2 for Enter, Exit, Miss
    //
    // cannot use static here with CUDA, does that mean the lookup gets created every time ?
    //
    const unsigned _boolean_lookup[3][3][3] = 
          { 
             { 
                 {Union_EnterA_EnterB, Union_EnterA_ExitB, Union_EnterA_MissB },
                 {Union_ExitA_EnterB,  Union_ExitA_ExitB,  Union_ExitA_MissB },
                 {Union_MissA_EnterB,  Union_MissA_ExitB,  Union_MissA_MissB }
             },
             { 
                 {Intersection_EnterA_EnterB, Intersection_EnterA_ExitB, Intersection_EnterA_MissB },
                 {Intersection_ExitA_EnterB,  Intersection_ExitA_ExitB,  Intersection_ExitA_MissB },
                 {Intersection_MissA_EnterB,  Intersection_MissA_ExitB,  Intersection_MissA_MissB }
             },
             { 
                 {Difference_EnterA_EnterB, Difference_EnterA_ExitB, Difference_EnterA_MissB },
                 {Difference_ExitA_EnterB,  Difference_ExitA_ExitB,  Difference_ExitA_MissB },
                 {Difference_MissA_EnterB,  Difference_MissA_ExitB,  Difference_MissA_MissB }
             }
         } ;
    return _boolean_lookup[(int)op][(int)stateA][(int)stateB] ; 
}




#ifdef __CUDACC__
__host__
__device__
#endif
int union_action( IntersectionState_t stateA, IntersectionState_t stateB  )
{
    int offset = 3*(int)stateA + (int)stateB ;   
    int action = Union_MissA_MissB ; 
    switch(offset)
    {
       case 0: action=Union_EnterA_EnterB ; break ; 
       case 1: action=Union_EnterA_ExitB  ; break ; 
       case 2: action=Union_EnterA_MissB  ; break ; 
       case 3: action=Union_ExitA_EnterB  ; break ; 
       case 4: action=Union_ExitA_ExitB   ; break ; 
       case 5: action=Union_ExitA_MissB   ; break ; 
       case 6: action=Union_MissA_EnterB  ; break ; 
       case 7: action=Union_MissA_ExitB   ; break ; 
       case 8: action=Union_MissA_MissB   ; break ; 
    }
    return action ; 
}

#ifdef __CUDACC__
__host__
__device__
#endif
int intersection_action( IntersectionState_t stateA, IntersectionState_t stateB  )
{
    int offset = 3*(int)stateA + (int)stateB ;   
    int action = Intersection_MissA_MissB ; 
    switch(offset)
    {
       case 0: action=Intersection_EnterA_EnterB ; break ; 
       case 1: action=Intersection_EnterA_ExitB  ; break ; 
       case 2: action=Intersection_EnterA_MissB  ; break ; 
       case 3: action=Intersection_ExitA_EnterB  ; break ; 
       case 4: action=Intersection_ExitA_ExitB   ; break ; 
       case 5: action=Intersection_ExitA_MissB   ; break ; 
       case 6: action=Intersection_MissA_EnterB  ; break ; 
       case 7: action=Intersection_MissA_ExitB   ; break ; 
       case 8: action=Intersection_MissA_MissB   ; break ; 
    }
    return action ; 
}

#ifdef __CUDACC__
__host__
__device__
#endif
int difference_action( IntersectionState_t stateA, IntersectionState_t stateB  )
{
    int offset = 3*(int)stateA + (int)stateB ;   
    int action = Difference_MissA_MissB ; 
    switch(offset)
    {
       case 0: action=Difference_EnterA_EnterB ; break ; 
       case 1: action=Difference_EnterA_ExitB  ; break ; 
       case 2: action=Difference_EnterA_MissB  ; break ; 
       case 3: action=Difference_ExitA_EnterB  ; break ; 
       case 4: action=Difference_ExitA_ExitB   ; break ; 
       case 5: action=Difference_ExitA_MissB   ; break ; 
       case 6: action=Difference_MissA_EnterB  ; break ; 
       case 7: action=Difference_MissA_ExitB   ; break ; 
       case 8: action=Difference_MissA_MissB   ; break ; 
    }
    return action ; 
}



#ifdef __CUDACC__
__host__
__device__
#endif
int boolean_actions( OpticksCSG_t operation, IntersectionState_t stateA, IntersectionState_t stateB  )
{
    int action = ReturnMiss ; 
    switch(operation)
    {
       case CSG_INTERSECTION: action = intersection_action( stateA, stateB ) ; break ;
       case CSG_UNION:        action = union_action( stateA, stateB ) ; break ;
       case CSG_DIFFERENCE:   action = difference_action( stateA, stateB ) ; break ;
       case CSG_PRIMITIVE:    action = ReturnMiss                          ; break ;   // perhaps return error flag ?
    }
    return action ; 
}

#ifdef __CUDACC__
__host__
__device__
#endif
int boolean_decision(int acts, bool ACloser)
{
    int act = ReturnMiss ; 

    // convert potentially multiple bits of acts into single bit act 
    // NB the order of this if is critical
    //
    // hmm : not so many possible values of acts and two possible values of ACloser
    //       can this been done via lookup too ?
    //
    // hmm the ordering multiplies the possible "states" for the lookup ?
    //      but not by that much, especially when split into two for ACloser or not
    //

    if      (acts & ReturnMiss)                            act = ReturnMiss ; 
    else if (acts & ReturnA)                               act = ReturnA ; 
    else if ((acts & ReturnAIfCloser) && ACloser)          act = ReturnAIfCloser ; 
    else if ((acts & ReturnAIfFarther) && !ACloser)        act = ReturnAIfFarther ; 
    else if (acts & ReturnB)                               act = ReturnB ;
    else if ((acts & ReturnBIfCloser) && !ACloser)         act = ReturnBIfCloser ;
    else if ((acts & ReturnFlipBIfCloser) && !ACloser)     act = ReturnFlipBIfCloser ;
    else if ((acts & ReturnBIfFarther) && ACloser)         act = ReturnBIfFarther ;
    else if (acts & AdvanceAAndLoop)                       act = AdvanceAAndLoop ;
    else if ((acts & AdvanceAAndLoopIfCloser) && ACloser)  act = AdvanceAAndLoopIfCloser ;
    else if (acts & AdvanceBAndLoop)                       act = AdvanceBAndLoop ;
    else if ((acts & AdvanceBAndLoopIfCloser) && !ACloser) act = AdvanceBAndLoopIfCloser ;

    return act  ;
}


#ifdef __CUDACC__
__host__
__device__
#endif
int boolean_ctrl(int act)
{
    // TODO: combine with boolean_decision when debugged
    int ctrl = CTRL_ERROR ; 
    if(act & ReturnMiss)                                           ctrl = CTRL_RETURN_MISS ; 
    else if( act & (ReturnA | ReturnAIfCloser | ReturnAIfFarther)) ctrl = CTRL_RETURN_A ; 
    else if( act & (ReturnB | ReturnBIfCloser | ReturnBIfFarther)) ctrl = CTRL_RETURN_B ; 
    else if( act & ReturnFlipBIfCloser)                            ctrl = CTRL_RETURN_FLIP_B ; 
    else if( act & (AdvanceAAndLoop | AdvanceAAndLoopIfCloser))    ctrl = CTRL_LOOP_A  ; 
    else if( act & (AdvanceBAndLoop | AdvanceBAndLoopIfCloser))    ctrl = CTRL_LOOP_B  ; 
    return ctrl ; 
}










