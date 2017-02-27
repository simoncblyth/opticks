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


