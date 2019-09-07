/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#pragma once
/**
boolean_solid.h
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

NB this enum header is converted into a python equivalent with::

    cd ~/opticks/optixrap/cu
    c_enums_to_python.py boolean_solid.h > boolean_solid.py 

Use from python with for example::

    from opticks.optixrap.cu.boolean_solid import Act_, CTRL_, State_, Intersection_ 
    # classnames correspond to common prefixes on enum keys  

**/


enum 
{
    Act_ReturnMiss              = 0x1 << 0,
    Act_ReturnAIfCloser         = 0x1 << 1,
    Act_ReturnAIfFarther        = 0x1 << 2,
    Act_ReturnA                 = 0x1 << 3,
    Act_ReturnBIfCloser         = 0x1 << 4,
    Act_ReturnBIfFarther        = 0x1 << 5,
    Act_ReturnB                 = 0x1 << 6,
    Act_ReturnFlipBIfCloser     = 0x1 << 7,
    Act_AdvanceAAndLoop         = 0x1 << 8,
    Act_AdvanceBAndLoop         = 0x1 << 9,
    Act_AdvanceAAndLoopIfCloser = 0x1 << 10,    
    Act_AdvanceBAndLoopIfCloser = 0x1 << 11
};


// provide values for python conversion
enum { 
     CTRL_RETURN_MISS    = 0,
     CTRL_RETURN_A       = 1,
     CTRL_RETURN_B       = 2,
     CTRL_RETURN_FLIP_B  = 3,
     CTRL_LOOP_A         = 4,   
     CTRL_LOOP_B         = 5
};  

typedef enum { 
    State_Enter = 0, 
    State_Exit  = 1, 
    State_Miss  = 2 
} IntersectionState_t ;



enum {
     ERROR_LHS_POP_EMPTY         = 0x1 << 0, 
     ERROR_RHS_POP_EMPTY         = 0x1 << 1, 
     ERROR_LHS_END_NONEMPTY      = 0x1 << 2, 
     ERROR_RHS_END_EMPTY         = 0x1 << 3,
     ERROR_BAD_CTRL              = 0x1 << 4,
     ERROR_LHS_OVERFLOW          = 0x1 << 5,
     ERROR_RHS_OVERFLOW          = 0x1 << 6,
     ERROR_LHS_TRANCHE_OVERFLOW  = 0x1 << 7,
     ERROR_RHS_TRANCHE_OVERFLOW  = 0x1 << 8,
     ERROR_RESULT_OVERFLOW       = 0x1 << 9,
     ERROR_OVERFLOW              = 0x1 << 10,
     ERROR_TRANCHE_OVERFLOW      = 0x1 << 11,
     ERROR_POP_EMPTY             = 0x1 << 12,
     ERROR_XOR_SIDE              = 0x1 << 13,
     ERROR_END_EMPTY             = 0x1 << 14,
     ERROR_ROOT_STATE            = 0x1 << 15
};


enum 
{
    Union_EnterA_EnterB = Act_ReturnAIfCloser | Act_ReturnBIfCloser,
    Union_EnterA_ExitB  = Act_ReturnBIfCloser | Act_AdvanceAAndLoop,
    Union_EnterA_MissB  = Act_ReturnA, 
    Union_ExitA_EnterB  = Act_ReturnAIfCloser | Act_AdvanceBAndLoop,
    Union_ExitA_ExitB   = Act_ReturnAIfFarther | Act_ReturnBIfFarther,
    Union_ExitA_MissB   = Act_ReturnA ,
    Union_MissA_EnterB  = Act_ReturnB ,
    Union_MissA_ExitB   = Act_ReturnB ,
    Union_MissA_MissB   = Act_ReturnMiss 
};

// below ACloser_ and BCloser_ manually obtained from above source table
enum
{
    ACloser_Union_EnterA_EnterB = CTRL_RETURN_A,
    ACloser_Union_EnterA_ExitB  = CTRL_LOOP_A,
    ACloser_Union_EnterA_MissB  = CTRL_RETURN_A,
    ACloser_Union_ExitA_EnterB  = CTRL_RETURN_A,
    ACloser_Union_ExitA_ExitB   = CTRL_RETURN_B,
    ACloser_Union_ExitA_MissB   = CTRL_RETURN_A,
    ACloser_Union_MissA_EnterB  = CTRL_RETURN_B,
    ACloser_Union_MissA_ExitB   = CTRL_RETURN_B,
    ACloser_Union_MissA_MissB   = CTRL_RETURN_MISS
};

enum
{
    BCloser_Union_EnterA_EnterB = CTRL_RETURN_B,
    BCloser_Union_EnterA_ExitB  = CTRL_RETURN_B,
    BCloser_Union_EnterA_MissB  = CTRL_RETURN_A,
    BCloser_Union_ExitA_EnterB  = CTRL_LOOP_B,
    BCloser_Union_ExitA_ExitB   = CTRL_RETURN_A,
    BCloser_Union_ExitA_MissB   = CTRL_RETURN_A,
    BCloser_Union_MissA_EnterB  = CTRL_RETURN_B,
    BCloser_Union_MissA_ExitB   = CTRL_RETURN_B,
    BCloser_Union_MissA_MissB   = CTRL_RETURN_MISS
};



enum 
{
    Difference_EnterA_EnterB =  Act_ReturnAIfCloser | Act_AdvanceBAndLoop,
    Difference_EnterA_ExitB  =  Act_AdvanceAAndLoopIfCloser | Act_AdvanceBAndLoopIfCloser,
    Difference_EnterA_MissB  =  Act_ReturnA,
    Difference_ExitA_EnterB  =  Act_ReturnAIfCloser | Act_ReturnFlipBIfCloser,
    Difference_ExitA_ExitB   =  Act_ReturnFlipBIfCloser | Act_AdvanceAAndLoop,
    Difference_ExitA_MissB   =  Act_ReturnA,
    Difference_MissA_EnterB  =  Act_ReturnMiss,
    Difference_MissA_ExitB   =  Act_ReturnMiss,
    Difference_MissA_MissB   =  Act_ReturnMiss
};
// below ACloser_ and BCloser_ manually obtained from above source table
enum
{
    ACloser_Difference_EnterA_EnterB = CTRL_RETURN_A, 
    ACloser_Difference_EnterA_ExitB  = CTRL_LOOP_A,
    ACloser_Difference_EnterA_MissB  = CTRL_RETURN_A,
    ACloser_Difference_ExitA_EnterB  = CTRL_RETURN_A,
    ACloser_Difference_ExitA_ExitB   = CTRL_LOOP_A,
    ACloser_Difference_ExitA_MissB   = CTRL_RETURN_A,
    ACloser_Difference_MissA_EnterB  = CTRL_RETURN_MISS,
    ACloser_Difference_MissA_ExitB   = CTRL_RETURN_MISS, 
    ACloser_Difference_MissA_MissB   = CTRL_RETURN_MISS 
};
enum
{
    BCloser_Difference_EnterA_EnterB = CTRL_LOOP_B, 
    BCloser_Difference_EnterA_ExitB  = CTRL_LOOP_B,
    BCloser_Difference_EnterA_MissB  = CTRL_RETURN_A,
    BCloser_Difference_ExitA_EnterB  = CTRL_RETURN_FLIP_B,
    BCloser_Difference_ExitA_ExitB   = CTRL_RETURN_FLIP_B,
    BCloser_Difference_ExitA_MissB   = CTRL_RETURN_A,
    BCloser_Difference_MissA_EnterB  = CTRL_RETURN_MISS,
    BCloser_Difference_MissA_ExitB   = CTRL_RETURN_MISS, 
    BCloser_Difference_MissA_MissB   = CTRL_RETURN_MISS 
};






enum 
{
    Intersection_EnterA_EnterB = Act_AdvanceAAndLoopIfCloser | Act_AdvanceBAndLoopIfCloser,
    Intersection_EnterA_ExitB  = Act_ReturnAIfCloser | Act_AdvanceBAndLoop,
    Intersection_EnterA_MissB  = Act_ReturnMiss,
    Intersection_ExitA_EnterB  = Act_ReturnBIfCloser | Act_AdvanceAAndLoop,
    Intersection_ExitA_ExitB   = Act_ReturnAIfCloser | Act_ReturnBIfCloser,
    Intersection_ExitA_MissB   = Act_ReturnMiss,
    Intersection_MissA_EnterB  = Act_ReturnMiss, 
    Intersection_MissA_ExitB   = Act_ReturnMiss,
    Intersection_MissA_MissB   = Act_ReturnMiss 
};
// below ACloser_ and BCloser_ manually obtained from above source table
enum
{
    ACloser_Intersection_EnterA_EnterB = CTRL_LOOP_A,
    ACloser_Intersection_EnterA_ExitB  = CTRL_RETURN_A,
    ACloser_Intersection_EnterA_MissB  = CTRL_RETURN_MISS,
    ACloser_Intersection_ExitA_EnterB  = CTRL_LOOP_A,
    ACloser_Intersection_ExitA_ExitB   = CTRL_RETURN_A,
    ACloser_Intersection_ExitA_MissB   = CTRL_RETURN_MISS,
    ACloser_Intersection_MissA_EnterB  = CTRL_RETURN_MISS,
    ACloser_Intersection_MissA_ExitB   = CTRL_RETURN_MISS,
    ACloser_Intersection_MissA_MissB   = CTRL_RETURN_MISS
};
enum
{
    BCloser_Intersection_EnterA_EnterB = CTRL_LOOP_B,
    BCloser_Intersection_EnterA_ExitB  = CTRL_LOOP_B,
    BCloser_Intersection_EnterA_MissB  = CTRL_RETURN_MISS,
    BCloser_Intersection_ExitA_EnterB  = CTRL_RETURN_B,
    BCloser_Intersection_ExitA_ExitB   = CTRL_RETURN_B,
    BCloser_Intersection_ExitA_MissB   = CTRL_RETURN_MISS,
    BCloser_Intersection_MissA_EnterB  = CTRL_RETURN_MISS,
    BCloser_Intersection_MissA_ExitB   = CTRL_RETURN_MISS,
    BCloser_Intersection_MissA_MissB   = CTRL_RETURN_MISS
};



