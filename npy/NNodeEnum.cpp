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


#include <cstddef>
#include <cmath>
#include <cassert>
#include <sstream>

#include "NNodeEnum.hpp"



const char* NNodeEnum::PAIR_MINMIN_ = "PAIR_MINMIN" ; 
const char* NNodeEnum::PAIR_MINMAX_ = "PAIR_MINMAX" ;
const char* NNodeEnum::PAIR_MAXMIN_ = "PAIR_MAXMIN" ;
const char* NNodeEnum::PAIR_MAXMAX_ = "PAIR_MAXMAX" ;

const char* NNodeEnum::PairType(NNodePairType pair)
{
    const char* s = NULL ; 
    switch(pair)
    {
       case PAIR_MINMIN : s = PAIR_MINMIN_ ; break ; 
       case PAIR_MINMAX : s = PAIR_MINMAX_ ; break ; 
       case PAIR_MAXMIN : s = PAIR_MAXMIN_ ; break ; 
       case PAIR_MAXMAX : s = PAIR_MAXMAX_ ; break ; 
    }
    return s ; 
}






const char* NNodeEnum::JOIN_UNCLASSIFIED_ = "JOIN_UNCLASSIFIED" ; 
const char* NNodeEnum::JOIN_COINCIDENT_ = "JOIN_COINCIDENT" ; 
const char* NNodeEnum::JOIN_OVERLAP_  = "JOIN_OVERLAP" ; 
const char* NNodeEnum::JOIN_SPLIT_     = "JOIN_SPLIT" ; 

const char* NNodeEnum::JoinType(NNodeJoinType join )
{
    const char* s = NULL ; 
    switch(join)
    {
       case JOIN_UNCLASSIFIED: s = JOIN_UNCLASSIFIED_ ; break ; 
       case JOIN_COINCIDENT: s = JOIN_COINCIDENT_ ; break ; 
       case JOIN_OVERLAP: s = JOIN_OVERLAP_ ; break ; 
       case JOIN_SPLIT: s = JOIN_SPLIT_ ; break ; 
    }
    return s ; 
} 

/**
NNodeEnum::JoinClassify
---------------------------

Illustrations are for MINMAX, see NNodeNudger.


JOIN_SPLIT
    zj-zi > 0.

    +-------+
    |       |
    |       |
    |       |
    +-------+  zj

    +-------+  zi
    |       |
    |       |
    |       |
    +-------+
   
    
JOIN_OVERLAP
    zj-zi < 0.

        +-------+
        |       |
        |   +-------+ zi
        |   |   |   | 
    zj  +---|---+   |
            |       |   
            +-------+
         

JOIN_COINCIDENT
    |zj-zi| < epsilon

        +-------+
        |       |
        |       |
        |       |
     zj +---+---+---+ zi
            |       |
            |       | 
            |       |
            +-------+       

**/

NNodeJoinType NNodeEnum::JoinClassify( float zi, float zj, float epsilon )
{
    float delta = zj - zi   ; 
    NNodeJoinType join = JOIN_UNCLASSIFIED ; 

    if( fabsf(delta) < epsilon )
    {
         join = JOIN_COINCIDENT ; 
    } 
    else if( delta < 0.f )
    {
         join = JOIN_OVERLAP ; 
    }
    else if( delta > 0.f )
    {
         join = JOIN_SPLIT ; 
    }
    else
    {
         assert(0);
    } 
    return join ; 
}



const char* NNodeEnum::NUDGE_NONE_ = "NUDGE_NONE" ; 
const char* NNodeEnum::NUDGE_J_DECREASE_Z1_ = "NUDGE_J_DECREASE_Z1" ; 
const char* NNodeEnum::NUDGE_I_INCREASE_Z2_ = "NUDGE_I_INCREASE_Z2" ; 
const char* NNodeEnum::NudgeType(NNodeNudgeType nudge)
{
    const char* s = NULL ; 
    switch(nudge)
    {
       case NUDGE_NONE:          s = NUDGE_NONE_          ; break ; 
       case NUDGE_J_DECREASE_Z1: s = NUDGE_J_DECREASE_Z1_ ; break ; 
       case NUDGE_I_INCREASE_Z2: s = NUDGE_I_INCREASE_Z2_ ; break ; 
    }
    return s ; 
} 



const char* NNodeEnum::FRAME_MODEL_ = "FRAME_MODEL" ;
const char* NNodeEnum::FRAME_LOCAL_ = "FRAME_LOCAL" ;
const char* NNodeEnum::FRAME_GLOBAL_ = "FRAME_GLOBAL" ;

const char* NNodeEnum::FrameType(NNodeFrameType fr)
{
    const char* s = NULL ;
    switch(fr)
    {
        case FRAME_MODEL: s = FRAME_MODEL_ ; break ; 
        case FRAME_LOCAL: s = FRAME_LOCAL_ ; break ; 
        case FRAME_GLOBAL: s = FRAME_GLOBAL_ ; break ; 
    }
    return s ;
}


const char* NNodeEnum::POINT_INSIDE_ = "POINT_INSIDE" ;
const char* NNodeEnum::POINT_SURFACE_ = "POINT_SURFACE" ;
const char* NNodeEnum::POINT_OUTSIDE_ = "POINT_OUTSIDE" ;

const char* NNodeEnum::PointType(NNodePointType pt)
{
    const char* s = NULL ;
    switch(pt)
    {
        case POINT_INSIDE: s = POINT_INSIDE_ ; break ; 
        case POINT_SURFACE: s = POINT_SURFACE_ ; break ; 
        case POINT_OUTSIDE: s = POINT_OUTSIDE_ ; break ; 
    }
    return s ;
}




const char* NNodeEnum::FEATURE_PARENT_LINKS_ = "FEATURE_PARENT_LINKS" ;
const char* NNodeEnum::FEATURE_GTRANSFORMS_ = "FEATURE_GTRANSFORMS" ;
const char* NNodeEnum::FEATURE_GTRANSFORM_IDX_ = "FEATURE_GTRANSFORM_IDX" ;

const char* NNodeEnum::Feature(NNodeFeature ft)
{
    const char* s = NULL ;
    switch(ft)
    {
        case FEATURE_PARENT_LINKS   : s = FEATURE_PARENT_LINKS_   ; break ; 
        case FEATURE_GTRANSFORMS    : s = FEATURE_GTRANSFORMS_    ; break ; 
        case FEATURE_GTRANSFORM_IDX : s = FEATURE_GTRANSFORM_IDX_ ; break ; 
    }
    return s ;
}






NNodePointType NNodeEnum::PointClassify( float sd, float epsilon )
{
    return fabsf(sd) < epsilon ? POINT_SURFACE : ( sd < 0 ? POINT_INSIDE : POINT_OUTSIDE ) ; 
}

std::string NNodeEnum::PointMask(unsigned mask)
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < 3 ; i++) 
    {
        NNodePointType pt = (NNodePointType)(0x1 << i) ;
        if( pt & mask ) ss << PointType(pt) << " " ;  
    }
    return ss.str();
}



