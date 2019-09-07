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

#include <string>

typedef enum
{  
   FRAME_MODEL, 
   FRAME_LOCAL, 
   FRAME_GLOBAL 

} NNodeFrameType ;


typedef enum
{  
   POINT_INSIDE  = 0x1 << 0, 
   POINT_SURFACE = 0x1 << 1, 
   POINT_OUTSIDE = 0x1 << 2  

} NNodePointType ;


typedef enum { 
   FEATURE_PARENT_LINKS = 0x1 << 0, 
   FEATURE_GTRANSFORMS  = 0x1 << 1,
   FEATURE_GTRANSFORM_IDX  = 0x1 << 2
} NNodeFeature  ; 


typedef enum
{  
   NODE_ALL, 
   NODE_OPERATOR, 
   NODE_PRIMITIVE 

} NNodeType ;


typedef enum 
{  
   PAIR_MINMIN = 0 , 
   PAIR_MINMAX = 1 , 
   PAIR_MAXMIN = 2 , 
   PAIR_MAXMAX = 3 
}  NNodePairType ;



typedef enum {
    JOIN_UNCLASSIFIED, 
    JOIN_COINCIDENT, 
    JOIN_OVERLAP, 
    JOIN_SPLIT
} NNodeJoinType ; 



typedef enum {
    NUDGE_NONE,
    NUDGE_J_DECREASE_Z1, 
    NUDGE_I_INCREASE_Z2 
} NNodeNudgeType ; 



#include "NPY_API_EXPORT.hh"

class NPY_API NNodeEnum
{
    public:
        static const char* FRAME_MODEL_ ;
        static const char* FRAME_LOCAL_;
        static const char* FRAME_GLOBAL_ ;
        static const char* FrameType(NNodeFrameType fr);

        static const char* POINT_INSIDE_;
        static const char* POINT_SURFACE_;
        static const char* POINT_OUTSIDE_;
        static const char* PointType(NNodePointType pt);

        static const char* FEATURE_PARENT_LINKS_ ;
        static const char* FEATURE_GTRANSFORMS_ ;
        static const char* FEATURE_GTRANSFORM_IDX_ ;
        static const char* Feature(NNodeFeature ft);

        static const char* PAIR_MINMIN_;
        static const char* PAIR_MINMAX_;
        static const char* PAIR_MAXMIN_;
        static const char* PAIR_MAXMAX_;
        static const char* PairType(NNodePairType pair);

        static const char* JOIN_UNCLASSIFIED_ ;
        static const char* JOIN_COINCIDENT_ ;
        static const char* JOIN_OVERLAP_ ;
        static const char* JOIN_SPLIT_ ;
        static const char* JoinType(NNodeJoinType join);
        static NNodeJoinType JoinClassify( float za, float zb, float epsilon);

        static const char* NUDGE_NONE_ ; 
        static const char* NUDGE_J_DECREASE_Z1_ ; 
        static const char* NUDGE_I_INCREASE_Z2_ ; 
        static const char* NudgeType(NNodeNudgeType nudge);  

        static NNodePointType PointClassify( float sdf_ , float epsilon );
        static std::string PointMask(unsigned mask);


};
