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

#include "CStage.hh"


const char* CStage::UNKNOWN_ = "UNKNOWN" ;
const char* CStage::START_   = "START" ;
const char* CStage::COLLECT_ = "COLLECT" ;
const char* CStage::REJOIN_  = "REJOIN" ;
const char* CStage::RECOLL_  = "RECOLL" ;

const char* CStage::Label( CStage_t stage)
{
    const char* s = 0 ; 
    switch(stage)
    {
        case UNKNOWN:  s = UNKNOWN_ ; break ;
        case START:    s = START_   ; break ;
        case COLLECT:  s = COLLECT_ ; break ;
        case REJOIN:   s = REJOIN_  ; break ;
        case RECOLL:   s = RECOLL_  ; break ;
    } 
    return s ; 
}



