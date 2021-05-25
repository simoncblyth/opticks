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

#include "G4VUserTrackInformation.hh"

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

/**
CTrackInfo
============

**/

struct CFG4_API CTrackInfo : public G4VUserTrackInformation
{
    CTrackInfo( unsigned record_id_ , char gentype_  )
        :   
        packed((record_id_ & 0x7fffffff) | unsigned(gentype_ == 'C') << 31 )   
    {   
    }   
    unsigned packed  ;   

    char gentype() const       { return ( packed & 0x80000000 ) ? 'C' : 'S' ;  }
    unsigned record_id() const { return ( packed & 0x7fffffff ) ; } 
};


#include "CFG4_TAIL.hh"
