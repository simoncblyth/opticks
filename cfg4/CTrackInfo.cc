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

#include <sstream>
#include "PLOG.hh"
#include "CTrackInfo.hh"

/**
CTrackInfo
============

   0001   1
   0011   3
   0100   4
   0111   7  
   1000   8
   1111   f  

**/


const plog::Severity CTrackInfo::LEVEL = PLOG::EnvLevel("CTrackInfo", "DEBUG") ; 

CTrackInfo::CTrackInfo( unsigned photon_id_ , char gentype_, bool reemission_ )
    :   
    G4VUserTrackInformation("CTrackInfo"),
    trk(photon_id_, gentype_, reemission_ )
{   
    LOG(LEVEL) ; 
}   

CTrackInfo::~CTrackInfo()
{
}
G4String* CTrackInfo::type() const { return pType ; }

std::string CTrackInfo::desc() const 
{ 
    return trk.desc(); 
}


