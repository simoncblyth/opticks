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
#include "plog/Severity.h"
#include "G4VUserTrackInformation.hh"

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API CTrackInfo : public G4VUserTrackInformation
{
        static const plog::Severity LEVEL ; 
    public:
        CTrackInfo( unsigned photon_id_ , char gentype_, bool reemission_ );
        virtual ~CTrackInfo(); 

        unsigned    packed()     const ; 
        unsigned    photon_id()  const ;
        char        gentype()    const ;
        bool        reemission() const ;
        G4String*   type()       const ; 
        std::string desc()       const ; 
    private:
        unsigned m_packed  ;   

};

#include "CFG4_TAIL.hh"

