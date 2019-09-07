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

class NGunConfig ; 

#include "CSource.hh"
#include "CFG4_API_EXPORT.hh"

/**
CGunSource
============

Converts NGunConfig into G4VPrimaryGenerator 
with GeneratePrimaryVertex(G4Event \*evt)

**/


class CFG4_API CGunSource: public CSource
{
    public:
        CGunSource(Opticks* ok);
        virtual ~CGunSource();
        void configure(NGunConfig* gc);
    public:
        void GeneratePrimaryVertex(G4Event* event);
    private:
        NGunConfig*   m_config ; 

};


