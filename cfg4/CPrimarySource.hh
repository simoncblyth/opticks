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

#include "CSource.hh"
#include "CFG4_API_EXPORT.hh"
#include <vector>

class G4Event ; 
class G4PrimaryVertex ;
class G4PrimaryParticle ;

class NPri ; 
template <typename T> class NPY ; 


/**
CPrimarySource
================

**/

class CFG4_API CPrimarySource: public CSource
{
    public:
        unsigned getNumG4Event() const ;
        CPrimarySource(Opticks* ok,  NPY<float>* input_primaries, int verbosity);
        virtual ~CPrimarySource();
    private:
        void init();
        void findVertices( std::vector<unsigned>& vtx_start, std::vector<unsigned>& vtx_count );
        G4PrimaryVertex*   makePrimaryVertex(unsigned idx) const ;
        G4PrimaryParticle* makePrimaryParticle(unsigned idx) const ;
    public:
        // G4VPrimaryGenerator interface
        void GeneratePrimaryVertex(G4Event *evt);
    public:
    private:
        NPri*                 m_pri ;
        unsigned              m_event_count ;   
        // event count should be in base class : but base needs a rewrite so leave it here for now


};


