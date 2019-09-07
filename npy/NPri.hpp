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
#include <glm/fwd.hpp>

template <typename T> class NPY ; 
#include "NPY_API_EXPORT.hh"

/**
NPri
======

Wrapper for an (n,4,4) primary particle buffer providing higher level
accessors and dumping.

**/

class NPY_API NPri {
    public:  
        static void Dump(NPY<float>* ox, unsigned modulo, unsigned margin, const char* msg) ;
    public:  
        unsigned getNumG4Event() const ;
        NPri(const NPY<float>* primaries); 
    private:
        void init();   
    public:  
        const NPY<float>*     getPrimaries() const ;
        unsigned              getNumPri() const ;
    public:  
        glm::vec4             getPositionTime(unsigned i) const ; 
        glm::vec4             getDirectionWeight(unsigned i) const ; 
        glm::vec4             getPolarizationKineticEnergy(unsigned i) const ;
        glm::ivec4            getFlags(unsigned i) const ;

        int                   getEventIndex(unsigned i) const ; 
        int                   getVertexIndex(unsigned i) const ; 
        int                   getParticleIndex(unsigned i) const ;  // within the vertex 
        int                   getPDGCode(unsigned i) const ; 

        std::string           desc(unsigned i) const ;
        std::string           desc() const ;
        void                  dump(unsigned modulo, unsigned margin, const char* msg="NPri::dump") const ;
   private:
        const NPY<float>*     m_primaries ; 

};

