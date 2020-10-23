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
NPho
======

Wrapper for an (n,4,4) photon buffer providing higher level
accessors and dumping.

BUT : being in NPY means have no access to geometry, 
so cannot convert global positions to local at this level.

**/

class NPY_API NPho {
    public:  
        static void Dump(NPY<float>* ox, unsigned maxDump=0, const char* opt="post,dirw,flgs") ;
        static void Dump(NPY<float>* ox, unsigned modulo, unsigned margin, const char* opt="post,dirw,flgs") ;
    public:  
        NPho(NPY<float>* photons, const char* opt="mski,post,dirw,polw,flgs"); 
    private:
        void init();   
    public:  
        NPY<float>*           getPhotons() const ;
        unsigned              getNumPhotons() const ;
    public:  
        glm::vec4             getPositionTime(unsigned i) const ; 
        glm::vec4             getDirectionWeight(unsigned i) const ; 
        glm::vec4             getPolarizationWavelength(unsigned i) const ;
        glm::ivec4            getFlags(unsigned i) const ;
        std::string           desc(unsigned i) const ;
        std::string           desc() const ;
        void                  dump(unsigned modulo, unsigned margin, const char* msg="NPho::dump") const ;
        void                  dump(const char* msg="NPho::dump", unsigned maxDump=0) const ; 
   private:
       NPY<float>*            m_photons ; 
       NPY<unsigned>*         m_msk ; 
       unsigned               m_num_photons ; 
       unsigned               m_num_msk ; 

       bool                   m_mski ; 
       bool                   m_post ; 
       bool                   m_dirw ; 
       bool                   m_polw ; 
       bool                   m_flgs ; 
};

