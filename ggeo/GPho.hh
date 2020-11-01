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
#include "plog/Severity.h"

template <typename T> class NPY ; 
class GGeo ; 

#include "GGEO_API_EXPORT.hh"

/**
GPho
======

Wrapper for an (n,4,4) photon buffer providing higher level
accessors and dumping with use of GGeo for access to transforms
enabling local frame positions, directions and polarizations
to be provided.

NPho provides similar functionality but without access
to geometry transforms.

**/

class GGEO_API GPho {
    private:
        static const plog::Severity LEVEL ; 
        static const char* A ; 
        static const char* L ; 
        static const char* H ; 
    public:  
        static void Dump(const NPY<float>* ox, const GGeo* ggeo, unsigned maxDump=0, const char* opt="post,dirw,flgs") ;
        static void Dump(const NPY<float>* ox, const GGeo* ggeo,  unsigned modulo, unsigned margin, const char* opt="post,dirw,flgs") ;
    public:  
        void        setSelection(char selection) ;
        const char* getSelectionName() const ;
    public:  
        GPho(const GGeo* ggeo, const char* opt="mski,post,dirw,polw,flgs"); 
        void setPhotons(const NPY<float>* photons); 
    public:  
        const NPY<float>*     getPhotons() const ;
        unsigned              getNumPhotons() const ;
    public:  
        bool                  isLandedOnSensor(unsigned i) const ;  // not necessarily a hit 
        bool                  isHit(unsigned i) const ; 
    public:  
        glm::vec4             getPositionTime(unsigned i) const ; 
        glm::vec4             getDirectionWeight(unsigned i) const ;   // weight is stomped upon, holding unsigned_as_float(nidx)
        glm::vec4             getPolarizationWavelength(unsigned i) const ;
        glm::ivec4            getFlags(unsigned i) const ;
    public:  
        // GGeo info on volume which the ray intersected last   
        unsigned              getLastIntersectNodeIndex(unsigned i) const ;
        glm::uvec4            getLastIntersectNRPO(unsigned i) const ;
        glm::mat4             getLastIntersectTransform(unsigned i) const ; 
        glm::mat4             getLastIntersectInverseTransform(unsigned i) const ;
    public:  
        glm::vec4             getLocalPositionTime(unsigned i) const ;
        glm::vec4             getLocalDirectionWeight(unsigned i) const ;
        glm::vec4             getLocalPolarizationWavelength(unsigned i) const ;
        NPY<float>*           makeLocalPhotons() const ;
        void                  saveLocalPhotons(const char* path) const ;
    public:  
        std::string           desc(unsigned i) const ;
        std::string           desc() const ;
        void                  dump(unsigned modulo, unsigned margin, const char* msg="NPho::dump") const ;
        void                  dump(const char* msg="NPho::dump", unsigned maxDump=0) const ; 
   private:
       // these three are set by setPhotons 
       const NPY<float>*      m_photons ; 
       const NPY<unsigned>*   m_msk ; 
       unsigned               m_num_photons ; 
   private:
       const GGeo*            m_ggeo ; 
       const char*            m_opt ; 
       char                   m_selection ; 
   private:
       // dumping options
       bool                   m_nidx ; 
       bool                   m_nrpo ; 
       bool                   m_mski ; 
       bool                   m_post ; 
       bool                   m_lpst ; 
       bool                   m_ldrw ; 
       bool                   m_lpow ; 
       bool                   m_dirw ; 
       bool                   m_polw ; 
       bool                   m_flgs ; 
};

