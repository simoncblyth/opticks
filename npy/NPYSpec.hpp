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

#include "NPYBase.hpp"
#include "NPY_API_EXPORT.hh"

// TODO: maybe type enum and statics should live in here to simplify NPYBase

class NPY_API NPYSpec {
   public:
        NPYSpec(const char* name, unsigned int ni, unsigned int nj, unsigned int nk, unsigned int nl, unsigned int nm, NPYBase::Type_t type, const char* ctrl, bool optional=false, int verbosity=0);
        void setNumItems(unsigned ni) ; 

        NPYSpec* clone() const ; 
        NPYBase::Type_t getType() const ;

        const char*     getName() const ;
        const char*     getTypeName() const ;
        const char*     getCtrl() const ;
        unsigned int    getDimension(unsigned int i) const ;
        int             getVerbosity() const ; 
        bool isOptional() const ; 
        bool isEqualTo(const NPYSpec* other) const ;
        bool isSameItemShape(const NPYSpec* other) const ; 
        void dumpComparison(const NPYSpec* other, const char* msg) const ; 

        std::string     description() const  ;
        std::string     desc() const  ;
        void Summary(const char* msg="NPYSpec::Summary") const ;

   private:
        const char*      m_name ; 
        unsigned         m_ni ; 
        unsigned         m_nj ; 
        unsigned         m_nk ; 
        unsigned         m_nl ; 
        unsigned         m_nm ; 
        unsigned int     m_bad_index ; 
        NPYBase::Type_t  m_type ; 
        const char*      m_ctrl  ; 
        bool             m_optional ; 
        int              m_verbosity ; 
};


 
