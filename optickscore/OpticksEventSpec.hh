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

#include <cstddef>
#include <string>

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class OpticksEventSpec ; 

/**
OpticksEventSpec
==================

Base clase of OpticksEvent.

Instances m_spec (positive tag) and m_nspec (negative tag) 
are instanciated by Opticks::defineEventSpec as part of Opticks::configure.



**/

class OKCORE_API OpticksEventSpec {
   public:
        static const char* OK_ ; 
        static const char* G4_ ; 
        static const char* NO_ ; 
   public:
        OpticksEventSpec(OpticksEventSpec* spec);
        OpticksEventSpec(const char* pfx, const char* typ, const char* tag, const char* det, const char* cat=NULL);
        OpticksEventSpec* clone(unsigned tagoffset=0) const ;   // non-zero tagoffset increments if +ve, and decrements if -ve
        void Summary(const char* msg="OpticksEventSpec::Summary") const ;
        std::string brief() const ;
        bool isG4() const ;
        bool isOK() const ;
        const char*  getEngine() const ;
   public:
        const char*  getPfx() const ;
        const char*  getTyp() const ;
        const char*  getTag() const ;
        const char*  getDet() const ;
        const char*  getCat() const ;
        const char*  getUDet() const ;
   private:
        const char*  formDir() const ; 
        const char*  formFold() const ; 
        const char*  formRelDir() const ; 
   public:
        const char*  getDir() ;
        const char*  getRelDir() ; // without the base, ie returns directory portion starting "evt/"
        const char*  getFold() ;   // one level above Dir without the tag 
   public:
        int          getITag() const ;
   protected:
        const char*  m_pfx ; 
        const char*  m_typ ; 
        const char*  m_tag ; 
        const char*  m_det ; 
        const char*  m_cat ; 
        const char*  m_udet ; 
   private:
        const char*  m_dir ; 
        const char*  m_reldir ; 
        const char*  m_fold ; 

        int          m_itag ; 
};

#include "OKCORE_TAIL.hh"


