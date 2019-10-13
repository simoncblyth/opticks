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

class GItemIndex ; 

class Opticks ; 
class OpticksHub ; 
class OpticksRun ; 
class OpticksEvent ; 

class OpticksAttrSeq ;

#include "plog/Severity.h"


#include "OKGEO_API_EXPORT.hh"

/**
OpticksIdx
===========

Wrapper around hostside(only?) indexing functionality


**/


class OKGEO_API OpticksIdx {
       static const plog::Severity LEVEL ; 
   public:
       OpticksIdx(OpticksHub* hub);
   public:
       // presentation prep
       GItemIndex* makeHistoryItemIndex();
       GItemIndex* makeMaterialItemIndex();
       GItemIndex* makeBoundaryItemIndex();

   public:
       // used for GUI seqmat and boundaries presentation
       OpticksAttrSeq*  getMaterialNames();
       OpticksAttrSeq*  getBoundaryNames();
       std::map<unsigned int, std::string> getBoundaryNamesMap();

   public:
       // hostside indexing 
       void indexEvtOld();
       void indexBoundariesHost();
       void indexSeqHost();
   private:
        // from OpticksRun, uncached
        OpticksEvent* getEvent();
   private:
        OpticksHub*    m_hub ; 
        Opticks*       m_ok ; 
        OpticksRun*    m_run ; 

};



 
