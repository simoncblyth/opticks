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

#include "OpticksPhoton.h"
#include <string>
#include <map>
#include "plog/Severity.h"

class Index ; 
class BMeta ; 

/**
OpticksFlags
=============

Much of the former OpticksFlags has been migrated down to sysrap/OpticksPhoton.hh
leaving up here only the aspects that use BMeta and npy/Index,NSequence.

TO-CONSIDER migrate the entire thing down to sysrap:

1. replacing BMeta with SMeta 
2. npy/Index.hpp NSequence.hpp 


Canonical m_flags resides in OpticksResource and is 
instanciated with it.

OpticksPhoton.h enum header is parsed at instanciation, loading 
names and enum values into an Index m_index instance
see: OpticksFlagsTest --OKCORE debug 

Actually the index is little used, the static methods using 
case statement conversions being more convenient.


Profit from repositioning ?

* this needs to sink from okc- together with OpticksPhoton.h 
  so can use from NPY (for getting rid of Types.hpp usage from RecordsNPY for example)

* Index prevents below NPY, SBit BRegex beneath brap- but 
  most of the utility could work from lowest level sysrap- 
  SFlags.hh ? 


**/

#include "OKCORE_API_EXPORT.hh"

class OKCORE_API OpticksFlags {
       static const plog::Severity LEVEL ; 
    public:
       static const char* ABBREV_META_NAME ;  
       static const char* ENUM_HEADER_PATH ;  
    public:
       static const char* SourceType(int code);          // OpticksGenstep::Gentype
       static unsigned int SourceCode(const char* type); // OpticksGenstep::SourceCode


    public:
        OpticksFlags(const char* path=ENUM_HEADER_PATH);
        void save(const char* installcachedir);
    private:
        Index* parseFlags(const char* path);
        static BMeta* MakeAbbrevMeta(); 
        static BMeta* MakeFlag2ColorMeta(); 
    public:
        Index*             getIndex() const ;  
        BMeta*             getAbbrevMeta() const ; 
        BMeta*             getColorMeta() const ; 
    private:
        Index*             m_index ; 
        BMeta*             m_abbrev_meta ;  
        BMeta*             m_color_meta ;  
};

 
