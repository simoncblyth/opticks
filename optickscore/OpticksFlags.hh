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
class NMeta ; 

/**
OpticksFlags
=============

Canonical m_flags resides in OpticksResource and is 
instanciated with it.


OpticksPhoton.h enum header is parsed at instanciation, loading 
names and enum values into an Index m_index instance
see: OpticksFlagsTest --OKCORE debug 

Actually the index is little used, the static methods using 
case statement conversions being more convenient.


TODO : Repositioning 
-----------------------

* this needs to sink from okc- together with OpticksPhoton.h 
  so can use from NPY (for getting rid of Types.hpp usage from RecordsNPY for example)

* Index prevents below NPY, BBit BRegex beneath brap- but 
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
       static const char* CERENKOV_ ;
       static const char* SCINTILLATION_ ;
       static const char* NATURAL_ ;
       static const char* FABRICATED_ ;
       static const char* MACHINERY_ ;
       static const char* TORCH_ ;
       static const char* G4GUN_ ;   
       static const char* EMITSOURCE_ ;   
       static const char* PRIMARYSOURCE_ ;   
       static const char* GENSTEPSOURCE_ ;   
       static const char* OTHER_ ;
    public:
       static const char* cerenkov_ ;
       static const char* scintillation_ ;
       static const char* natural_ ;
       static const char* fabricated_ ;
       static const char* machinery_ ;
       static const char* torch_ ;
       static const char* g4gun_ ;
       static const char* emitsource_ ;
       static const char* primarysource_ ;
       static const char* genstepsource_ ;
       static const char* other_ ;
    public:
       static const char* ZERO_ ;
       static const char* MISS_ ;
       static const char* BULK_ABSORB_ ;
       static const char* BULK_REEMIT_ ;
       static const char* BULK_SCATTER_ ;
       static const char* SURFACE_DETECT_ ;
       static const char* SURFACE_ABSORB_ ;
       static const char* SURFACE_DREFLECT_ ;
       static const char* SURFACE_SREFLECT_ ;
       static const char* BOUNDARY_REFLECT_ ;
       static const char* BOUNDARY_TRANSMIT_ ;
       static const char* NAN_ABORT_ ;
       static const char* BAD_FLAG_ ;
    public:
       static const char* _ZERO ;
       static const char* _NATURAL ;
       static const char* _FABRICATED ;
       static const char* _MACHINERY ;
       static const char* _G4GUN ;   
       static const char* _EMITSOURCE ;   
       static const char* _PRIMARYSOURCE ;   
       static const char* _GENSTEPSOURCE ;   
       static const char* _CERENKOV ;
       static const char* _SCINTILLATION ;
       static const char* _TORCH ;
       static const char* _MISS ;
       static const char* _BULK_ABSORB ;
       static const char* _BULK_REEMIT ;
       static const char* _BULK_SCATTER ;
       static const char* _SURFACE_DETECT ;
       static const char* _SURFACE_ABSORB ;
       static const char* _SURFACE_DREFLECT ;
       static const char* _SURFACE_SREFLECT ;
       static const char* _BOUNDARY_REFLECT ;
       static const char* _BOUNDARY_TRANSMIT ;
       static const char* _NAN_ABORT ;
       static const char* _BAD_FLAG ;
    public:
       static const char* SourceType(int code);
       static const char* SourceTypeLowercase(int code);
       static unsigned int SourceCode(const char* type);
    public:
       static const char* Flag(const unsigned flag);
       static const char* Abbrev(const unsigned flag);
    public:
       static unsigned EnumFlag(unsigned bitpos);
       static unsigned BitPos(unsigned flag);
       static unsigned AbbrevToFlag( const char* abbrev );
       static unsigned long long AbbrevToFlagSequence( const char* abbseq, char delim=' ');
       static unsigned AbbrevSequenceToMask( const char* abbseq, char delim=' ');
       static void AbbrevToFlagValSequence( unsigned long long& seqhis, unsigned long long& seqval, const char* seqmap, char edelim=' ') ;

       static unsigned PointVal1( const unsigned long long& seqval , unsigned bitpos );
       static unsigned PointFlag( const unsigned long long& seqhis , unsigned bitpos );
       static const char* PointAbbrev( const unsigned long long& seqhis , unsigned bitpos );

    public:
       static std::string FlagSequence(const unsigned long long seqhis, bool abbrev=true, int highlight=-1);
       static std::string FlagMask(const unsigned mskhis, bool abbrev=true);
    public:
        OpticksFlags(const char* path=ENUM_HEADER_PATH);
        void save(const char* installcachedir);
    private:
        Index* parseFlags(const char* path);
        static NMeta* MakeAbbrevMeta(); 
        static NMeta* MakeFlag2ColorMeta(); 
    public:
        Index*             getIndex() const ;  
        NMeta*             getAbbrevMeta() const ; 
        NMeta*             getColorMeta() const ; 
    private:
        Index*             m_index ; 
        NMeta*             m_abbrev_meta ;  
        NMeta*             m_color_meta ;  
};

 
