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
/**
SAbbrev
========

**/


#include <string>
#include <vector>
#include "plog/Severity.h"

#include "SYSRAP_API_EXPORT.hh"

class SASCII ; 


struct SYSRAP_API SAbbrev
{
    static const plog::Severity LEVEL ;
    static SAbbrev* Load(const char* path); 
    static SAbbrev* FromString( const char* str ); 

    SAbbrev( const std::vector<std::string>& names_ ); 

    void init(); 


    std::string form_candidate( const SASCII* n ); 

    bool isFree(const std::string& ab) const ;
    void dump() const ; 
    void save(const char* path_) const ; 
    void save(const char* fold, const char* name) const ; 

   
    const std::vector<std::string>& names ; 
    std::vector<std::string> abbrev ; 
};


