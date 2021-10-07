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

#include "X4_API_EXPORT.hh"

template <typename T> struct SLabelCache ; 

#include "plog/Severity.h"
#include <string>
#include <vector>
#include <map>

class G4Material ; 
class G4LogicalSurface ; 

/**
X4
===

BaseName  
    basename with pointer trimmed
BaseNameAsis  
    basename as found, no trimming 

GetItemIndex
    index of an item in an array 
GetOpticksIndex
    combined logical surface index with skin surfaces offset 
    by the number of border surfaces

Array
    codegen for small arrays
Value 
    fixed precision formatting of value




**/


class X4_API X4 
{
    public: 
        static const plog::Severity LEVEL ; 
        static SLabelCache<int>* surface_index_cache ; 
        static SLabelCache<int>* MakeSurfaceIndexCache() ; 
        static const int MISSING_SURFACE ; 
        static const char* PREFIX_G4Material ; 
    public: 
        static const char* X4GEN_DIR ; 
    public: 
        static const char* Name( const std::string& name );
        static const char* ShortName( const std::string& name );
        static const char* BaseName_( const std::string& name, const char* prefix);     
        static const char* BaseNameAsis_( const std::string& name, const char* prefix  );

        template<typename T> static const char* Name( const T* const obj ); 
        template<typename T> static const char* ShortName( const T* const obj ); 
        template<typename T> static const char* BaseName( const T* const obj ); 
        template<typename T> static const char* BaseNameAsis( const T* const obj ); 
        template<typename T> static const char* GDMLName( const T* const obj ); 
        
        template<typename T> static int GetItemIndex( const std::vector<T*>* vec, const T* const item ); 

        static size_t GetOpticksIndex( const G4LogicalSurface* const surf );

        template<typename T> static std::string Value( T v );  
        template<typename T> static std::string Argument( T v );  
        static std::string Array( const double* a, unsigned nv, const char* identifier );  

    private:
        std::map<void*,int>*  m_surface_index_cache ; 
};


