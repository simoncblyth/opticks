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

#include <vector>
#include <string>
#include <array>
#include <map>

class NPYBase ; 
class NPYSpec ; 
class NPYSpecList ; 


#include "NPY_API_EXPORT.hh"

/**
NPYList
=========

**/

class NPY_API NPYList
{
    public:
        enum { MAX_BUF = 16 } ;
    public:
        NPYList(const NPYSpecList* specs);
    private:
        void init();
    public:
        NPYBase::Type_t getBufferType( int bid ) const ;
        const char*     getBufferName( int bid ) const ;
        const NPYSpec*  getBufferSpec( int bid ) const ;
        std::string     getBufferPath( const char* treedir, int bid ) const ;
        std::string     desc() const ; 
    public:
        void            setLocked(int bid, bool locked=true) ; 
        bool            isLocked(int bid) ; 
        NPYBase*        getBuffer(int bid) const  ; 
        std::string     getBufferShape(int bid) const  ;
        unsigned        getNumItems(int bid) const  ;
    public:
        void            saveBuffer(const char* treedir, int bid , const char* msg=NULL ) const ;
        void            initBuffer(int bid, int ni, bool zero, const char* msg=NULL) ; 
        void            setBuffer(int bid, NPYBase* buffer , const char* msg=NULL ) ; 
        void            loadBuffer(const char* treedir, int bid , const char* msg=NULL ) ;
    private:
        const NPYSpecList*               m_specs ; 
        std::array<NPYBase*, MAX_BUF>    m_buf ;
        std::array<bool,     MAX_BUF>    m_locked ;



}; 
 
