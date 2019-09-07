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
#include <map>

class NMeta ; 

#include "NPY_API_EXPORT.hh"

/**
NPYMeta : integer keyed map of NMeta dicts used for node metadata
=====================================================================

Primary usage so far is as the m_meta instance of NCSG node trees,
providing per-node metadata for the trees.

**/

class NPY_API NPYMeta
{
    public:
        static NMeta*       LoadMetadata(const char* treedir, int item=-1);
        static bool         ExistsMeta(const char* treedir, int item=-1);
    private:
        static const char*  META ; 
        static const char*  ITEM_META ; 
        static std::string  MetaPath(const char* treedir, int item=-1);
        enum { NUM_ITEM = 16 } ;  // default number of items to look for
    public:
        // item -1 corresponds to global metadata 
        NPYMeta(); 
        NMeta*  getMeta(int item=-1) const ;   
        bool          hasMeta(int idx) const ;
    public:
        int                       getIntFromString(const char* key, const char* fallback, int item=-1 ) const ;
        template<typename T> T    getValue(const char* key, const char* fallback, int item=-1 ) const ;
        template<typename T> void setValue(const char* key, T value, int item=-1);
    public:
        void load(const char* dir, int num_item = NUM_ITEM ) ;
        void save(const char* dir) const ;
    private:
        std::map<int, NMeta*>    m_meta ;    
        // could be a complete binary tree with loadsa nodes, so std::array not appropriate

};



