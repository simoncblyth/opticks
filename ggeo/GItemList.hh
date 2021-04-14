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
#include <vector>
#include <map>

struct NSlice ;
#include "NSequence.hpp"

#include "plog/Severity.h"

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GItemList : public NSequence {
   public:
       static unsigned int UNSET ; 
       static const char* GITEMLIST ; 
       static const plog::Severity LEVEL ; 
       static GItemList* Load(const char* idpath, const char* itemtype, const char* reldir=NULL);
       static GItemList* Repeat( const char* itemtype, const char* name, unsigned numRepeats, const char* reldir=NULL );
   public:
       GItemList(const char* itemtype, const char* reldir=NULL);
       void add(const char* name);
       void add(GItemList* other);
       void save(const char* idpath);
       void save(const char* idpath, const char* reldir, const char* txtname);  // for debug 
       void dump(const char* msg="GItemList::dump");
       const std::string& getRelDir() const ;  
    public:
       GItemList* make_slice(const char* slice);
       GItemList* make_slice(NSlice* slice);
    public:
       void dumpFields(const char* msg="GItemList::dumpFields", const char* delim="/", unsigned int fwid=30);
       void replaceField(unsigned int field, const char* from, const char* to, const char* delim="/");
    public:
       unsigned int getNumItems();
    public:
       // fulfil NSequence protocol
       const char* getKey(unsigned index) const ;
       unsigned getNumKeys() const ;
       unsigned getNumUniqueKeys() const ; 
    public:
       void setKey(unsigned int index, const char* newkey);
       static bool isUnset(unsigned int index);
   public:
       void getIndicesWithKey( std::vector<unsigned>& indices, const char* key ) const ; 
       void getIndicesWithKeyEnding( std::vector<unsigned>& indices, const char* ending ) const ;  
       void getIndicesWithKeyStarting( std::vector<unsigned>& indices, const char* key_start ) const ; 
       int  findIndexWithKeyStarting( const char* starting ) const ;  // first index is returned, gives -1 if none found
       int  findIndex( const char* key ) const ;  // first index is returned, gives -1 if none found
       unsigned getIndex(const char* key) const ;    // 0-based index of first matching name, OR UINT_MAX if no match
   public:
       bool operator()(const std::string& a_, const std::string& b_);
       void setOrder(std::map<std::string, unsigned int>& order);
       void sort();
   public:
       void getCurrentOrder( std::map<std::string, unsigned int>& order );
   private:
       void save_(const char* txtpath);
       void read_(const char* txtpath);
       void load_(const char* idpath);
   private:
       std::string              m_itemtype ;
       std::string              m_reldir ;
       std::vector<std::string> m_list ; 
       std::map<std::string, unsigned int> m_order ; 
       std::string              m_empty ; 
};

#include "GGEO_TAIL.hh"

