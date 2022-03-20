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
#include "plog/Severity.h"

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

#include "json.hpp"


/**
BMeta (formerly NMeta)
========================

TODO: there is a more lightweight metadata class sysrap/SMeta that does not use boost, 
      are aiming to replace use of BMeta with SMeta where possible by migrating 
      functionality downwards : be guided is this by features needed by users 
      which also have downwards pressure like okc/OpticksFlags 


Metadata persistency using nlohmann::json single header which 

* https://github.com/nlohmann/json
* https://nlohmann.github.io/json/

* BMeta adds a limitation of only handling keyed structures, not lists
* newer versions has a "contains(key)" method that may be handy 

In a prior life this was npy/NMeta however as it depended
very little on NPY but used file handling from BoostRap it 
has been moved down in the heirarchy to BMeta. In addition NMeta 
depended on an ancient nlogmann::json that came along with YoctoGL.
Instead this directly uses a recent header from nljson- external.

**/

class BRAP_API BMeta {
       static const plog::Severity LEVEL ; 
   public:
       static BMeta* Load(const char* path);
       static BMeta* Load(const char* dir, const char* name);
       static BMeta* FromTxt(const char* txt);
   public:
       BMeta();
       BMeta(const BMeta& other);

       void append(BMeta* other); // duplicate keys are overwritten

       unsigned size() const ; 

       nlohmann::json& js();
       const nlohmann::json& cjs() const ;
   public:
       const char* getKey(unsigned idx) const ;
       const char* getValue(unsigned idx) const ;

       const char* getKey_old(unsigned idx) const ;
       unsigned    getNumKeys_old() ;             // non-const as may updateKeys
       unsigned    getNumKeys() const ;           // assumes obj 
       void        getKV(unsigned i, std::string& k, std::string& v ) const ; 

       std::vector<std::string>& getLines();  // non-const may prepLines
       std::string desc(unsigned wid=0);
       void fillMap(std::map<std::string, std::string>& mss, bool dump=false ); 
   private:
       void        updateKeys();
       void        prepLines();

   public:
       void   setObj(const char* name, BMeta* obj); 
       BMeta* getObj(const char* name) const ;
   public:
       template <typename T> void add(const char* name, T value);   // same as set, for easier migration for B_P_a_r_a_m_e_t_e_r_s
       template <typename T> void set(const char* name, T value);

       void appendString(const char* name, const std::string& value, const char* delim=" ");

       template <typename T> T get(const char* name) const ;
       template <typename T> T get(const char* name, const char* fallback) const ;
       int getIntFromString(const char* name, const char* fallback) const ;

   public:

       bool hasItem(const char* name) const ;
       bool hasKey(const char* key) const ; // same as hasItem

       void kvdump() const ;

   public:
       template <typename T> static T Get(const BMeta* meta, const char* name, const char* fallback)  ;
   public:
       void save(const char* path) const ;
       void save(const char* dir, const char* name) const ;
       void dump() const ; 
       void dump(const char* msg) const ; 
       void dumpLines(const char* msg="BMeta::dumpLines") ; 
   public:
       void addEnvvar( const char* key ) ;
       void addEnvvarsWithPrefix( const char* prefix="OPTICKS_", bool trim=true );  



   public:
       void load(const char* path);
       void load(const char* dir, const char* name);
       void loadTxt(const char* txt);

   private:
       // formerly used separate NJS, but that makes copy-ctor confusing 
       void read(const char* path0, const char* path1=NULL);
       void write(const char* path0, const char* path1=NULL) const ;
       void readTxt(const char* txt);
 
   private:
       nlohmann::json  m_js ;  
       std::vector<std::string> m_keys ; 
       std::vector<std::string> m_lines ; 

};

#include "BRAP_TAIL.hh"


