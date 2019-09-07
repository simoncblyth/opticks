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
#include <map>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"


template <typename A, typename B>
class BRAP_API BList {
   public:
      static void save( std::vector<std::pair<A,B> >* vec, const char* dir, const char* name );
      static void save( std::vector<std::pair<A,B> >* vec, const char* path );
      static void load( std::vector<std::pair<A,B> >* vec, const char* dir, const char* name );
      static void load( std::vector<std::pair<A,B> >* vec, const char* path );
      static void dump( std::vector<std::pair<A,B> >* vec, const char* msg="BList::dump");
   public:
      BList( std::vector<std::pair<A,B> >* vec );
   public:
      void save(const char* dir, const char* name);
      void save(const char* path);
      void load( const char* dir, const char* name) ;
      void load( const char* path) ;
      void dump( const char* msg="BList::dump" ) ;

   private:
      std::vector<std::pair<A,B> >* m_vec ; 
};


#include "BRAP_TAIL.hh"





