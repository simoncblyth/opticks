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

#include <map>
#include <boost/property_tree/ptree.hpp>
#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

template <typename A, typename B>
class BRAP_API  BMap {
   public:
      static void save( std::map<A,B>* , const char* dir, const char* name) ;
      static void save( std::map<A,B>* , const char* path) ;
      static int  load( std::map<A,B>* , const char* dir, const char* name, unsigned int depth=0) ;
      static int  LoadJSONString( std::map<A,B>* , const char* json, unsigned int depth=0) ;
      static int  load( std::map<A,B>* , const char* path, unsigned int depth=0 ) ;
      static void dump( std::map<A,B>* , const char* msg="BMap::dump");
   public:
      BMap( std::map<A,B>* mp );
   public:
      void save(const char* dir, const char* name);
      void save(const char* path);
      int load( const char* dir, const char* name, unsigned int depth=0) ;
      int load( const char* path, unsigned int depth=0) ;
      int loadJSONString( const char* json, unsigned int depth=0) ;

      void dump( const char* msg="BMap::dump" ) ;
      void import(const boost::property_tree::ptree& t, unsigned depth);


   private:
      std::map<A,B>* m_map ; 
};

#include "BRAP_TAIL.hh"
 
