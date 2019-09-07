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

#include <sstream>
#include <iostream>
#include <string>
#include <map>
#include <vector>

//#include <boost/regex.hpp>
#include "BRegex.hpp"

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

class BRAP_API BRegex {
   public:
      typedef std::pair<std::string, std::string>   pair_t ;
      typedef std::vector<pair_t>                   pairs_t ;

      typedef std::pair<int, std::string>           ipair_t ;
      typedef std::vector<ipair_t>                  ipairs_t ;

      typedef std::pair<unsigned int, std::string>  upair_t ;
      typedef std::vector<upair_t>                  upairs_t ;


      static void regexsearch( pairs_t& pairs, std::istream& is, boost::regex& e );
      static void enum_read(std::map<std::string, unsigned int>& emap, const char* path);
      static void enum_regexsearch( upairs_t& upairs, const char* path );

      static std::string os_path_expandvars(const char* s, bool debug=false);
      static std::string regex_extract_quoted(const char* line);
      static std::string regex_matched_element(const char* line);

      static void dump(  pairs_t& pairs, const char* msg="dump" );
      static void udump( std::vector<std::pair<unsigned int, std::string> >& pairs, const char* msg="udump");

};

#include "BRAP_TAIL.hh"



