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

#include "regexsearch.hh"
#include "fsutil.hh"
#include <cstdio>
#include <vector>

int main(int argc, char** argv)
{
    printf("%s\n", argv[0]);

    // observe special casing of getenv("HOME") on mingw 
    // so better to avoid HOME ? 

    std::vector<std::string> ss ; 
    ss.push_back("$OPTICKS_HOME/optickscore/OpticksPhoton.h");
    ss.push_back("$HOME/.opticks/GColors.json");
    ss.push_back("$HOME/.opticks");
    ss.push_back("$HOME/");
    ss.push_back("$HOME");
    ss.push_back("$OPTICKS_HOME");
    ss.push_back("$HOME/$OPTICKS_HOME");

    for(unsigned int i=0 ; i < ss.size() ; i++)
    {

       std::string s = ss[i] ;
       //std::string x = s ;
       //std::string x = os_path_expandvars(s.c_str() );
       std::string x = fsutil::FormPath(s.c_str() );
       printf("  [%s] -->  [%s] \n", s.c_str(), x.c_str());
    }


    return 0 ; 
}
