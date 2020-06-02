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

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cassert>

#include "SASCII.hh"
#include "SAbbrev.hh"

#include "PLOG.hh"

const plog::Severity SAbbrev::LEVEL = PLOG::EnvLevel("SAbbrev", "DEBUG") ;

SAbbrev::SAbbrev( const std::vector<std::string>& names_ ) 
    :
    names(names_)
{
    init();
}

bool SAbbrev::isFree(const std::string& ab) const 
{
    return std::find( abbrev.begin(), abbrev.end(), ab ) == abbrev.end() ; 
}

void SAbbrev::init()
{
    for(unsigned i=0 ; i < names.size() ; i++)
    {
        const char* name = names[i].c_str(); 

        SASCII n(name);  

        std::string ab ; 

        if( n.upper > 1 )
        { 
            ab = n.getFirstUpper(2) ; 
        }
        else 
        {
            ab = n.getFirst(2) ; 
        }

        LOG(LEVEL) 
            << " name [" << name << "]" 
            << " ab [" << ab << "]" 
            ;

        if(!isFree(ab))
        {
            ab = n.getFirstLast(); 
        } 

        bool is_now_free = isFree(ab) ;  

        LOG(LEVEL) 
            << " name [" << name << "]" 
            << " ab [" << ab << "]" 
            << " is_now_free " << is_now_free
            ;

        assert( is_now_free && "failed to abbreviate "); 
        abbrev.push_back(ab) ;  
    }
}

void SAbbrev::dump() const 
{
    for(unsigned i=0 ; i < names.size() ; i++)
    {
         std::cout 
             << std::setw(30) << names[i]
             << " : " 
             << std::setw(2) << abbrev[i]
             << std::endl 
             ;
    }
}




