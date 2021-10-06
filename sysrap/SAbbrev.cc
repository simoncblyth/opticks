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
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cassert>

#include "SPath.hh"
#include "SASCII.hh"
#include "SAbbrev.hh"
#include "SDice.hh"

#include "PLOG.hh"

const plog::Severity SAbbrev::LEVEL = PLOG::EnvLevel("SAbbrev", "DEBUG") ;

SAbbrev* SAbbrev::FromString(const char* str) // static
{
    std::vector<std::string>* nms = new std::vector<std::string>() ; 

    std::stringstream ss(str) ;    
    std::string line ; 
    while (std::getline(ss, line))  // newlines are swallowed by getline
    {   
        if(line.empty()) continue ;   
        //std::cout << "[" << line << "]" << std::endl ; 
        nms->push_back(line); 
    }    

    const std::vector<std::string>& names = *nms ; 
    return new SAbbrev(names); 
}


SAbbrev* SAbbrev::Load(const char* path_) // static
{
    const char* path = SPath::Resolve(path_); 
    std::ifstream fp(path, std::ios::in);
    if(fp.fail()) return nullptr ;  

    std::vector<std::string>* nms = new std::vector<std::string>() ; 

    std::stringstream ss ;     
    std::string line ; 
    while (std::getline(fp, line))  // newlines are swallowed by getline
    {    
        nms->push_back(line); 
    }    
    
    const std::vector<std::string>& names = *nms ; 
    return new SAbbrev(names); 
}





SAbbrev::SAbbrev( const std::vector<std::string>& names_ ) 
    :
    names(names_)
{
    init();
}

void SAbbrev::save(const char* fold, const char* name) const 
{
    int rc = SPath::MakeDirs(fold); assert( rc == 0 ); 
    const char* path = SPath::Resolve(fold, name); 
    save(path); 
}

void SAbbrev::save(const char* path_) const 
{
    const char* path = SPath::Resolve(path_); 
    LOG(LEVEL) << path ; 
    std::ofstream fp(path, std::ios::out);
    for(unsigned i=0 ; i < names.size() ; i++ ) fp << names[i] << std::endl ; 
}




bool SAbbrev::isFree(const std::string& ab) const 
{
    return std::find( abbrev.begin(), abbrev.end(), ab ) == abbrev.end() ; 
}

void SAbbrev::init()
{
    for(unsigned i=0 ; i < names.size() ; i++) LOG(LEVEL) << names[i].c_str();
 
    for(unsigned i=0 ; i < names.size() ; i++)
    {
        const char* name = names[i].c_str(); 
        SASCII* n = new SASCII(name); 

        int chars_after_other = n->other == 1 ? strlen(name) - n->first_other_index - 1 : -1 ;   
        bool underscored = chars_after_other > 2 ;  

        // when have an underscore abbreviate the string after it 
        if( underscored )
        {   
            int idx = n->first_other_index ;  
            delete n ; 
            n = new SASCII(name+idx+1) ;   
        }

        std::string ab = form_candidate(n); 

        bool is_now_free = isFree(ab) ;  

        if(!is_now_free)
        {
            save("$TMP", "SAbbrev_FAIL.txt"); 
        }

        LOG(LEVEL) 
            << " name [" << name << "]" 
            << " ab [" << ab << "]" 
            << " is_now_free " << is_now_free
            ;

        assert( is_now_free && "failed to abbreviate "); 
        abbrev.push_back(ab) ;  
    }
}


std::string SAbbrev::form_candidate( const SASCII* n )
{
    std::string ab ; 

    if( n->upper > 0 && n->number > 0 ) // 1 or more upper and number 
    {
        int iu = n->first_upper_index ; 
        int in = n->first_number_index ; 
        ab = n->getTwoChar( iu < in ? iu : in ,  iu < in ? in : iu  ); 
    }
    else if( n->upper > 1 ) // more than one uppercase : form abbrev from first two uppercase chars 
    { 
        ab = n->getFirstUpper(2) ; 
    }
    else 
    {
        ab = n->getFirst(2) ; 
    }

    LOG(LEVEL) 
        << " n.s [" << n->s << "]" 
        << " ab [" << ab << "]" 
        ;

    if(!isFree(ab))
    {
        ab = n->getFirstLast(); 
    } 


    SDice<26> rng ; 

    unsigned count = 0 ; 
    while(!isFree(ab) && count < 100)
    {
        ab = n->getTwoRandom(rng); 
        count += 1 ; 
    }

    return ab ; 
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




