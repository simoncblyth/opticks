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

#include "BFile.hh"
#include "BDir.hh"
#include <iostream>

#include "OPTICKS_LOG.hh"


typedef std::vector<std::string> VS ; 

void dump(const char* msg, std::vector<std::string>& names)
{
    std::cerr << msg << std::endl ; 
    for(VS::const_iterator it=names.begin() ; it != names.end() ; it++ ) std::cerr << *it << std::endl ; 
}

void test_dirlist()
{
    LOG(info); 
    std::string home = BFile::FormPath("~");
    const char* dir = home.c_str();

    VS names ;
    BDir::dirlist(names, dir);
    dump("all", names);
     
    names.clear();
    BDir::dirlist(names, dir, ".ini" );
    dump(".ini", names);

    names.clear();
    BDir::dirlist(names, dir, ".json" );
    dump(".json", names);
}

void test_dirdirlist_0()
{
    LOG(info); 
    std::string home = BFile::FormPath("~");
    const char* dir = home.c_str();

    VS names ;
    BDir::dirdirlist(names, dir ); 
    dump("all", names);
}

void test_dirdirlist_1()
{
    LOG(info); 
    std::string home = BFile::FormPath("~");
    const char* dir = home.c_str();

    const char* name_suffix = "_en" ; 
    bool endswith = true ; 

    VS names ;
    BDir::dirdirlist(names, dir, name_suffix, endswith ); 
    dump("dirs with names ending _en", names);

    endswith = false ; 
    names.clear();
    BDir::dirdirlist(names, dir, name_suffix, endswith ); 
    dump("dirs with names NOT ending _en", names);

}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //test_dirlist();    
    //test_dirdirlist_0(); 
    test_dirdirlist_1(); 

    return 0 ; 
}
