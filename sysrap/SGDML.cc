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

#include <string>
#include <sstream>
#include <algorithm>
#include "SGDML.hh"


std::string SGDML::GenerateName(const char* name, const void* const ptr, bool addPointerToName )
{
    std::stringstream ss; 
    ss << name;
    if(addPointerToName) ss << ptr ; 
    std::string nameOut = ss.str();

    if(nameOut.find(' ') != std::string::npos)
         nameOut.erase(std::remove(nameOut.begin(),nameOut.end(),' '),nameOut.end());

    //  std::remove 
    //         Removes all elements satisfying specific criteria from the range [first, last) 
    //         and returns a past-the-end iterator for the new end of the range.
    //

    return nameOut;
}



// after G4GDMLRead::Strip
std::string SGDML::Strip(const std::string& name)  // static 
{
    std::string sname = name.substr(0, name.find("0x")) ;
    return sname ;
}


std::string SGDML::Strip(const char* name_)  // static 
{
    std::string name(name_); 
    std::string sname = name.substr(0, name.find("0x")) ;
    return sname ;
}






