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
#include <sstream>
#include <fstream>
#include <cstring>

#include "BFile.hh"


#include "NJS.hpp"
#include "PLOG.hh"

#ifdef _MSC_VER
#define strdup _strdup
#endif





NJS::NJS()
   :
   m_js()
{
}

NJS::NJS(const NJS& other)
   :
   m_js(other.cjs())
{
}

NJS::NJS(const nlohmann::json& js)
   :
   m_js(js)
{
}

nlohmann::json& NJS::js()
{
    return m_js ; 
}  

const nlohmann::json& NJS::cjs() const 
{
    return m_js ; 
} 

 

void NJS::read(const char* path0, const char* path1)
{
    std::string path = BFile::FormPath(path0, path1);

    LOG(info) << "read from " << path ; 

    std::ifstream in(path.c_str(), std::ios::in);

    if(!in.is_open()) 
    {   
        LOG(fatal) << "NJS::read failed to open " << path ; 
        return ;
    }   
    in >> m_js ; 
}

void NJS::write(const char* path0, const char* path1) const 
{
    std::string path = BFile::FormPath(path0, path1);

    std::string pdir = BFile::ParentDir(path.c_str());

    BFile::CreateDir(pdir.c_str()); 

    LOG(info) << "write to " << path ; 

    std::ofstream out(path.c_str(), std::ios::out);

    if(!out.is_open()) 
    {   
        LOG(fatal) << "NJS::write failed to open" << path ; 
        return ;
    }   

    out << m_js ; 
    out.close();
}

void NJS::dump(const char* msg) const  
{
    LOG(info) << msg ; 
    std::cout << std::setw(4) << m_js << std::endl ; 
}



