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
#include <climits>

#include "BFile.hh"

#include "BTxt.hh"
#include "PLOG.hh"

#ifdef _MSC_VER
#define strdup _strdup
#endif



BTxt* BTxt::Load(const char* path)
{
    std::string p = BFile::FormPath(path); 
    BTxt* txt = new BTxt(p.c_str());
    txt->read(); 
    return txt ; 
}








BTxt::BTxt(const char* path)
   :
   m_path(path ? strdup(path) : NULL)
{
}

const std::vector<std::string>& BTxt::getLines() const 
{
    return m_lines ; 
}



std::string BTxt::desc() const 
{  
    std::stringstream ss ;
    ss << "BTxt " 
       << " path " << m_path  
       << " NumLines " << getNumLines()
       ;
    return ss.str();
}


const std::string& BTxt::getString(unsigned int num) const
{
    return m_lines[num]; 
}

const char* BTxt::getLine(unsigned int num) const 
{
   return num < m_lines.size() ? m_lines[num].c_str() : NULL ; 
}
unsigned int  BTxt::getNumLines() const 
{
   return m_lines.size() ; 
}

void BTxt::dump(const char* msg) const 
{
    unsigned n = getNumLines() ;
    LOG(info) << msg << " NumLines " << n ; 
    for(unsigned i=0 ; i < n ; i++) std::cout << getLine(i) << std::endl ; 
}


unsigned int BTxt::getIndex(const char* line) const 
{
   std::string s(line);
   for(unsigned int i=0 ; i < m_lines.size() ; i++) if(m_lines[i].compare(s)==0) return i ;
   return UINT_MAX ; 
}


void BTxt::read()
{
    std::ifstream in(m_path, std::ios::in);
    if(!in.is_open()) 
    {   
        LOG(fatal) << "BTxt::read failed to open " << m_path ; 
        return ;
    }   

    std::string line ; 
    while(std::getline(in, line)) 
    {   
         m_lines.push_back(line);
    }   
    in.close();

    LOG(debug) << "BTxt::read " 
              << " path " << m_path 
              << " lines " << m_lines.size() 
              ;   

}


void BTxt::prepDir(const char* path_) const 
{
    const char* path = path_ ? path_ : m_path ; 
    std::string pdir = BFile::ParentDir(path);
    BFile::CreateDir(pdir.c_str()); 

    LOG(debug) << "BTxt::prepDir"
              << " pdir " << pdir
              ;
}

void BTxt::write(const char* path_) const 
{
    const char* path = path_ ? path_ : m_path ; 

    assert(path); 

    std::string p = BFile::preparePath(path) ;  

    //prepDir(path);

    std::ofstream out(p.c_str(), std::ios::out);
    if(!out.is_open()) 
    {   
        LOG(fatal) << "failed to open " << p.c_str() ; 
        return ;
    }   


    for(VS_t::const_iterator it=m_lines.begin() ; it != m_lines.end() ; it++)
    {
        out << *it << std::endl ; 
    }

    out.close();
}

void BTxt::addLine(const std::string& line)
{
    addLine(line.c_str());
}
void BTxt::addLine(const char* line)
{
    m_lines.push_back(line);
}

template<typename T>
void BTxt::addValue(T value)
{
    std::stringstream ss ; 
    ss << value ; 
    std::string s = ss.str(); 
    m_lines.push_back(s);
}




template BRAP_API void BTxt::addValue(int );
template BRAP_API void BTxt::addValue(unsigned );
template BRAP_API void BTxt::addValue(float );
template BRAP_API void BTxt::addValue(double );



