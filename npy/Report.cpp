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

#include <algorithm>
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>

// brap-
#include "BTime.hh"
#include "BFile.hh"

// npy-
#include "Report.hpp"

#include "PLOG.hh"

const char* Report::NAME = "report.txt" ;
const char* Report::TIMEFORMAT = "%Y%m%d_%H%M%S" ;


Report::Report()
{
}

void Report::add(const VS& lines)
{
    for(VS::const_iterator it=lines.begin() ; it!=lines.end() ; it++) m_lines.push_back(*it);
}



void Report::save(const char* dir)
{
    save(dir, NAME);
}

Report* Report::load(const char* dir)
{
    Report* report = new Report ; 
    report->load(dir, NAME);
    return report ; 
}

void Report::load(const char* dir, const char* name_)
{
    std::string path = BFile::preparePath(dir, name_, true);
    if(path.empty()) return ; 

    LOG(debug)<<"Report::load from " << path ; 
    std::ifstream ifs(path.c_str()) ;
    //std::copy(std::istream_iterator<std::string>(ifs), std::istream_iterator<std::string>(), std::back_inserter(m_lines));
    std::string line;
    while ( std::getline(ifs, line) )
    {
        if (line.empty()) continue;
        m_lines.push_back(line);
    }
}

void Report::save(const char* dir, const char* name_)
{
    std::string path = BFile::preparePath(dir, name_, true);
    if(path.empty()) return ; 

    LOG(debug)<<"Report::save to " << path ; 
    std::ofstream ofs(path.c_str()) ;
    std::copy( m_lines.begin(), m_lines.end(), std::ostream_iterator<std::string>(ofs, "\n"));

}

void Report::dump(const char* msg)
{
    LOG(info) << msg ; 
    for(unsigned int i=0 ; i < m_lines.size() ; i++ ) std::cout << m_lines[i] << std::endl ; 
}


std::string Report::timestamp()
{
    std::string timestamp =  BTime::now(Report::TIMEFORMAT,0);
    return timestamp ; 
}

std::string Report::name(const char* typ, const char* tag)
{
    std::stringstream ss ; 
    ss << Report::timestamp() 
       << "_"
       << typ
       << "_"
       << tag
       << ".txt"
       ;

    return ss.str() ;
}


