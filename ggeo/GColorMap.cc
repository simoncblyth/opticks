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

#include <cassert>
#include <iostream>
#include <iomanip>

#include "BFile.hh"
#include "BMap.hh"

#include "GColorMap.hh"

#include "PLOG.hh"



GColorMap::GColorMap()
{
}

void GColorMap::addItemColor(const char* iname, const char* color)
{
     m_iname2color[iname] = color ; 
}

const char* GColorMap::getItemColor(const char* iname, const char* missing)
{
     // hmm maybe string data moved around as item/color are added to map, 
     // so best to query only after the map is completed for consistent pointers
     return m_iname2color.count(iname) == 1 ? m_iname2color[iname].c_str() : missing ; 
}



GColorMap* GColorMap::load(const char* dir, const char* name)
{
    assert(0);

    if(!BFile::ExistsFile(dir, name))
    {
        LOG(warning) << "GColorMap::load FAILED no file at  " << dir << "/" << name ; 
        return NULL ;
    }
    GColorMap* cm = new GColorMap ; 
    cm->loadMaps(dir, name);
    return cm ; 
}

void GColorMap::loadMaps(const char* idpath, const char* name)
{
    BMap<std::string, std::string>::load( &m_iname2color, idpath, name );  
}

void GColorMap::dump(const char* msg)
{
    LOG(info) << msg ;
    typedef std::map<std::string, std::string> MSS ; 
    for(MSS::iterator it=m_iname2color.begin() ; it != m_iname2color.end() ; it++ ) 
    {
         std::cout 
             << std::setw(25) << it->first 
             << std::setw(25) << it->second
             << std::endl ;  
    }
}




