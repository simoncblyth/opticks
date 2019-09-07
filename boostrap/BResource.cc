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

#include "BResource.hh"
#include "BFile.hh"
#include "PLOG.hh"

BResource* BResource::INSTANCE = NULL ; 

const BResource* BResource::GetInstance()
{
    return INSTANCE ; 
}

void BResource::Dump(const char* msg)
{
    LOG(info) << msg ;

    const BResource* br = INSTANCE ;  

    if(!br)
    {
        LOG(fatal) << " No BResource::INSTANCE " ; 
        return ; 
    } 

    br->dumpNames("names");
    br->dumpDirs("dirs");
    br->dumpPaths("paths");
}


const char* BResource::GetPath(const char* label){ return INSTANCE ? INSTANCE->getPath(label) : NULL ; }
const char* BResource::GetDir(const char* label){  return INSTANCE ? INSTANCE->getDir(label) : NULL ; }
const char* BResource::GetName(const char* label){ return INSTANCE ? INSTANCE->getName(label) : NULL ; }

void BResource::SetPath(const char* label, const char* value){ if(!INSTANCE) INSTANCE=new BResource ; INSTANCE->setPath(label, value) ; }
void BResource::SetDir(const char* label, const char* value){  if(!INSTANCE) INSTANCE=new BResource ; INSTANCE->setDir(label, value) ; }
void BResource::SetName(const char* label, const char* value){ if(!INSTANCE) INSTANCE=new BResource ; INSTANCE->setName(label, value) ; }


BResource::BResource()
{
    INSTANCE=this ; 
}

BResource::~BResource()
{
}

const char* BResource::getPath(const char* label) const { return get(label, m_paths); }
const char* BResource::getDir(const char* label) const { return get(label, m_dirs); }
const char* BResource::getName(const char* label) const { return get(label, m_names); }

bool BResource::hasPath(const char* label) const { return count(label, m_paths) > 0u ;  }
bool BResource::hasDir(const char* label) const { return count(label, m_dirs) > 0u ;  }
bool BResource::hasName(const char* label) const { return count(label, m_names) > 0u ;  }

void BResource::setPath( const char* label, const char* value ){   set(label, value, m_paths) ; }
void BResource::setDir( const char* label, const char* value ){    set(label, value, m_dirs) ; }
void BResource::setName( const char* label, const char* value ){   set(label, value, m_names) ; }

const char* BResource::get(const char* label, const std::vector<std::pair<std::string, std::string>>& vss) // static  
{
    typedef std::pair<std::string, std::string> SS ; 
    typedef std::vector<SS> VSS ; 

    const char* path = NULL ; 
 
    for(VSS::const_iterator it=vss.begin() ; it != vss.end() ; it++)
    {
        const SS& ss = *it ;
        if(ss.first.compare(label) == 0) 
        {
            path = ss.second.c_str() ; 
        }
    }
    return path ; 
}



void BResource::set(const char* label, const char* value, std::vector<std::pair<std::string, std::string>>& vss) // static  
{
    if(count(label, vss) == 0u) 
    {
        add(label, value, vss ); 
    }
    else
    {
        typedef std::pair<std::string, std::string> SS ; 
        typedef std::vector<SS> VSS ; 

        for(VSS::iterator it=vss.begin() ; it != vss.end() ; it++)
        {
            SS& ss = *it ;
            if(ss.first.compare(label) == 0)  
            {
                LOG(info) << label << " change " << ss.second << " to " << value ;   
                ss.second = value ;    
            }
        }
    }
}


unsigned BResource::count(const char* label, const std::vector<std::pair<std::string, std::string>>& vss) // static  
{
    typedef std::pair<std::string, std::string> SS ; 
    typedef std::vector<SS> VSS ; 

    unsigned count(0); 
    for(VSS::const_iterator it=vss.begin() ; it != vss.end() ; it++)
    {
        const SS& ss = *it ;
        if(ss.first.compare(label) == 0)  count += 1 ; 
    }
    return count ; 
}


void BResource::addName( const char* label, const char* value) { add( label, value, m_names ) ; } 
void BResource::addPath( const char* label, const char* value) { add( label, value, m_paths ) ; } 
void BResource::addDir(  const char* label, const char* value) { add( label, value, m_dirs ) ; } 


void BResource::add( const char* label, const char* value,  std::vector<std::pair<std::string, std::string>>& vss   )
{
    assert(count(label,vss) == 0u);     
    typedef std::pair<std::string, std::string> SS ; 
    vss.push_back( SS(label, value ? value : "") );
}


void BResource::dumpNames(const char* msg) const 
{
    LOG(info) << msg ; 

    typedef std::pair<std::string, std::string> SS ; 
    typedef std::vector<SS> VSS ; 

    for(VSS::const_iterator it=m_names.begin() ; it != m_names.end() ; it++)
    {
        const char* label = it->first.c_str() ; 
        const char* name = it->second.empty() ? NULL : it->second.c_str() ; 
        std::cerr
             << std::setw(30) << label
             << " : " 
             << std::setw(2) << "-" 
             << " : " 
             << std::setw(50) << ( name ? name : "-" )
             << std::endl 
             ;
    }
}

void BResource::dumpPaths(const char* msg) const 
{
    LOG(info) << msg ; 

    typedef std::pair<std::string, std::string> SS ; 
    typedef std::vector<SS> VSS ; 

    for(VSS::const_iterator it=m_paths.begin() ; it != m_paths.end() ; it++)
    {
        const char* name = it->first.c_str() ; 
        const char* path = it->second.empty() ? NULL : it->second.c_str() ; 

        bool exists = path ? BFile::ExistsFile(path ) : false ; 

        const char* path2 = getPath(name) ; 

        std::cerr
             << std::setw(30) << name
             << " : " 
             << std::setw(2) << ( exists ? "Y" : "N" ) 
             << " : " 
             << std::setw(50) << ( path ? path : "-" )
             << std::endl 
             ;


        bool match = path2 == path ;         
        if(!match) LOG(fatal) 
                    << " path [" << path << "] " 
                    << " path2 [" << path2 << "] "
                    ;

        assert( match );
    } 
}

void BResource::dumpDirs(const char* msg) const 
{
    LOG(info) << msg ; 

    typedef std::pair<std::string, std::string> SS ; 
    typedef std::vector<SS> VSS ; 

    for(VSS::const_iterator it=m_dirs.begin() ; it != m_dirs.end() ; it++)
    {
        const char* name = it->first.c_str() ; 
        const char* dir = it->second.empty() ? NULL : it->second.c_str() ; 
        bool exists = dir ? BFile::ExistsDir(dir ) : false ; 

        std::cerr
             << std::setw(30) << name
             << " : " 
             << std::setw(2) << ( exists ? "Y" : "N" ) 
             << " : "  
             << std::setw(50) << ( dir ? dir : "-") 
             << std::endl 
             ;
    } 
}


