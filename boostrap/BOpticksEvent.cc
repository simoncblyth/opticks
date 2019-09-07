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
#include <cstring>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <boost/algorithm/string/replace.hpp>


#include "BFile.hh"
#include "BResource.hh"
#include "BOpticksEvent.hh"

#include "PLOG.hh"


const plog::Severity BOpticksEvent::LEVEL = debug ; 

const char* BOpticksEvent::DEFAULT_DIR_TEMPLATE_NOTAG = "$OPTICKS_EVENT_BASE/$0/evt/$1/$2" ;  
const char* BOpticksEvent::DEFAULT_DIR_TEMPLATE       = "$OPTICKS_EVENT_BASE/$0/evt/$1/$2/$3" ; 
const char* BOpticksEvent::DEFAULT_DIR_TEMPLATE_RELATIVE = "$0/evt/$1/$2/$3" ;  // 
const char* BOpticksEvent::OVERRIDE_EVENT_BASE = NULL ; 

const int BOpticksEvent::DEFAULT_LAYOUT_VERSION = 2 ; 
int BOpticksEvent::LAYOUT_VERSION = 2 ; 


void BOpticksEvent::SetOverrideEventBase(const char* override_event_base)
{
   OVERRIDE_EVENT_BASE = override_event_base ? strdup(override_event_base) : NULL ; 
}
void BOpticksEvent::SetLayoutVersion(int version)
{
    LAYOUT_VERSION = version ; 
}
void BOpticksEvent::SetLayoutVersionDefault()
{
    LAYOUT_VERSION = DEFAULT_LAYOUT_VERSION ; 
}

void BOpticksEvent::Summary(const char* msg)
{
    LOG(info) << msg ; 
}

std::string BOpticksEvent::directory_template(bool notag)
{
    std::string deftmpl(notag ? DEFAULT_DIR_TEMPLATE_NOTAG : DEFAULT_DIR_TEMPLATE) ; 
    if(OVERRIDE_EVENT_BASE)
    {
       LOG(LEVEL) << "OVERRIDE_EVENT_BASE replacing OPTICKS_EVENT_BASE with " << OVERRIDE_EVENT_BASE ; 
       boost::replace_first(deftmpl, "$OPTICKS_EVENT_BASE/$0/evt", OVERRIDE_EVENT_BASE );
       LOG(LEVEL) << "deftmpl " << deftmpl ; 
    } 
    return deftmpl ; 
}

/**
BOpticksEvent::directory_
----------------------------

pfx 
    highest level directory name, eg "source"  

top (geometry)
    old and new: BoxInBox,PmtInBox,dayabay,prism,reflect,juno,... 

sub 
    old: cerenkov,oxcerenkov,oxtorch,txtorch   (constituent+source)
    new: cerenkov,scintillation,natural,torch  (source only)
    
tag
    old: tag did not contribute to directory 
    
anno
    normally NULL, used for example with metadata for a timestamp folder
    within the tag folder


When a tag is provided the DEFAULT_DIR_TEMPLATE yields::

    $OPTICKS_EVENT_BASE/{pfx}/evt/{top}/{sub}/{tag}

This is appended with ANNO when that is provided


**/

std::string BOpticksEvent::directory(const char* pfx, const char* top, const char* sub, const char* tag, const char* anno)
{
    bool notag = tag == NULL ; 
    std::string base = directory_template(notag);
    std::string base0 = base ;  

    replace(base, pfx, top, sub, tag) ; 

    std::string dir = BFile::FormPath( base.c_str(), anno  ); 

    LOG(LEVEL) 
        << " base0 " << base0 
        << " anno " << ( anno ? anno : "NULL" )
        << " base " << base 
        << " dir " << dir
        ; 

    return dir ; 
}

std::string BOpticksEvent::reldir(const char* pfx, const char* top, const char* sub, const char* tag )
{
    std::string base = DEFAULT_DIR_TEMPLATE_RELATIVE ; 
    replace(base, pfx, top, sub, tag) ; 
    return base ; 
}

/**
BOpticksEvent::replace
------------------------

Inplace replaces tokens in base argument. 


**/

void BOpticksEvent::replace( std::string& base , const char* pfx, const char* top, const char* sub, const char* tag )
{
    LOG(LEVEL) 
        << " pfx " << pfx
        << " top " << top
        << " sub " << sub
        << " tag " << ( tag ? tag : "NULL" )
        ; 
 
    if(pfx) boost::replace_first(base, "$0", pfx ); 
    boost::replace_first(base, "$1", top ); 
    boost::replace_first(base, "$2", sub ); 
    if(tag) boost::replace_first(base, "$3", tag ); 
}




std::string BOpticksEvent::path_(const char* pfx, const char* top, const char* sub, const char* tag, const char* stem, const char* ext)
{
    const char* anno = NULL ;  
    std::string dir = directory(pfx, top, sub, tag, anno);
    std::stringstream ss ; 
    ss << dir << "/" << stem << ext ;
    std::string path = ss.str();
    return path ; 
}





/**
BOpticksEvent::path
----------------------

Canonical usage from OpticksEvent::getPath::

    1862 const char* OpticksEvent::getPath(const char* xx)
    1863 {   
    1864     std::string name = m_abbrev.count(xx) == 1 ? m_abbrev[xx] : xx ;
    1865     const char* udet = getUDet(); // cat overrides det if present 
    1866     std::string path = BOpticksEvent::path(m_pfx, udet, m_typ, m_tag, name.c_str() );
    1867     return strdup(path.c_str()) ;
    1868 }


::

    $OPTICKS_EVENT_BASE/{pfx}/evt/{top}/{sub}/{tag}/{stem}{ext} 


Examples::

     /home/blyth/local/opticks/tmp/tboolean-proxy-19/evt/tboolean-proxy-19/torch/1/ox.npy
     /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1/source/evt/g4live/torch/-1/ox.npy


     OPTICKS_EVENT_BASE 
           /home/blyth/local/opticks/tmp
           /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1
     pfx                
           tboolean-proxy-19
           source
           scan-ph
    
           OpticksEvent::m_pfx  control via "--pfx" option 

     evt
           fixed marker

     top                
           tboolean-proxy-19
           g4live

           OpticksEvent::getUDet() providing  m_cat ? m_cat : m_det      control via "--cat" option 

     sub                
           torch               
           torch               

           identifies photon source, eg cerenkov, scintillation, natural, torch  

           OpticksEvent::m_typ 

     tag      
           1 

           non-zero integer string, default "1"

           OpticksEvent::m_tag

     stem               
           ox

           filename without extension eg "ox", "ph", "rx" 

           OpticksEvent::m_abbrev[xx]   where xx is array name

     ext   
           .npy 

           filename extension including the dot, eg ".txt" ".npy", defaults to ".npy"


**/



std::string BOpticksEvent::path(const char* pfx, const char* top, const char* sub, const char* tag, const char* stem, const char* ext)  // static 
{

    std::string p_ ; 
    if(LAYOUT_VERSION == 1)
    {
        // to work with 3-arg form for gensteps:  ("cerenkov","1","dayabay" )  top=dayabay sub=cerenkov tag=1 stem="" 
        assert( pfx == NULL );  
        assert( stem == NULL );  

        std::stringstream ss ; 
        ss << tag << ext ; 
        std::string name = ss.str();  // eg 1.npy 
    
        bool notag = false ;  
        std::string base = directory_template(notag);
         
        boost::replace_first(base, "$1", top ); 
        boost::replace_first(base, "$2", sub ); 
        boost::replace_first(base, "$3", name.c_str() ); 
          
        p_ = base ; 
    }  
    else if(LAYOUT_VERSION == 2)
    {
        const char* ustem = ( stem != NULL && stem[0] == '\0' ) ? "gs" : stem ;     
        // spring "gs" stem into life for argument compatibility with old layout : 
        // gensteps effectively has empty stem in old layout 
        p_ = path_(pfx, top, sub, tag, ustem, ext );
    }
    std::string p = BFile::FormPath( p_.c_str() ); 

     
    LOG(LEVEL)
          << " pfx " << pfx 
          << " top " << top 
          << " sub " << sub 
          << " tag " << tag 
          << " stem " << stem 
          << " ext " << ext 
          << " p " << p
          ;
    // eg top tboolean-box sub torch tag 1 stem so ext .npy p /tmp/blyth/opticks/evt/tboolean-box/torch/1/so.npy

    return p ; 
}


/**
BOpticksEvent::srctagdir
----------------------------

srcevtbase
     inside the geocache keydir eg:
     /usr/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/792496b5e2cc08bdf5258cc12e63de9f/1/source

**/

const char* BOpticksEvent::srctagdir( const char* det, const char* typ, const char* tag) // static
{
    const char* srcevtbase = BResource::GetDir("srcevtbase");   
    if( srcevtbase == NULL ) srcevtbase = BResource::GetDir("tmpuser_dir") ;   
    assert( srcevtbase ); 

    std::string path = BFile::FormPath(srcevtbase, "evt", det, typ, tag ); 
    //  source/evt/g4live/natural/1/        gs.npy

    return strdup(path.c_str()) ; 
}


