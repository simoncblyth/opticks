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

#include <sstream>
#include <iomanip>
#include <cassert>
#include <cstring>
#include <csignal>
#include <vector>

#include "SAr.hh"
#include "SSys.hh"
#include "SDigest.hh"
#include "SStr.hh"
#include "SPath.hh"
#include "SOpticksKey.hh"

#include "PLOG.hh"

const plog::Severity SOpticksKey::LEVEL = PLOG::EnvLevel("SOpticksKey", "DEBUG") ; 

SOpticksKey* SOpticksKey::fKey = NULL ; 

const char* SOpticksKey::G4LIVE = "g4live" ; 
const char* SOpticksKey::IDSTEM = "g4ok" ; 
const char* SOpticksKey::IDFILE = "g4ok.gltf" ; 
const char* SOpticksKey::IDSUBD = "g4ok_gltf" ; 
int         SOpticksKey::LAYOUT = 1 ; 
const char* SOpticksKey::LAYOUT_ = "1" ; 


bool SOpticksKey::IsSet()  // static
{
    return fKey != NULL ; 
}


SOpticksKey* SOpticksKey::GetKey()
{
    // invoked by SOpticksResource::SOpticksResource at Opticks instanciation
    return fKey ; 
}

const char* SOpticksKey::StemName( const char* ext, const char* sep )
{
    return SStr::Concat(IDSTEM, sep, ext );
}


/**
SOpticksKey::SetKey
---------------------

Argument spec NULL is the normal case, signalling an 
attempt to get the spec from envvar OPTICKS_KEY.


**/

bool SOpticksKey::SetKey(const char* spec)
{
    if(SOpticksKey::IsSet())
    {
        LOG(LEVEL) << "key is already set, ignoring update with spec " << spec ;
        return true ;  
    }

    bool live = spec != NULL ;  ; 

    if(spec == NULL)
    {
        spec = SSys::getenvvar("OPTICKS_KEY");  
        LOG(LEVEL) << "from OPTICKS_KEY envvar " << spec ; 
    } 

    LOG(LEVEL) << " spec " << spec ; 

    fKey = spec ? new SOpticksKey(spec) : NULL  ; 

    if(fKey) 
    {
         LOG(LEVEL) << std::endl << fKey->desc() ; 
         fKey->setLive(live); 
    }

    return true ; 
}

void SOpticksKey::Desc()
{
    if(fKey) LOG(info) << std::endl << fKey->desc() ; 
}



bool SOpticksKey::IsLive() // static
{
    return fKey ? fKey->isLive() : false ; 
}
void SOpticksKey::setLive(bool live)
{
    m_live = live ; 
}
bool SOpticksKey::isLive() const 
{
    return m_live ; 
}

   


bool SOpticksKey::isKeySource() const  // current executable is geocache creator 
{
    return m_current_exename && m_exename && strcmp(m_current_exename, m_exename) == 0 ; 
}

SOpticksKey::SOpticksKey(const char* spec)
    :
    m_spec( spec ? strdup(spec) : NULL ),
    m_exename( NULL ),
    m_class( NULL ),
    m_volname( NULL ),
    m_digest( NULL ),
    m_idname( NULL ),
    m_idfile( StemName("gltf", ".") ),
    m_idgdml( StemName("gdml", ".") ),
    m_idsubd( IDSUBD ),
    m_layout( LAYOUT ),
    m_current_exename( SAr::Instance ? SAr::Instance->exename() : "OpticksEmbedded" ), 
    m_live(false)
{
    std::vector<std::string> elem ; 
    SStr::Split(spec, '.', elem ); 

    bool four = elem.size() == 4  ;
    if(!four) LOG(fatal) << " expecting 4 element spec delimited by dot " << spec ;  
    assert( four ); 
    
    m_exename = strdup(elem[0].c_str()); 
    m_class = strdup(elem[1].c_str()); 
    m_volname   = strdup(elem[2].c_str()); 
    m_digest = strdup(elem[3].c_str()); 

    assert( SDigest::IsDigest(m_digest) ); 

    std::stringstream ss ; 
    ss 
        << m_exename 
        << "_"
        << m_volname 
        << "_"
        << G4LIVE 
        ;

    std::string idname = ss.str();

    m_idname = strdup(idname.c_str()); 
}

const char* SOpticksKey::getSpec() const 
{
    return m_spec ;  
}

std::string SOpticksKey::export_() const 
{
    std::stringstream ss ; 
    ss   
        << "# SOpticksKey::export_ " 
        << "\n" 
        << "export OPTICKS_KEY=" << m_spec 
        << "\n" 
        ;    
    return ss.str();
}
const char* SOpticksKey::getExename() const 
{
    return m_exename ;  
}
const char* SOpticksKey::getClass() const 
{
    return m_class ;  
}
const char* SOpticksKey::getVolname() const 
{
    return m_volname ;  
}
const char* SOpticksKey::getDigest() const 
{
    return m_digest ;  
}
const char* SOpticksKey::getIdname() const 
{
    return m_idname ;  
}
const char* SOpticksKey::getIdfile() const 
{
    return m_idfile ;  
}
const char* SOpticksKey::getIdGDML() const 
{
    return m_idgdml ;  
}
const char* SOpticksKey::getIdsubd() const 
{
    return m_idsubd ;  
}
int SOpticksKey::getLayout() const 
{
    return m_layout ;  
}


std::string SOpticksKey::desc() const 
{
    std::stringstream ss ; 
    ss 
        << std::setw(25) << " SOpticksKey " << " : " << ( isKeySource() ? "KEYSOURCE" : " " ) << std::endl 
        << std::setw(25) << " spec (OPTICKS_KEY) " << " : " << m_spec    << std::endl 
        << std::setw(25) << " exename " << " : " << m_exename << std::endl 
        << std::setw(25) << " current_exename " << " : " << m_current_exename << std::endl 
        << std::setw(25) << " class "   << " : " << m_class   << std::endl 
        << std::setw(25) << " volname " << " : " << m_volname << std::endl 
        << std::setw(25) << " digest "  << " : " << m_digest  << std::endl 
        << std::setw(25) << " idname "  << " : " << m_idname  << std::endl 
        << std::setw(25) << " idfile "  << " : " << m_idfile  << std::endl 
        << std::setw(25) << " idgdml "  << " : " << m_idgdml  << std::endl 
        << std::setw(25) << " layout "  << " : " << m_layout  << std::endl 
        ;
    return ss.str(); 
}


const char* SOpticksKey::getIdPath(const char* base) const
{
    int create_dirs = 0 ;  // 0:noop
    return SPath::Resolve(base, m_idname, m_idsubd, m_digest, LAYOUT_, create_dirs ) ; 
}





